# Effects Pipeline

The OpenCV effects pipeline applies frame-level transformations in a single pass — decode, process, encode — with no intermediate files.

## Single-Pass Architecture

```
Input Video (may be HDR)
    │
    ▼
ffmpeg decode pipe (RGB24, with tone-mapping if HDR)
    │
    ▼
┌─────────────────────────────┐
│   Frame Processing Loop     │
│                             │
│  for frame_index in range(total_frames):│
│    if in active interval:   │
│      Phase  5: Vignette     │
│      Phase 10: Color        │
│      Phase 20: Blur         │
│      Phase 25: Whip         │
│      Phase 30: Zoom         │
│      Phase 40: Subtitle     │
│      Phase 50: Speed Ramp   │
│                             │
└─────────────┬───────────────┘
              │
              ▼
ffmpeg encode pipe (libx264, CRF 16, bt709)
              │
              ▼
         Output MP4
```

**Key files:**

- Pipeline orchestrator: `activities/apply_effects.py`
- Base effect interface: `effects/base.py`
- Phase ordering: `effect_registry.py`

## Phase Ordering

Effects execute in strict numeric order. Gaps allow inserting new phases without renumbering.

| Phase | Effect | Why this order |
|-------|--------|----------------|
| 5 | Vignette | Edge darkening applied before color grading |
| 10 | Color | Pixel-level grading, no geometry changes |
| 20 | Blur | Region blur before geometric transforms |
| 25 | Whip | Motion blur transition before zoom |
| 30 | Zoom | Geometric crop/scale (face-tracked) |
| 40 | Subtitle | Text on top of everything |
| 50 | Speed Ramp | Visual speed effect runs last |

The `group_by_phase()` function in `effect_registry.py` groups all active `EffectCue` objects by their phase number.

## Base Effect Interface

All effects implement the `BaseEffect` ABC from `effects/base.py`:

```python
class BaseEffect(ABC):
    @abstractmethod
    def setup(self, video_info: VideoInfo, effect_cues: list[EffectCue],
              *, cache_dir: str | None = None, video_path: str | None = None) -> None:
        """Pre-compute expensive data (face tracking, masks) once."""

    @abstractmethod
    def apply_frame(self, frame: np.ndarray, timestamp: float,
                    context: EffectContext) -> np.ndarray:
        """Apply effect to a single RGB24 frame. Returns modified frame."""
```

`EffectContext` provides per-frame metadata:

```python
@dataclass
class EffectContext:
    video_info: VideoInfo
    frame_index: int
    timestamp: float
    total_frames: int
```

Helper methods on `BaseEffect`: `is_active(timestamp)`, `get_active_cues(timestamp)`, `get_active_ranges()`.

## Frame Decoding

The `_FrameDecoder` class spawns an ffmpeg subprocess that outputs raw RGB24 frames via stdout pipe.

- Frame size: `width * height * 3` bytes
- Reshape: `(height, width, 3)` numpy array
- Contiguity enforced before encoding: `np.ascontiguousarray(frame)`

### HDR Tone-Mapping

HDR is detected by checking `color_transfer` (`arib-std-b67`, `smpte2084`) and `color_primaries` (`bt2020`).

When HDR is detected, the decode filter chain applies Hable tone-mapping:

```
zscale=t=linear:npl=100,format=gbrpf32le,
zscale=p=bt709,
tonemap=tonemap=hable:desat=0,
zscale=t=bt709:m=bt709:r=tv,
format=rgb24
```

Steps: linear light conversion → BT.709 primaries → Hable tone-map → BT.709 colorspace → RGB24.

## Encoder Settings

| Setting | Value |
|---------|-------|
| Codec | libx264 (H.264/AVC) |
| Preset | medium |
| CRF | 16 (high quality) |
| Pixel format | yuv420p |
| Color metadata | BT.709 (colorspace, primaries, transfer) |
| Range | tv (limited 16–235) |
| Flags | `-movflags +faststart` |
| Audio | `-an` (stripped; muxed later in G7) |

## Activity Structure

The pipeline is split into three Temporal activities:

1. **`vfx_prepare_render()`** — Probe dimensions, detect HDR, build render plan, compute active intervals
2. **`vfx_setup_processors()`** — Initialize processors with expensive setup (face tracking, segmentation); cache results to disk
3. **`vfx_render_video()`** — Re-create processors (loading from cache), run the full frame loop

The monolithic `apply_effects()` function does all three in one call (used for testing).

## Active Intervals

To avoid processing frames where no effects are active, the pipeline merges all effect time ranges into `active_intervals`. Frames outside these intervals pass through untouched.

---

## Effect Implementations

### Zoom

**File:** `effects/zoom.py`

Parameters (`ZoomParams`):
```python
tracking: Literal["face", "center", "point"] = "center"
zoom_level: float = Field(1.5, ge=1.0, le=3.0)
easing: Literal["smooth", "snap", "overshoot"] = "smooth"
action: Literal["bounce", "in", "out"] = "bounce"
motion_blur: float = Field(0.0, ge=0.0, le=1.0)
```

**Hold intervals:** Consecutive `in`/`out` cues are paired. Between the end of `in` and start of `out`, the zoom holds at peak level.

**Easing functions** (progress `t` ∈ [0, 1]):
- `bounce`: `sin(π·t)` — symmetric bell curve (0→1→0)
- `in`: `sin(π/2·t)` — ramp up (0→1)
- `out`: `cos(π/2·t)` — ramp down (1→0)

Each mode has `smooth`, `snap`, and `overshoot` variants.

**Face tracking:**
- Detects faces via MediaPipe FaceLandmarker (once per cue activation range)
- Dampening: `damp_pos = dampen × current + (1-dampen) × prev` where `dampen = 1/max(zoom, 1.001)`
- Higher zoom = heavier smoothing to prevent jitter amplification
- Results cached to `{cache_dir}/face_tracking_zoom.json` with dimension validation

**Affine transform:**
```
M = [[zoom, 0, w/2 - tx·zoom],
     [0, zoom, h/2 - ty·zoom]]
```
Applied via `cv2.warpAffine()` with `BORDER_REPLICATE`.

**Radial motion blur** (when `motion_blur > 0`):
- Extra zoom: `1.0 + blur_strength × 0.08`
- Blended: `cv2.addWeighted(original, 1-α, blurred, α, 0)`, α clamped to 0.7

### Blur

**File:** `effects/blur.py`

Parameters (`BlurParams`):
```python
blur_type: Literal["gaussian", "face_pixelate", "background", "radial"] = "gaussian"
radius: float = Field(15.0, ge=1.0)
target_region: TargetRegion  # normalized x, y, width, height
```

**Modes:**

| Mode | Implementation |
|------|---------------|
| `gaussian` | `cv2.GaussianBlur(roi, (k, k), 0)` on target region. Kernel: `int(radius)*2+1` |
| `face_pixelate` | Detect faces (MediaPipe or Haar cascade fallback), downscale→upscale ROI. Pixel size: `max(2, int(radius/3))` |
| `background` | MediaPipe SelfieSegmentation → mask → blur everything except person. Bilateral filter for edge refinement |
| `radial` | 6 samples at different zoom offsets from center, accumulated and blended. Spread: `0.05 × strength × base_zoom` |

### Color

**File:** `effects/color.py`

Parameters (`ColorParams`):
```python
preset: Literal["warm", "cool", "bw", "sepia", "dramatic", "custom"] = "warm"
intensity: float = Field(0.5, ge=0.0, le=1.0)
r_adjust: float = 0.0
g_adjust: float = 0.0
b_adjust: float = 0.0
```

**Presets** (RGB deltas scaled by intensity):

| Preset | R | G | B |
|--------|---|---|---|
| warm | +10 | -5 | -15 |
| cool | -15 | 0 | +15 |
| dramatic | -10 | -20 | 0 |
| sepia | Sepia kernel matrix, blended |
| bw | Grayscale conversion, blended |

### Whip

**File:** `effects/whip.py`

Parameters (`WhipParams`):
```python
direction: Literal["left", "right", "up", "down"] = "right"
intensity: float = Field(1.0, ge=0.3, le=2.0)
```

Builds a directional motion blur kernel. Intensity follows a bell curve: `sin(π·progress) × params.intensity`. Kernel size: `int(intensity × max(w, h) × 0.15)`. Applied via `cv2.filter2D()`.

### Vignette

**File:** `effects/vignette.py`

Parameters (`VignetteParams`):
```python
strength: float = Field(0.5, ge=0.1, le=1.0)
radius: float = Field(0.8, ge=0.3, le=1.0)
```

Pre-computes a normalized distance mask from center (cached by resolution). Falloff: `clip((dist - radius) / (1.4 - radius), 0, 1)`. Multiplies frame by `1.0 - falloff × strength`. Fades in/out at cue boundaries (0.5s or 1/4 of cue duration).

### Speed Ramp

**File:** `effects/speed_ramp.py`

Parameters (`SpeedRampParams`):
```python
speed: float = Field(2.0, ge=1.5, le=8.0)
easing: Literal["smooth", "snap"] = "smooth"
```

Visual-only effect (no time remapping). Smooth easing: `1.0 + (speed - 1) × sin(π·progress)`. Creates horizontal motion blur proportional to speed. Kernel size: `int(blur_amount × 30)`, alpha clamped to 0.6.

### Subtitle

**File:** `effects/subtitle.py`

Parameters (`SubtitleParams`):
```python
text: str = ""
font_size: int = 48
color: str = "#FFFFFF"
background_color: Optional[str] = "#000000CC"
position: Literal["bottom", "top", "center"] = "bottom"
bold: bool = True
```

Legacy burn-in using `cv2.FONT_HERSHEY_SIMPLEX`. When Remotion MG is enabled, subtitles are handled by the Remotion `Subtitles` component instead.
