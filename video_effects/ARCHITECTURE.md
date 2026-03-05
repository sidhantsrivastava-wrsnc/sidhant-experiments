# Video Effects Pipeline — Architecture

## Overview

A Temporal-based video effects pipeline that analyzes spoken content in videos, uses an LLM (Claude) to generate context-aware visual effects (zoom, blur, color grading, subtitles), and applies them in a single-pass frame processing pipeline. A human-in-the-loop approval gate lets users review and iterate on the generated timeline before effects are rendered.

## System Components

| Component | File | Role |
|-----------|------|------|
| CLI | `cli.py` | Interactive entry point — launches workflow, polls for approval |
| Temporal Worker | `worker.py` | Hosts workflow + activities in a thread-pool executor |
| Workflow Orchestrator | `workflow.py` | 7-group activity pipeline with signal/query approval gate |
| Activities (7) | `activities/` | Discrete units of work: probe, extract, transcribe, parse, validate, apply, compose |
| Effect Processors (4) | `effects/` | Phase-ordered frame-level processors: Color, Blur, Zoom, Subtitle |
| LLM Integration | `helpers/llm.py` | Anthropic Claude API wrapper with structured tool-use output |
| Configuration | `config.py` | Pydantic-settings with `VFX_` env prefix |

## Data Flow

```
CLI Input (video path)
  │
  ▼
G1: Probe Video ──┐  (parallel)
G1: Extract Audio ─┘
  │
  ▼
G2: Transcribe Audio (ElevenLabs / Whisper)
  │
  ▼
┌─────────────────────────────────────┐
│  G3–G5 Loop  (max 5 retries)       │
│                                     │
│  G3: LLM Parse Effect Cues         │
│       ▼                             │
│  G4: Validate Timeline              │
│       ▼                             │
│  G5: CLI Approval (signal/query)    │
│       │                             │
│   Rejected? ── feedback ──► G3      │
│   Approved? ──────────────────►     │
└─────────────────────────────────────┘
  │
  ▼
G6: Apply Effects (single-pass decode → process → encode)
  │
  ▼
G7: Compose Final (mux original audio)
  │
  ▼
Output video
```

## Workflow Orchestration (`workflow.py`)

### Activity Groups

| Group | Activity | Timeout | Notes |
|-------|----------|---------|-------|
| **G1** | `vfx_get_video_info` + `vfx_extract_audio` | 10 min | Run in parallel via `asyncio.gather` |
| **G2** | `vfx_transcribe_audio` | 10 min | ElevenLabs API or local Whisper |
| **G3** | `vfx_parse_effect_cues` | 10 min | Claude via structured tool-use |
| **G4** | `vfx_validate_timeline` | 10 min | Clamp, dedupe, sort |
| **G5** | Signal wait | 10 min | `approve_timeline` signal / `get_timeline` query |
| **G6** | `vfx_apply_effects` | 30 min | 2-min heartbeat interval |
| **G7** | `vfx_compose_final` | 10 min | Stream-copy or AAC fallback |

### Signal / Query Mechanism

- **Signal** `approve_timeline(args: list)` — `args[0]`: bool (approved), `args[1]`: str (feedback)
- **Query** `get_timeline() → dict | None` — returns current validated timeline for CLI display

### Retry Logic

On rejection, user feedback is passed into the next G3 iteration. The loop runs up to `MAX_RETRIES = 5` times. If `auto_approve=True` is set in the workflow input, G5 is skipped entirely.

## Activities Reference

### G1: `vfx_get_video_info` (`activities/video.py`)

Runs `ffprobe -print_format json -show_format -show_streams` and returns a `VideoInfo` dict with resolution, FPS, duration, codec, audio info, and HDR color metadata (`color_space`, `color_transfer`, `color_primaries`, `pix_fmt`).

### G1: `vfx_extract_audio` (`activities/video.py`)

Produces two audio files:
- **Transcription copy**: 16 kHz mono PCM WAV (Whisper input format)
- **Preservation copy**: stream-copy of original audio track (no re-encoding); falls back to AAC if stream-copy fails

### G2: `vfx_transcribe_audio` (`activities/transcribe.py`)

Returns `{ transcript: str, segments: list[dict] }` where each segment is `{ text, start, end, type: "word" }`.

**Providers** (in priority order):
1. **ElevenLabs** (if `ELEVENLABS_API_KEY` set) — `scribe_v1` model, word-level timestamps
2. **Local Whisper** (fallback) — `base` model, `word_timestamps=True`

### G3: `vfx_parse_effect_cues` (`activities/parse_cues.py`)

Sends the timestamped transcript to Claude with a system prompt (`prompts/parse_effect_cues.md`) and extracts structured `EffectCue` objects via Anthropic tool-use. On retry iterations, includes the user's rejection feedback.

Returns `{ effects: list[dict], reasoning: str }`.

### G4: `vfx_validate_timeline` (`activities/validate.py`)

Pipeline:
1. Parse raw dicts to `EffectCue` objects
2. Clamp times to `[0, duration]`, enforce minimum 1.0 s duration
3. Remove effects with `confidence < 0.3`
4. Resolve same-type overlapping conflicts (keep highest confidence)
5. Sort by phase order then start time

### G6: `vfx_apply_effects` (`activities/apply_effects.py`)

Single-pass frame processing pipeline — see [Frame Processing Pipeline](#frame-processing-pipeline) below.

### G7: `vfx_compose_final` (`activities/compose.py`)

Muxes processed video with the preserved original audio:
- Primary: `ffmpeg -c:v copy -c:a copy` (stream-copy, fastest)
- Fallback: AAC re-encode if stream-copy fails
- No-audio: simple file copy

## Effects System

### Base Class Contract (`effects/base.py`)

```python
class BaseEffect(ABC):
    def setup(self, video_info: VideoInfo, effect_cues: list[EffectCue]) -> None: ...
    def apply_frame(self, frame: np.ndarray, timestamp: float, context: EffectContext) -> np.ndarray: ...
    def is_active(self, timestamp: float) -> bool: ...
    def get_active_cues(self, timestamp: float) -> list[EffectCue]: ...
```

`EffectContext` carries `video_info`, `frame_index`, `timestamp`, and `total_frames`.

### Phase Ordering (`effect_registry.py`)

Effects execute in deterministic phase order (lower = earlier):

| Phase | Effect | Rationale |
|-------|--------|-----------|
| 10 | **Color** | Pixel-level, no geometry change |
| 20 | **Blur** | Region processing before overlays |
| 30 | **Zoom** | Geometric transform (warpAffine) |
| 40 | **Subtitle** | Top-level text overlay |

### Effect Processors

#### ColorEffect (`effects/color.py`) — Phase 10

Presets: `warm` (+10R, -5G, -15B), `cool` (-15R, +0G, +15B), `dramatic` (-10R, -20G), `sepia` (kernel matrix), `bw` (grayscale).

`custom` mode applies per-channel RGB adjustments scaled by `intensity ∈ [0, 1]`.

#### BlurEffect (`effects/blur.py`) — Phase 20

| Mode | Method |
|------|--------|
| `gaussian` | `cv2.GaussianBlur` on target region |
| `face_pixelate` | MediaPipe face detection → downscale/upscale pixelation |
| `background` | MediaPipe segmentation mask → blur background, preserve person |
| `radial` | 6-sample multi-scale zoom blur from center |

#### ZoomEffect (`effects/zoom.py`) — Phase 30

Tracking modes: `face` (MediaPipe), `center`, `point`. Easing: `smooth` (cubic), `snap` (quick ramp), `overshoot`. Intensity ramps 0→100% over first 15% of duration, holds, then ramps down over last 15%. Uses `cv2.warpAffine` with `BORDER_REPLICATE`.

#### SubtitleEffect (`effects/subtitle.py`) — Phase 40

Renders text with `cv2.putText`. Supports configurable font size, hex color, background with alpha blending, and positioning (top/center/bottom).

## Frame Processing Pipeline (`activities/apply_effects.py`)

### Architecture

```
┌──────────────────────────────────────────────────────┐
│  FFmpeg Decode (pipe)                                │
│  ┌──────────────────────────────────────────────┐    │
│  │ HDR? → zscale(linear) → tonemap(hable) →     │    │
│  │         zscale(bt709) → format(rgb24)         │    │
│  │ SDR? → format(rgb24)                          │    │
│  └──────────────────────────────────────────────┘    │
└──────────────┬───────────────────────────────────────┘
               │ raw RGB24 frames
               ▼
┌──────────────────────────────────────────────────────┐
│  Per-Frame Loop (with active interval optimization)  │
│                                                      │
│  Color(10) → Blur(20) → Zoom(30) → Subtitle(40)     │
│                                                      │
│  Inactive frames pass through untouched              │
└──────────────┬───────────────────────────────────────┘
               │ processed RGB24 frames
               ▼
┌──────────────────────────────────────────────────────┐
│  FFmpeg Encode (pipe)                                │
│  libx264 · CRF 16 · yuv420p · BT.709 metadata       │
│  -movflags +faststart                                │
└──────────────────────────────────────────────────────┘
```

### HDR Detection & Tone-Mapping

HDR is detected when `color_transfer ∈ {arib-std-b67, smpte2084}` or `color_primaries == bt2020`.

Filter chain:
```
zscale=t=linear:npl=100,format=gbrpf32le,
zscale=p=bt709,
tonemap=tonemap=hable:desat=0,
zscale=t=bt709:m=bt709:r=tv,
format=rgb24
```

### Active Interval Optimization

Before the frame loop, all effect cue time ranges are merged into a sorted list of `(start_frame, end_frame)` intervals. Frames outside these intervals skip effect processing entirely, significantly reducing CPU cost for videos with sparse effects.

### Encoder Settings

| Parameter | Value |
|-----------|-------|
| Codec | libx264 |
| Preset | medium |
| CRF | 16 |
| Pixel format | yuv420p |
| Color space | BT.709 (colorspace, primaries, trc) |
| Color range | tv |
| Flags | +faststart |

## Schemas Reference

### Workflow I/O (`schemas/workflow.py`)

**`VideoEffectsInput`**: `input_video: str`, `output_video: str`, `auto_approve: bool = False`

**`VideoEffectsOutput`**: `output_video: str`, `effects_applied: int`, `transcript_length: int`, `phases_executed: int`, `error: Optional[str]`

### Video Metadata (`schemas/effects.py`)

**`VideoInfo`**: `width`, `height`, `fps`, `duration`, `codec`, `total_frames`, `audio_codec`, `has_audio`, `color_space`, `color_transfer`, `color_primaries`, `pix_fmt`

### Effect Models (`schemas/effects.py`)

**`EffectType`** enum: `ZOOM`, `BLUR`, `COLOR_CHANGE`, `SUBTITLE`

**`EffectCue`**: `effect_type`, `start_time`, `end_time`, `verbal_cue`, `confidence ∈ [0, 1]`, plus optional type-specific params:

| Params Model | Key Fields |
|-------------|------------|
| `ZoomParams` | `tracking` (face/center/point), `zoom_level` [0.1–3.0], `easing` (smooth/snap/overshoot) |
| `BlurParams` | `blur_type` (gaussian/face_pixelate/background/radial), `radius`, `target_region` |
| `ColorParams` | `preset` (warm/cool/bw/sepia/dramatic/custom), `intensity`, `r/g/b_adjust` |
| `SubtitleParams` | `text`, `font_size`, `color`, `background_color`, `position`, `bold` |

**`ValidatedTimeline`**: `effects: list[EffectCue]`, `conflicts_resolved: int`, `total_duration: float`

**`TargetRegion`**: `x`, `y`, `width`, `height` — all normalized [0–1]

## Configuration (`config.py`)

`VideoEffectsSettings` uses `pydantic-settings` with prefix `VFX_` and loads from `.env`.

| Variable | Default | Purpose |
|----------|---------|---------|
| `VFX_TASK_QUEUE` | `video_effects_queue` | Temporal task queue name |
| `VFX_TEMPORAL_NAMESPACE` | `default` | Temporal namespace |
| `VFX_TEMPORAL_ENDPOINT` | `localhost:7233` | Temporal server address |
| `VFX_TEMPORAL_API_KEY` | — | Temporal Cloud API key |
| `VFX_ANTHROPIC_API_KEY` | — | Anthropic Claude API key |
| `VFX_LLM_MODEL` | `claude-sonnet-4-20250514` | Claude model for cue parsing |
| `VFX_ELEVENLABS_API_KEY` | — | ElevenLabs transcription (optional) |
| `VFX_TEMP_DIR` | `/tmp/video_effects` | Working directory for intermediates |
| `VFX_FACE_LANDMARKER_PATH` | `cv_experiments/face_landmarker.task` | MediaPipe model path |
| `VFX_FACE_DETECTION_STRIDE` | `3` | Process every Nth frame for face detection |
| `VFX_SMOOTHING_ALPHA` | `0.1` | EMA smoothing for face tracking |

## Key Files Index

| File | Purpose |
|------|---------|
| `__main__.py` | Entry point (`python -m video_effects`) |
| `cli.py` | Argparse CLI, workflow launch, interactive approval loop |
| `worker.py` | Temporal worker setup with thread-pool executor (4 workers) |
| `workflow.py` | `VideoEffectsWorkflow` — 7-group orchestration with signal/query |
| `config.py` | `VideoEffectsSettings` singleton |
| `effect_registry.py` | `EFFECT_PHASES` dict + `group_by_phase()` helper |
| `schemas/__init__.py` | Schema re-exports |
| `schemas/workflow.py` | `VideoEffectsInput`, `VideoEffectsOutput` |
| `schemas/effects.py` | `EffectType`, `EffectCue`, `VideoInfo`, `ValidatedTimeline`, param models |
| `activities/__init__.py` | `ALL_VIDEO_EFFECTS_ACTIVITIES` list |
| `activities/video.py` | `get_video_info`, `extract_audio` (G1) |
| `activities/transcribe.py` | `transcribe_audio` (G2) |
| `activities/parse_cues.py` | `parse_effect_cues` (G3) |
| `activities/validate.py` | `validate_timeline` (G4) |
| `activities/apply_effects.py` | `apply_effects` (G6) — frame pipeline |
| `activities/compose.py` | `compose_final` (G7) |
| `effects/__init__.py` | Effect re-exports |
| `effects/base.py` | `BaseEffect` ABC, `EffectContext` dataclass |
| `effects/zoom.py` | `ZoomEffect` — face/center/point tracking with easing |
| `effects/blur.py` | `BlurEffect` — gaussian, face pixelate, background, radial |
| `effects/color.py` | `ColorEffect` — presets + custom RGB adjustments |
| `effects/subtitle.py` | `SubtitleEffect` — text overlay with positioning |
| `helpers/llm.py` | `call_structured()` — Anthropic tool-use wrapper |
| `helpers/face_tracking.py` | `detect_faces()`, `smooth_data()` — MediaPipe wrapper |
| `prompts/parse_effect_cues.md` | System prompt for LLM effect parsing |
| `prompts/schema.py` | `ParsedEffectCues` response model |
