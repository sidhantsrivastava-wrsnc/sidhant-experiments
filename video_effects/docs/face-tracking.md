# Face Tracking

Face tracking data flows through the entire system — from zoom dampening in OpenCV effects to spatial layout hooks in Remotion components. The system detects faces via MediaPipe, builds spatial context (safe regions) for overlay placement, and provides per-frame face/zoom data to the rendering layer.

## Face Detection

**File:** `helpers/face_tracking.py`

### `detect_faces(video_path, active_ranges, total_frames, stride=None)`

Returns `list[tuple[cx, cy, fw, fh]]` — one entry per frame, in pixel coordinates.

**Pipeline:**
1. Spawn ffmpeg decode pipe (single-process, avoids macOS fork issues)
2. Detect faces at stride intervals (default: every 3 frames, from `VFX_FACE_DETECTION_STRIDE`)
3. Interpolate skipped frames via numpy linear interpolation
4. Fill gaps between active ranges with last detected face

**MediaPipe configuration:**
- Model: `face_landmarker.task` (path from `VFX_FACE_LANDMARKER_PATH`)
- Running mode: `VIDEO` (temporal consistency)
- Max faces: 1
- Detection confidence: 0.5
- Presence confidence: 0.5
- Tracking confidence: 0.5

**Landmarks used:**
- Index 4: Nose tip → center (cx, cy)
- Index 454 & 234: Left/right cheek → face width
- Index 152 & 10: Top/bottom → face height

### `smooth_data(data, alpha=None)`

Exponential moving average filter for jitter reduction.

```
output[i] = alpha × input[i] + (1 - alpha) × output[i-1]
```

Default alpha: `VFX_SMOOTHING_ALPHA` (0.1). Returns int32 numpy array.

### Dimension Probing

- `_probe_video(path)` — ffprobe for coded dimensions + fps
- `_probe_decoded_size(path)` — ffmpeg showinfo filter for actual decoded dimensions (handles rotation metadata)

## Face Tracking Cache

**Location:** `{cache_dir}/face_tracking_zoom.json`

### Format

```json
{
  "dimensions": [1920, 1080],
  "face_data": [
    [960, 540, 200, 250],
    [962, 541, 201, 251]
  ]
}
```

Each entry: `[center_x, center_y, face_width, face_height]` in pixel coordinates.

### Invalidation

Cache is rebuilt when:
- Cached `dimensions` don't match current decoded dimensions (rotation change, resize)
- Old list format (pre-dimensions tracking) is detected
- Cache file is missing

Validation in `zoom.py`:
```python
cached_dims = raw.get("dimensions")
if cached_dims == [dec_w, dec_h]:
    # Cache hit
else:
    # Stale — regenerate
```

## Spatial Context Building

**File:** `activities/remotion.py` → `vfx_build_remotion_context` (G8a)

Transforms raw face tracking data into windowed summaries that the LLM can use for overlay placement.

### Face Windows

Face data is averaged over 3-second windows:

```python
face_region = {
    "x": avg_cx - avg_fw / 2,   # normalized to [0, 1]
    "y": avg_cy - avg_fh / 2,
    "w": avg_fw,
    "h": avg_fh,
}
```

Each window spans `(start_time, end_time)` and includes `safe_regions`.

### Safe Regions

**Function:** `_compute_safe_regions(face)`

Given a face bounding box (normalized 0–1), computes non-overlapping screen regions where overlays won't occlude the speaker:

```
┌───────────────────────────────┐
│           TOP region          │  if face.y > 0.25
├──────┬──────────┬─────────────┤
│      │          │             │
│ LEFT │   FACE   │    RIGHT    │
│      │  (5% pad)│             │
│      │          │             │
├──────┴──────────┴─────────────┤
│         BOTTOM region         │  if face.bottom < 0.75
└───────────────────────────────┘
```

**Thresholds:**
- Left/right: Only if face is between 15%–85% horizontally
- Top: Only if face is below 25% vertically
- Bottom: Only if face is above 75% vertically
- Padding: 5% (0.05) around face in all directions

### Zoom State Export

**Function:** `export_zoom_state()` in `effects/zoom.py`

Exports per-frame zoom state as sparse JSON (only frames where zoom > 1.0):

```json
{
  "frames": {
    "42": [1.5, 0.45, 0.52],
    "105": [1.8, 0.47, 0.50]
  }
}
```

Each value: `[zoom_level, target_norm_x, target_norm_y]`.

### Output (spatial_context dict)

```python
{
    "video": { "width", "height", "fps", "duration", "total_frames" },
    "transcript": { "full_text", "segments" },
    "face_windows": [
        {
            "start_time": float,
            "end_time": float,
            "face_region": { "x", "y", "w", "h" },
            "safe_regions": [{ "x", "y", "w", "h", "label" }],
        }
    ],
    "opencv_effects": [...],
    "face_data_path": str,
    "zoom_state_path": str,
}
```

## Face-Aware Layout Hooks

**File:** `remotion/src/lib/spatial.ts`

### `useFaceAwareLayout(staticBounds, anchor?)`

Returns `{ left, top, scale, maxWidth, maxHeight }` in pixels.

**Anchor modes:**

| Mode | Behavior |
|------|----------|
| `static` | Use staticBounds as-is, apply face avoidance if overlapping |
| `face-right` | Position right of face: `cx + fw/2 + 0.03` |
| `face-left` | Position left of face: `cx - fw/2 - 0.03 - compW` |
| `face-below` | Position below face: `cy + fh/2 + 0.03` |
| `face-above` | Position above face: `cy - fh/2 - 0.03 - compH` |
| `face-beside` | Auto-pick left or right based on available space |

**Face scaling:** Scale proportional to face width, normalized to 0.25 baseline:
```typescript
const faceScale = Math.min(1.4, Math.max(0.6, face.fw / 0.25));
```

**Spring animation:** 15-frame ease-in from static position to face-relative position.

**Bounds clamping:** Elements kept within `[0.02, 0.98]` normalized range.

### `useFaceAvoidance(myBounds)`

Computes overlap between component and face. If overlap > 1%, pushes component away:

```typescript
const strength = overlap * 0.3;
// Apply offset proportional to overlap, animated with spring
```

## Zoom Compensation

**File:** `remotion/src/lib/spatial.ts` → `useZoomCompensation()`

When OpenCV zoom effects are active, overlay positions must be adjusted so they track with the zoomed content rather than staying fixed on screen.

The hook replicates the OpenCV affine transformation:

```
adjusted_x = zoom × normX + (0.5 - tx × zoom)
adjusted_y = zoom × normY + (0.5 - ty × zoom)
adjusted_scale = zoom
```

Where `zoom`, `tx`, `ty` come from `useZoomFrame()`.

### Zoom Dampening in OpenCV

When the zoom effect uses `tracking="face"`, face positions are dampened to prevent jitter amplification:

```python
dampen = 1.0 / max(current_zoom, 1.001)
damp_fx = dampen * fx + (1 - dampen) * damp_fx
```

Higher zoom → heavier smoothing (dampening factor approaches 0).

## Data Loading in Remotion

**File:** `remotion/src/DynamicComposition.tsx`

Face data and zoom state are loaded asynchronously on mount:

1. **Face data** (`faceDataPath`): Array of `[cx, cy, fw, fh]` per frame → `FaceFrame[]`
2. **Zoom state** (`zoomStatePath`): `{frames: {idx: [zoom, tx, ty]}}` → `Map<number, ZoomFrame>`

Both use `delayRender`/`continueRender` to ensure data is available before rendering starts.

Data is provided via React context:
- `FaceDataProvider` + `useFaceFrame()` → `FaceFrame | null`
- `ZoomDataProvider` + `useZoomFrame()` → `ZoomFrame` (default: `{zoom: 1, tx: 0.5, ty: 0.5}`)

## Face Detection in Blur Effect

**File:** `effects/blur.py`

The `face_pixelate` blur mode has its own face detection (per-frame, not cached):

- **Preferred:** MediaPipe `FaceDetection` (model_selection=1, confidence=0.5)
- **Fallback:** OpenCV Haar cascade (`haarcascade_frontalface_default.xml`)

Pixelation: downscale ROI to `max(2, int(radius/3))` pixels, then upscale back with nearest-neighbor interpolation.
