# Zoom Follow Effect

## Overview

A video effect that smoothly zooms into a face, pans it to one side of the frame, fills the exposed background with an edge-fade, and optionally composites a tracking overlay (text, image, or video clip) next to the face. Implemented in `zoom_text.py`.

## When to Use

- Speaker introduction or emphasis moments in talking-head videos
- Drawing attention to a face while displaying a name/title/caption beside it
- Overlaying logos, badges, or animated assets that track with the face
- Creating dynamic "Ken Burns" style focus pulls on faces

## Function Signature

```python
create_zoom_follow_effect(
    input_path: str,            # Source video file path
    output_path: str,           # Output video file path
    zoom_max: float,            # Max zoom level (1.0 = no zoom, 1.5 = 50% zoom)
    t_start: float,             # Time (seconds) when zoom animation begins
    t_end: float,               # Time (seconds) when zoom animation completes
    face_side: str,             # "left" or "right" - where face lands on screen
    overlay_config: dict|None,  # Overlay config (None = no overlay)
    text_config: dict|None,     # Deprecated alias for overlay_config
)
```

## Parameter Reference

| Parameter | Type | Range | Default | Notes |
|-----------|------|-------|---------|-------|
| `zoom_max` | float | 1.0-2.0 | 1.5 | 1.0-1.15 = subtle, 1.15-1.3 = moderate, 1.3+ = dramatic. Values >1.5 risk cropping |
| `t_start` | float | 0+ | 0 | Seconds into video. Animation eases in via smoothstep |
| `t_end` | float | >t_start | 5 | Must be > t_start. Duration = t_end - t_start |
| `face_side` | str | "left"/"right" | "right" | Where the face is positioned after zoom |

### overlay_config Dictionary

The `overlay_config` dict controls what asset is composited and where it appears. The `type` key selects the overlay kind (defaults to `"text"` for backward compatibility).

#### Common keys (all overlay types)

| Key | Type | Required | Default | Options |
|-----|------|----------|---------|---------|
| `type` | str | No | "text" | "text", "image", "clip" |
| `position` | str | No | "left" | "left", "right", "top", "bottom" - relative to face |
| `margin` | float | No | 1.8 | Multiplier for distance from face edge. 1.0 = touching face bounding box |
| `t_start` | float | No | same as effect t_start | When overlay starts appearing (fades in) |
| `t_end` | float | No | same as effect t_end | When overlay stops being visible |

#### Text overlay keys (`type: "text"`)

| Key | Type | Required | Default | Options |
|-----|------|----------|---------|---------|
| `content` | str | Yes | "Text" | The text to display |
| `color` | str | No | "white" | Any CSS color name or hex |
| `fontsize` | int | No | 80 | Font size in pixels |
| `font` | str | No | "Arial-Bold" | Font name |

#### Image overlay keys (`type: "image"`)

| Key | Type | Required | Default | Notes |
|-----|------|----------|---------|-------|
| `path` | str | Yes | — | Path to PNG/image file. Alpha channel supported. |

#### Clip overlay keys (`type: "clip"`)

| Key | Type | Required | Default | Notes |
|-----|------|----------|---------|-------|
| `path` | str | Yes | — | Path to video file. Time is clamped to clip duration. |

### Backward Compatibility

The old `text_config` parameter still works. If `overlay_config` is not provided but `text_config` is, it's used as the overlay config. Since `type` defaults to `"text"`, existing code needs no changes:

```python
# Old style — still works
text_config={"content": "Hello", "position": "left", "color": "yellow"}

# New style — equivalent
overlay_config={"type": "text", "content": "Hello", "position": "left", "color": "yellow"}
```

## Critical Pairing Rule

**`face_side` and `overlay_config.position` must be on opposite sides.** The face occupies one side; the overlay goes on the other where there's space.

| face_side | overlay position | Result |
|-----------|-----------------|--------|
| "right" | "left" | Face right, overlay left |
| "left" | "right" | Face left, overlay right |
| "right" | "top"/"bottom" | Works but less common |
| "right" | "right" | Overlay overlaps face - BAD |

## Pipeline Integration

### Text overlay (default)

```python
from zoom_text import create_zoom_follow_effect

create_zoom_follow_effect(
    input_path="clip.mp4",
    output_path="clip_zoomed.mp4",
    zoom_max=1.1,
    t_start=1.0,
    t_end=6.0,
    face_side="right",
    overlay_config={
        "content": "Speaker Name",
        "position": "left",
        "color": "yellow"
    }
)
```

### Image overlay (logo/badge)

```python
create_zoom_follow_effect(
    input_path="clip.mp4",
    output_path="clip_logo.mp4",
    zoom_max=1.1,
    t_start=1.0,
    t_end=6.0,
    face_side="right",
    overlay_config={
        "type": "image",
        "path": "logo.png",
        "position": "left"
    }
)
```

### Clip overlay (animated asset)

```python
create_zoom_follow_effect(
    input_path="clip.mp4",
    output_path="clip_kinetic.mp4",
    zoom_max=1.1,
    t_start=1.0,
    t_end=6.0,
    face_side="right",
    overlay_config={
        "type": "clip",
        "path": "kinetic.mp4",
        "position": "left"
    }
)
```

### As a CLI Command

```bash
cd /path/to/cv_experiments
python -c "
from zoom_text import create_zoom_follow_effect
create_zoom_follow_effect('input.mp4', 'output.mp4', zoom_max=1.1, t_start=1.0, t_end=6.0, face_side='right', overlay_config={'content': 'Hello', 'position': 'left', 'color': 'yellow'})
"
```

### LLM Decision Guide

When choosing parameters from a prompt like "zoom into the speaker and show their name":

1. **zoom_max**: Default to 1.1 for subtle, 1.2 for noticeable. Only go higher if explicitly asked for dramatic zoom.
2. **t_start/t_end**: Match the segment timing. If processing a 10s clip, `t_start=0.5, t_end=3.0` gives a quick zoom-in that holds.
3. **face_side**: If an overlay is needed, put face on the opposite side from where it reads naturally. Default: face right, overlay left.
4. **overlay type**: Use "text" for names/captions, "image" for logos/badges, "clip" for animated assets.
5. **overlay color** (text only): "white" for dark backgrounds, "yellow" for general visibility.

## Dependencies

- `face_landmarker.task` model file must exist in the same directory as `zoom_text.py`
- Python packages: `opencv-python`, `numpy`, `mediapipe`, `moviepy`
- Environment: Use the venv at `/path/to/cv_experiments/.venv/`

## Constraints

- Processes one face only (first detected face)
- Output is always 24fps regardless of input fps
- Input must be a video file readable by OpenCV
- Static overlays (text, image) are rendered once and composited per-frame
- Clip overlays sample a new frame at each time step (clamped to clip duration)
- No audio pass-through by default (moviepy handles this via `write_videofile`)
- Processing is CPU-bound and slow (~2-5x realtime depending on resolution)

## Architecture Notes

The pipeline processes each frame in this order:
1. Calculate zoom geometry from smoothed face tracking data
2. Warp the raw video frame (affine transform)
3. Compute screen-space face coordinates from the warp matrix
4. Apply edge-fade background effect
5. Composite overlay LAST onto the warped frame (keeps overlay crisp and unwarped)

Overlays are always drawn post-warp in screen space. Never pre-warp.

### Overlay Factory

The `create_overlay(config)` factory returns an `Overlay` subclass based on `config["type"]`:

| Type | Class | Behavior |
|------|-------|----------|
| `"text"` | `TextOverlay` | Static — renders TextClip once, returns same frame every call |
| `"image"` | `ImageOverlay` | Static — loads image with alpha, returns same frame every call |
| `"clip"` | `ClipOverlay` | Animated — samples VideoFileClip at time t each call |

All overlays implement `get_frame(t) -> (rgb_array, alpha_mask)`.
