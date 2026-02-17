# Zoom Bounce Effect

## Overview

An event-driven video effect that applies face-tracking zoom bounces, punch-in holds, whip transitions, and radial zoom blur to talking-head videos. Unlike `zoom_follow_effect` (which zooms in once and holds), this effect supports multiple independent zoom events across a video timeline — bounces that zoom in and back out, separate in/out events with hold periods, and post-processing effects like directional motion blur. Implemented in `zoom_bounce.py`.

## When to Use

- Emphasis punches on key phrases in talking-head videos (punch-in hold)
- Comedy/drama crash zooms (fast snap-in with overshoot)
- Energetic whip-pan transitions between segments
- Radial zoom blur for high-energy transitions
- Multiple zoom moments across a single video (LLM-driven from transcript timing)
- Stabilizing handheld footage with subtle face-centered crop

## Function Signature

```python
create_zoom_bounce_effect(
    input_path: str,
    output_path: str,
    zoom_max: float = 1.4,
    bounces: list = None,
    bounce_mode: str = "snap",
    face_side: str = "center",
    overlay_config: dict | None = None,
    text_config: dict | None = None,
    fade_mode: str = "band",
    stabilize: float = 0.0,
    stabilize_alpha: float = 0.02,
    debug_labels: bool = False,
)
```

## Parameter Reference

| Parameter | Type | Range | Default | Notes |
|-----------|------|-------|---------|-------|
| `zoom_max` | float | 1.0-2.0 | 1.4 | Global default peak zoom. Per-event `zoom` overrides this. |
| `bounces` | list | — | `[(1.0, 2.5)]` | List of event tuples or dicts (see Event Format below) |
| `bounce_mode` | str | `"smooth"` / `"snap"` / `"overshoot"` | `"snap"` | Global default easing. Per-event `ease` overrides this. |
| `face_side` | str | `"center"` / `"left"` / `"right"` | `"center"` | Where face lands on screen. `"center"` = pure zoom, no lateral shift. |
| `stabilize` | float | 0.0 or 1.01-1.1 | 0.0 | 0 = off. 1.03 = 3% crop for camera stabilization. Applied when no bounce is active. |
| `stabilize_alpha` | float | 0.01-0.1 | 0.02 | Smoothing alpha for stabilization face data. Lower = heavier smoothing. |
| `debug_labels` | bool | — | `False` | Draw active effect labels on frame (for testing) |
| `fade_mode` | str | `"band"` / `"average"` | `"band"` | Edge fade strategy (only applies when `face_side` != `"center"`) |

## Event Format (bounces list)

The `bounces` list accepts both legacy tuples and new dict events. They can be mixed freely.

### Legacy Tuple Format (backward compatible)

```python
(start, end)                    # uses global bounce_mode and zoom_max
(start, end, mode)              # per-bounce easing override
(start, end, mode, zoom_max)    # per-bounce easing + zoom override
```

All tuples produce a **bell curve** (0→1→0) — zoom in then immediately back out.

### Dict Event Format

```python
{"action": "...", "start": float, "end": float, ...}
```

#### action: "bounce"

Bell curve zoom (same as tuple format). Zooms in and back out within the time window.

| Key | Type | Required | Default | Notes |
|-----|------|----------|---------|-------|
| `action` | str | Yes | — | `"bounce"` |
| `start` | float | Yes | — | Start time in seconds |
| `end` | float | Yes | — | End time in seconds |
| `ease` | str | No | `bounce_mode` | `"smooth"`, `"snap"`, or `"overshoot"` |
| `zoom` | float | No | `zoom_max` | Peak zoom level for this bounce |

#### action: "in"

Attack ramp (0→1). Zooms in and **holds** at peak until a matching `"out"` event.

| Key | Type | Required | Default | Notes |
|-----|------|----------|---------|-------|
| `action` | str | Yes | — | `"in"` |
| `start` | float | Yes | — | When zoom-in begins |
| `end` | float | Yes | — | When zoom-in completes (holds at peak after this) |
| `ease` | str | No | `bounce_mode` | Easing for the attack ramp |
| `zoom` | float | No | `zoom_max` | Zoom level to hold at |

#### action: "out"

Release ramp (1→0). Zooms out from the held position. **Must follow a matching `"in"` event.**

| Key | Type | Required | Default | Notes |
|-----|------|----------|---------|-------|
| `action` | str | Yes | — | `"out"` |
| `start` | float | Yes | — | When zoom-out begins |
| `end` | float | Yes | — | When zoom-out completes (back to baseline) |
| `ease` | str | No | `bounce_mode` | Easing for the release ramp |

The `"out"` event automatically inherits the `zoom` level from its paired `"in"`.

#### action: "zoom_blur"

Radial zoom blur effect. Can overlap with bounces. Peaks at midpoint of the time window.

| Key | Type | Required | Default | Notes |
|-----|------|----------|---------|-------|
| `action` | str | Yes | — | `"zoom_blur"` |
| `start` | float | Yes | — | Start time |
| `end` | float | Yes | — | End time |
| `intensity` | float | No | 1.0 | 0.0-1.0, strength of the radial blur |
| `n_samples` | int | No | 8 | Number of zoom-offset samples. More = smoother but slower. |

#### action: "whip"

Directional motion blur simulating a fast whip-pan. Does not move the frame content.

| Key | Type | Required | Default | Notes |
|-----|------|----------|---------|-------|
| `action` | str | Yes | — | `"whip"` |
| `start` | float | Yes | — | Start time |
| `end` | float | Yes | — | End time |
| `direction` | str | No | `"h"` | `"h"` (horizontal) or `"v"` (vertical) |
| `intensity` | float | No | 1.0 | 0.0-1.0, blur strength |

### In/Out Pairing Rules

- Every `"in"` must be followed by a matching `"out"` (before another `"in"`)
- Double zoom-in raises `ValueError`
- Zoom-out without prior zoom-in raises `ValueError`
- Between `"in"` end and `"out"` start, zoom holds at peak (p=1.0)
- `"bounce"` events are self-contained and don't affect in/out state
- `"zoom_blur"` and `"whip"` are independent and can overlap anything

## Easing Modes

### Bell Curve (bounce action)

| Mode | Behavior |
|------|----------|
| `smooth` | `sin(pi*t)` — clean symmetric arc |
| `snap` | Fast quadratic attack, hold at peak, fast release — editorial punch |
| `overshoot` | Elastic — overshoots ~15%, springs back, then releases |

### Attack (in action, 0→1)

| Mode | Behavior |
|------|----------|
| `smooth` | `sin(pi/2 * t)` — gentle ramp to peak |
| `snap` | `(t/0.5)^2` clamped — quick quadratic attack |
| `overshoot` | Elastic attack, overshoots then settles at 1.0 |

### Release (out action, 1→0)

| Mode | Behavior |
|------|----------|
| `smooth` | `cos(pi/2 * t)` — gentle release from peak |
| `snap` | `1 - t^2` — quick quadratic drop |
| `overshoot` | Elastic release, undershoots then settles at 0.0 |

## Camera Stabilization

When `stabilize > 0` (e.g., `1.03`):

- **No bounce active (p=0):** Applies a subtle crop centered on heavily-smoothed face position. The crop absorbs handheld camera jitter. A value of `1.03` means 3% zoom crop.
- **Bounce active (p>0):** Blends smoothly from stabilization target to bounce target. The stabilization zoom is *replaced by* (not added to) the bounce zoom.
- **Face tracking dampening:** Regardless of `stabilize`, when zoomed in (p>0) the face tracking position blends toward the heavy-smoothed data. At p=1.0 (full zoom), tracking uses the very-smooth path to prevent jitter amplification.

## YouTube Effect Mapping

| YouTube Effect | How to Achieve |
|----------------|---------------|
| **Punch-in hold** (2nd camera) | `"in"` + `"out"` pair with 0.6-2.5s hold gap |
| **Crash zoom** | Short `"in"` (4-10 frames = 0.13-0.33s) with `"overshoot"` ease |
| **Whip transition** | `"whip"` action, 8-16 frames, `intensity: 0.8-1.0` |
| **Slow push-in** | Long `"in"` event (2-6s), `"smooth"` ease, low zoom (1.01-1.03) |
| **Zoom-blur transition** | `"zoom_blur"` action overlapping a bounce, 12-20 frames |

## Pipeline Integration

### Punch-in hold (LLM-driven from transcript)

```python
from zoom_bounce import create_zoom_bounce_effect

create_zoom_bounce_effect(
    input_path="clip.mp4",
    output_path="clip_punched.mp4",
    stabilize=1.03,
    bounces=[
        {"action": "in",  "start": 2.1, "end": 2.4, "ease": "snap",   "zoom": 1.4},
        {"action": "out", "start": 5.8, "end": 6.5, "ease": "smooth"},
    ],
)
```

### Crash zoom (comedy punch)

```python
create_zoom_bounce_effect(
    input_path="clip.mp4",
    output_path="clip_crash.mp4",
    bounces=[
        {"action": "in",  "start": 3.0, "end": 3.2, "ease": "overshoot", "zoom": 1.6},
        {"action": "out", "start": 4.5, "end": 5.0, "ease": "snap"},
    ],
)
```

### Multiple bounces across a video

```python
create_zoom_bounce_effect(
    input_path="clip.mp4",
    output_path="clip_multi.mp4",
    stabilize=1.03,
    bounces=[
        # Punch-in hold on first emphasis
        {"action": "in",  "start": 1.5, "end": 2.0, "ease": "snap",      "zoom": 1.4},
        {"action": "out", "start": 4.5, "end": 5.3, "ease": "smooth"},
        # Quick bounce on second emphasis
        {"action": "bounce", "start": 8.0, "end": 9.5, "ease": "overshoot", "zoom": 1.3},
        # Legacy tuple still works
        (12.0, 13.5, "smooth", 1.3),
    ],
)
```

### Whip + zoom blur combo

```python
create_zoom_bounce_effect(
    input_path="clip.mp4",
    output_path="clip_effects.mp4",
    bounces=[
        {"action": "bounce",    "start": 2.0, "end": 3.5, "ease": "smooth", "zoom": 1.3},
        {"action": "zoom_blur", "start": 2.2, "end": 3.3, "intensity": 0.8, "n_samples": 8},
        {"action": "whip",      "start": 5.0, "end": 5.5, "direction": "h", "intensity": 1.0},
    ],
)
```

### LLM Decision Guide

When choosing parameters from a transcript/prompt:

1. **Action type:** Use `"in"`/`"out"` pairs for emphasis phrases (hold during the phrase). Use `"bounce"` for quick one-shot punches. Use `"whip"` for segment transitions.
2. **Ease mode:** `"snap"` for editorial/punchy (default). `"smooth"` for gentle/cinematic. `"overshoot"` for comedy/playful.
3. **Zoom level:** 1.3 = subtle, 1.4 = noticeable, 1.5+ = dramatic. Match to content energy.
4. **In/out timing:** `"in"` duration 0.15-0.5s (fast attack). Hold gap = phrase duration. `"out"` duration 0.3-1.0s (slower release feels better).
5. **Cross-ease:** Mix easing modes freely — e.g., `"snap"` in + `"smooth"` out feels like a confident editorial cut.
6. **Stabilize:** Default to `1.03` for handheld footage. Set `0.0` for tripod/already-stabilized footage.
7. **face_side:** Use `"center"` (default) for pure zoom. Only use `"left"`/`"right"` if compositing an overlay on the opposite side.

## Constraints

- Processes one face only (first detected)
- Never changes video playback speed — audio stays in sync
- All effects are visual-only (zoom, blur, motion blur)
- `"in"`/`"out"` events must be properly paired
- Overlapping bounces: highest zoom wins per frame
- Processing speed: ~15-20 fps without effects, ~4-10 fps with zoom_blur (depends on n_samples)

## Dependencies

- `face_landmarker.task` model file must exist in the same directory as `zoom_bounce.py`
- Python packages: `opencv-python`, `numpy`, `mediapipe`, `moviepy`

## Architecture Notes

Per-frame pipeline:
1. Read frame from threaded video reader
2. Blend face tracking data toward heavy-smoothed data based on zoom intensity
3. Compute warp matrix (stabilization-aware if enabled)
4. Apply affine warp with BORDER_REPLICATE
5. Apply post-warp effects (zoom_blur, whip) if active
6. Apply gradient edge fade (only if face_side != "center")
7. Composite overlay if active and p > 0
8. Write to FFmpeg pipe

Curves are precomputed before the frame loop:
- `build_bounce_curves()` → per-frame p (intensity 0-1) and zoom arrays
- `build_effect_curves()` → per-frame blur/whip strength arrays
- Hold periods between in/out pairs are filled after all events are processed
