# Motion Graphics

The Remotion motion graphics system adds animated overlays (titles, lower thirds, lists, data visualizations, subtitles) to the processed video. It's driven by LLM planning with face-aware spatial validation.

## End-to-End Flow

```
G8a: Build Spatial Context
  │  (face windows, safe regions, zoom state)
  │
  ├──► G8b: LLM Plan Motion Graphics
  │      │  (plan_motion_graphics_base.md prompt)
  │      │
  │      ▼
  │    8-pass _validate_plan()
  │      │
  │      ▼
  │    HITL Approval (up to 5 rounds)
  │      │
  │      ├──► Merge infographic components
  │      │    Re-validate merged plan
  │      │    Inject subtitles (zIndex=100)
  │      │
  │      ▼
  │    G8e: Render Overlay
  │      │  (Remotion → ProRes 4444, transparent)
  │      │
  │      ▼
  │    G9: FFmpeg Composite
  │      │  (premultiply + overlay onto base)
  │      │
  │      ▼
  │    Final .mp4
```

**Key files:**
- Activities: `activities/remotion.py`
- Helpers: `helpers/remotion.py`
- Schemas: `schemas/mg_templates.py`, `schemas/motion_graphics.py`
- Prompts: `prompts/plan_motion_graphics_base.md`, `prompts/motion_graphics_schema.py`

## LLM Planning

### System Prompt Assembly

`build_mg_system_prompt(style_config, style_preset_name)` dynamically assembles the prompt from:

1. **Base rules** (`prompts/plan_motion_graphics_base.md`): Spatial constraints (never occlude face, 10% edge margin, max 2 concurrent overlays), temporal rules (align to transcript timestamps, 0.1–0.3s delay after spoken cue), and density targets.
2. **Style guide** (from the active [style preset](styles.md)): Preferred/avoided animations, density range, color palette, template preferences.
3. **Template catalog** (from `schemas/mg_templates.py`): Available templates with props, duration ranges, placement constraints, and per-template [creative guidance](llm-prompts.md).

### User Message

The user message includes:
- Spatial context: face windows with safe regions per 3-second window
- Transcript with word-level timestamps
- Active OpenCV effects (so overlays avoid zoom transitions)
- Optional feedback from previous rejection

### Response Model

```python
class MotionGraphicsPlanResponse(BaseModel):
    components: list[MGComponentSpec]
    color_palette: list[str]    # 2-3 CSS hex colors
    reasoning: str

class MGComponentSpec(BaseModel):
    template: str               # e.g., "animated_title"
    start_time: float
    end_time: float
    props: dict                 # Template-specific
    bounds: MGComponentBounds   # Normalized 0-1 rect
    z_index: int
    anchor: Literal["static", "face-right", "face-left",
                    "face-below", "face-above", "face-beside"]
    reasoning: str
```

## Template Registry

Four implemented templates, defined in `schemas/mg_templates.py`:

| Template | Props | Duration | Spatial | Edge-Aligned |
|----------|-------|----------|---------|--------------|
| `animated_title` | text, style (fade/slide-in/typewriter/bounce), fontSize, color, fontWeight | 2–5s | y: 0.05–0.25 | No |
| `lower_third` | name, title, accentColor, style (slide/fade), fontSize, color | 3–6s | y: 0.75–0.88, x: 0.03–0.45 | Yes |
| `listicle` | items (max 5), style (pop/slide), listStyle, staggerDelay, fontSize, color, accentColor | 3–8s | y: 0.15–0.75 | No |
| `data_animation` | style (counter/stat-callout/bar), value, label, startValue, suffix, prefix, delta, items, fontSize, color, accentColor | 2–6s | y: 0.15–0.75 | No |

Each template has a guidance markdown file in `prompts/mg_guidance/` loaded dynamically into the LLM prompt.

`IMPLEMENTED_TEMPLATES` is the source of truth — only templates in this set are available.

## Spatial Validation (`_validate_plan()`)

An 8-pass validation pipeline in `activities/remotion.py` (lines 599–815) ensures overlays don't occlude the speaker or conflict with each other.

### Pass Summary

| # | Check | Action |
|---|-------|--------|
| 1 | **Hard bounds clamping** | Keep within [0.02, 0.98], enforce min size 0.05 × 0.03 |
| 2 | **Time clamping** | Clamp to video duration, enforce min 0.5s |
| 3 | **Template duration limits** | Enforce max duration per template spec (e.g., lower_third 6s max) |
| 4 | **Face overlap relocation** | If >5% overlap with face, relocate to best safe region via `_find_best_safe_region()` |
| 5 | **Concurrent count enforcement** | Drop lowest z_index if ≥2 non-edge-aligned components overlap in time |
| 6 | **Zoom viewport clamping** | During active zooms, clamp bounds to the visible inner area (e.g., 67% for 1.5× zoom) |
| 7 | **Zoom transition buffer** | Shift overlay timing away from zoom ease-in/ease-out windows |
| 8 | **Multi-pass spatial conflict resolution** | Up to 3 iterations of `_resolve_spatial_conflicts()` to converge |

### Key Helpers

- `_rect_overlap_fraction(a, b)` — Overlap as fraction of rect a's area
- `_find_best_safe_region(comp_w, comp_h, safe_regions)` — Picks largest fitting safe region, shrinks if needed
- `_resolve_spatial_conflicts(components, edge_aligned_templates, issues)` — Shifts lower z_index component away (vertically then horizontally)
- `_compute_zoom_stable_window(zoom_cue)` — Returns time range where zoom is stable:
  - Bounce: 25%–75% of duration
  - In: 60%–end
  - Out: start–40%

## ProRes 4444 Rendering

**File:** `helpers/remotion.py` → `render_media()`

```bash
npx remotion render MotionOverlay output.mov \
  --codec prores --prores-profile 4444 \
  --pixel-format yuva444p10le \
  --image-format png \
  --props '...'
```

| Setting | Value | Reason |
|---------|-------|--------|
| Codec | ProRes 4444 | Preserves alpha channel |
| Pixel format | yuva444p10le | 10-bit YUVA 4:4:4 with alpha |
| Image format | PNG | Lossless intermediate frames |
| Timeout | 600s | Long renders for complex overlays |

The `CompositionPlan` is serialized as JSON props, containing all components, color palette, face data path, zoom state path, and style config.

## FFmpeg Compositing

**File:** `helpers/remotion.py` → `composite_overlay()`

```bash
ffmpeg -y -i base.mp4 -i overlay.mov \
  -filter_complex "[1:v]premultiply=inplace=1[ovr];[0:v][ovr]overlay=0:0:shortest=1" \
  -c:v libx264 -crf 16 -c:a copy \
  output.mp4
```

1. **Premultiply**: Convert ProRes 4444 straight alpha to premultiplied
2. **Overlay**: Full-frame placement (0,0), trimmed to shorter stream
3. **Encode**: H.264 CRF 16, copy audio stream

## Subtitle Injection

After the MG plan is finalized, word-level transcript segments are injected as a `subtitles` component with `zIndex=100` (renders above all other overlays). The Remotion [Subtitles component](remotion-components.md#subtitles) handles word-by-word highlighting.

## Parallelization

MG planning (G8b), infographic generation (child workflow), and video rendering (G6c) all run concurrently after G8a completes. The merge happens after all three finish, followed by the final Remotion render and FFmpeg composite.
