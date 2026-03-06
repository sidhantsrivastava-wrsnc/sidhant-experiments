# Motion Graphics Overlay Integration

## Current Status
**Phase**: 1 (MVP) — one template, manual plan, no LLM
**Branch**: `temporal-fafo`
**Last updated**: 2026-03-06

---

## What's Done

### Task 1: Pydantic schemas (DONE)
- Created `schemas/motion_graphics.py`
- Models: `MGTemplate`, `NormalizedRect`, `FaceWindow`, `SpatialContext`, `MotionGraphicsComponent`, `MotionGraphicsPlan`, `MotionGraphicsPreview`
- Follows existing patterns from `schemas/effects.py`

### Task 2: Workflow schema updates (DONE)
- `schemas/workflow.py` — added `enable_motion_graphics: bool = False`, `motion_graphics_style: str = ""` to `VideoEffectsInput`
- Added `motion_graphics_applied: int = 0` to `VideoEffectsOutput`
- Fully backward compatible (all new fields have defaults)

### Task 3: Remotion project scaffold (DONE)
- Created `video_effects/remotion/` with full project structure
- `package.json` with remotion 4.0.242, react 18, typescript 5.7
- `DynamicComposition.tsx` — reads plan JSON, maps components into `<Sequence>` elements
- `AnimatedTitle.tsx` — 4 styles: fade, slide-in, typewriter, bounce
- `lib/context.ts` — `FaceDataProvider` + `useFaceFrame` hook
- `lib/spatial.ts` — `useFaceAvoidance` hook + `computeOverlap`
- `lib/easing.ts` — spring configs (gentle, bouncy, snappy)
- TypeScript compiles clean, `npx remotion compositions` lists `MotionOverlay`

### Batch 1 Testing (DONE)
- **Still renders**: Verified at frames 0, 15, 45, 80 — AnimatedTitle renders correctly
  - Fade style: opacity ramps from 0 to 1 over 0.5s
  - Slide-in style: springs in from left with gold color
  - Both titles visible simultaneously at frame 45
- **Transparent ProRes**: `yuva444p12le` pixel format confirmed (alpha channel present)
  - **Key flags**: `--image-format png --pixel-format yuva444p10le` required for transparency
  - Without `--image-format png`, Remotion defaults to jpeg → no alpha
- **FFmpeg composite**: Successfully overlaid transparent ProRes onto solid dark blue base
  - Filter: `[1:v]premultiply=inplace=1[ovr];[0:v][ovr]overlay=0:0`
  - Text renders crisp with proper alpha blending on dark background
- Test artifacts in `remotion/test_output/` (not committed)

---

### Task 4: Python-Remotion bridge (DONE)
- Created `helpers/remotion.py` with 3 functions:
  - `render_still()` — `npx remotion still` subprocess wrapper
  - `render_media()` — `npx remotion render` with `--image-format png --pixel-format yuva444p10le` for transparent ProRes 4444
  - `composite_overlay()` — FFmpeg alpha composite with premultiply filter
- Auto-detects remotion dir relative to package, overridable via settings

### Task 5: Temporal activities (DONE)
- Created `activities/remotion.py` with 3 activities:
  - `vfx_render_motion_overlay` — renders transparent overlay .mov from composition plan
  - `vfx_composite_motion_graphics` — FFmpeg composites overlay onto base video
  - `vfx_preview_motion_graphics` — renders preview PNG snapshots at key frames
- Registered all 3 in `activities/__init__.py`

### Task 6: Wire G8e + G9 into workflow (DONE)
- Added `_run_motion_graphics()` method to `VideoEffectsWorkflow`
- Gated by `input.enable_motion_graphics` flag after G7
- Phase 1: reads `{temp_dir}/remotion/composition_plan.json` (manually placed)
- G8e calls `vfx_render_motion_overlay`, G9 calls `vfx_composite_motion_graphics`
- Composited result replaces original output file
- `motion_graphics_applied` count propagated to `VideoEffectsOutput`

### Task 7: Config + CLI flags (DONE)
- `config.py`: Added `REMOTION_DIR` and `REMOTION_CONCURRENCY` settings (env: `VFX_REMOTION_DIR`, `VFX_REMOTION_CONCURRENCY`)
- `cli.py`: Added `--motion-graphics` / `--mg` flag, result display shows component count

### Batch 2 Testing (DONE)
- Python imports: all schemas, helpers, activities import cleanly
- Python -> Remotion bridge: `render_still()` successfully calls npx subprocess, renders 70KB PNG
- Backward compatibility: `VideoEffectsInput()` without new fields works, defaults to MG off

### Bugfix: Standalone face tracking (DONE)
- **Problem**: `helpers/face_tracking.py` imported from `cv_experiments/zoom_bounce.py` which used
  `ProcessPoolExecutor` with `fork` — MediaPipe segfaults in forked processes on macOS (Apple Silicon)
  → `BrokenProcessPool` error during G6b
- **Fix**: Rewrote `face_tracking.py` as a fully standalone module:
  - Single-process ffmpeg-pipe decoding (no ProcessPoolExecutor, no fork)
  - Direct MediaPipe FaceLandmarker usage
  - Same stride-based detection + np.interp interpolation logic
  - Resolves `FACE_LANDMARKER_PATH` relative to project root
  - No dependency on `cv_experiments/` — zero imports from it
  - `smooth_data()` uses `settings.SMOOTHING_ALPHA` as default
- **Verified**: imports clean, `smooth_data()` produces correct output

---

### Phase 2: LLM Planner + Context (DONE)

#### Task 8: LLM planner prompt (DONE)
- Created `prompts/plan_motion_graphics.md` (5.1KB)
- Describes all 6 templates with props, constraints, typical usage
- Spatial rules: face avoidance, safe zones, edge margins, zoom viewport shrink
- Temporal rules: max 2 concurrent, respect OpenCV effects, natural pacing
- Style rules: minimal by default, consistent palette, match speaker energy

#### Task 9: Response schema (DONE)
- Created `prompts/motion_graphics_schema.py`
- Models: `MGComponentBounds`, `MGComponentSpec`, `MotionGraphicsPlanResponse`
- Generates valid JSON schema for Claude tool use

#### Task 10: G8a + G8b activities + validation + workflow wiring (DONE)
- **G8a `vfx_build_remotion_context`**: Loads face tracking cache, computes 3s time-windowed
  face averages with safe regions (left/right/top/bottom of face), packages transcript + effects
- **G8b `vfx_plan_motion_graphics`**: Calls Claude via `call_structured()` with spatial context,
  transcript, effects, and style hint. Converts LLM response (time-based) to Remotion format (frame-based).
- **Validation** (`_validate_plan`):
  - Clamps times to video duration
  - Max 2 concurrent overlays (drops lowest z_index)
  - Zoom viewport coordination (adjusts bounds during zoom periods)
- **Workflow**: Replaced hardcoded `composition_plan.json` reader with G8a → G8b → G8e → G9 pipeline
- **LLM helper fix**: `call_structured()` tool name now generic (`structured_output`) instead of
  hardcoded `parse_effects`
- Registered `build_remotion_context` and `plan_motion_graphics` in `activities/__init__.py`

---

## What's Next (Phase 3+)

### Phase 3: Preview + Approval
- Preview frame selection logic
- Second approval gate (G8d) in workflow
- CLI updates (plan display, snapshot paths, approval loop)

### Phase 4: Template Library
- LowerThird, ZoomCallout, TransitionWipe, KeywordHighlight, ProgressBar
- `useFaceAvoidance` hook shared across all components
- Update LLM prompt with all template descriptions

---

## Architecture Notes

### Data flow: Python <-> Node.js
- All communication via JSON files on disk + subprocess calls
- Python writes `composition_plan.json` + `spatial_context.json` to `{temp_dir}/remotion/`
- Python calls `npx remotion still/render` as subprocess
- Node.js reads plan, renders output to `{temp_dir}/remotion/output/`
- Python reads output PNGs/MOVs and composites with FFmpeg

### Pipeline extension
```
G1-G7 (unchanged) --> base_with_audio.mp4
G8a: Build Remotion context    (face tracking + transcript --> spatial_context)
G8b: LLM plan motion graphics  (Claude selects templates, positions, timing)
G8e: Render Remotion overlay    (renderMedia --> transparent ProRes 4444)
G9:  Final composite            (FFmpeg: base + overlay --> output.mp4)
```
Phase 3 will add G8c (preview snapshots) and G8d (approval gate).

### Key files
| File | Purpose |
|------|---------|
| `schemas/motion_graphics.py` | Pydantic models for MG plan, components, spatial context |
| `schemas/workflow.py` | Updated with `enable_motion_graphics`, `motion_graphics_applied` |
| `helpers/remotion.py` | Python subprocess wrappers for Remotion CLI + FFmpeg composite |
| `activities/remotion.py` | Temporal activities: context, plan, render, composite, preview |
| `workflow.py` | G8a-G8e + G9 stages, `_run_motion_graphics()` method |
| `prompts/plan_motion_graphics.md` | LLM system prompt for MG planning |
| `prompts/motion_graphics_schema.py` | Response schema for LLM structured output |
| `config.py` | `REMOTION_DIR`, `REMOTION_CONCURRENCY` settings |
| `cli.py` | `--motion-graphics` flag, MG result display |
| `remotion/src/DynamicComposition.tsx` | Core composition renderer |
| `remotion/src/components/AnimatedTitle.tsx` | First template (4 styles) |
| `remotion/src/lib/context.ts` | Face data React context |
| `remotion/src/lib/spatial.ts` | Face avoidance hook |
| `remotion/test_plan.json` | Test composition plan for manual testing |
