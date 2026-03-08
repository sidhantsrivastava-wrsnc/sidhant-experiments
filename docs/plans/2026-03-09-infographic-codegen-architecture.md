# Infographic Code-Generation Pipeline

## Overview

The infographic pipeline generates custom React/Remotion TSX components from transcript analysis. It runs as a Temporal child workflow (`InfographicGeneratorWorkflow`) in parallel with video rendering and motion graphics planning.

**Key idea**: Instead of using fixed templates for data visualizations, an LLM analyzes the transcript for data-rich moments, then generates bespoke TSX components that are type-checked and test-rendered before being included in the final video.

## Architecture

### Where it lives in the workflow

```
VideoEffectsWorkflow
  ├── G1-G2: Extract video + transcribe audio
  ├── CreativeDesignerWorkflow (child) → style_config
  ├── G3-G5: Parse effects + approve timeline
  ├── G6a: Prepare render plan
  │
  ├── [PARALLEL]
  │   ├── G6b-G6c: Render video frames
  │   ├── MG Planning + approval
  │   └── InfographicGeneratorWorkflow (child) ← THIS
  │
  ├── Merge infographic components into MG plan
  ├── G8e: Render Remotion overlay (includes infographics)
  └── G9: Composite overlay onto base video
```

The infographic workflow runs in parallel with video rendering. Its output (Remotion `ComponentSpec` dicts) gets merged into the motion graphics overlay plan before rendering.

### Gating

Requires `--infographics` CLI flag. This also auto-enables `--motion-graphics` since infographic components render through the MG overlay pipeline.

**Important**: The infographic pipeline only executes if video effects exist (lines 249-269 in `workflow.py` have early returns that prevent reaching the infographic start point).

## Pipeline Stages (A0-A4)

All stages are Temporal activities, callable directly as Python functions.

### A0: Cleanup (`vfx_cleanup_generated`)

Removes all previously generated `.tsx` files from `remotion/src/components/generated/`. Preserves `.gitignore`.

- **Input**: `{}` (no params)
- **Output**: `{"cleaned": int}`

### A1: Plan (`vfx_plan_infographics`)

LLM analyzes the transcript and identifies data-rich moments suitable for infographic visualization.

- **Model**: `claude-opus-4-6` (configurable via `INFOGRAPHIC_LLM_MODEL`)
- **Input**: spatial context, transcript, segments, style config, FPS
- **Output**: list of `InfographicSpec` dicts + reasoning string
- **Prompt**: `prompts/plan_infographics.md`

The planner outputs structured `InfographicSpec` objects:

```python
InfographicSpec:
    id: str              # e.g. "cloud_market_share"
    type: InfographicType  # pie_chart, bar_chart, line_chart, stat_dashboard, ...
    title: str           # Short display title
    description: str     # What to show
    data: dict           # All data needed to render (items, values, labels)
    start_time: float    # When to appear (seconds)
    end_time: float      # When to disappear
    bounds: dict         # Normalized screen position {x, y, w, h}
    anchor: str          # "static" or face-aware mode
```

Rules enforced by the prompt:
- Max 4 infographics per video
- Only concrete data (numbers, lists, comparisons)
- 3-8 seconds visible each
- No time overlaps (2s minimum gap)
- Don't overlap speaker's face

### A2: Generate (`vfx_generate_infographic_code`)

LLM generates a complete React/Remotion TSX component for one infographic spec.

- **Model**: `claude-opus-4-6`
- **Input**: spec, style config, video info, attempt number, previous errors/code
- **Output**: `{component_id, tsx_code, export_name, props}`
- **Prompt**: `prompts/generate_infographic_code.md` + `prompts/infographic_api_reference.md`

The codegen prompt includes:
- API reference for allowed imports (Remotion hooks, utility functions)
- Real component examples from `remotion/src/components/` (DataAnimation.tsx, AnimatedTitle.tsx)
- On retries: previous code + error messages for targeted fixes

### A3: Validate (`vfx_validate_infographic`)

Writes the TSX to disk and validates it in two steps:

1. **TypeScript type-check**: `npx tsc --noEmit` in the Remotion project. Filters errors to the generated file.
2. **Test render**: Renders a still frame via Remotion to catch runtime errors.

- **Input**: `{component_id, tsx_code, export_name}`
- **Output**: `{valid: bool, errors: list[str], preview_path: str}`

On failure, the generated file is cleaned up and errors are returned for the retry loop.

### Retry Loop

For each spec, the workflow runs A2+A3 in a loop up to `INFOGRAPHIC_MAX_RETRIES` (default: 3) times:

```
for attempt in 1..max_retries:
    code = A2_generate(spec, attempt, previous_errors, previous_code)
    result = A3_validate(code)
    if result.valid:
        return success
    previous_errors = result.errors
    previous_code = code
return failure → fallback
```

On exhausting retries, the spec falls back to an existing template via `FALLBACK_MAP`:

| Infographic Type | Fallback Template | Fallback Style |
|-----------------|-------------------|----------------|
| pie_chart       | data_animation    | bar            |
| bar_chart       | data_animation    | bar            |
| line_chart      | data_animation    | bar            |
| comparison      | listicle          | pop            |
| timeline        | listicle          | slide          |
| process         | listicle          | slide          |
| flowchart       | listicle          | slide          |
| stat_dashboard  | data_animation    | stat-callout   |
| custom          | animated_title    | fade           |

### A4: Build Registry (`vfx_build_generated_registry`)

Writes `_registry.ts` exporting all successful generated components, and converts time-based specs to frame-based Remotion `ComponentSpec` dicts.

- **Input**: list of generated components, FPS, style config
- **Output**: `{components: list[ComponentSpec], registry_path: str}`

## File Layout

```
video_effects/
├── activities/
│   └── infographic.py        # A0-A4 activity functions
├── schemas/
│   └── infographic.py        # InfographicSpec, InfographicPlanResponse, FALLBACK_MAP
├── infographic_workflow.py    # Temporal child workflow (orchestrates A0-A4)
├── test_infographic.py        # Standalone CLI for testing without Temporal
├── prompts/
│   ├── plan_infographics.md           # A1 system prompt
│   ├── generate_infographic_code.md   # A2 system prompt
│   └── infographic_api_reference.md   # Allowed imports for codegen
└── remotion/src/components/
    └── generated/             # Output directory for TSX files + _registry.ts
```

## Model Configuration

| Activity | Default Model | Config Key |
|----------|--------------|------------|
| A1: Plan | claude-opus-4-6 | `VFX_INFOGRAPHIC_LLM_MODEL` |
| A2: Generate | claude-opus-4-6 | `VFX_INFOGRAPHIC_LLM_MODEL` |
| Creative Designer | claude-sonnet-4-6 | `VFX_LLM_MODEL` |
| Effect Parser | claude-sonnet-4-6 | `VFX_LLM_MODEL` |

Opus is used for infographic codegen because the TSX generation requires higher reasoning quality to produce valid, well-animated components on fewer attempts.

## Test CLI

`video_effects/test_infographic.py` runs the A0-A4 pipeline without Temporal by mocking `activity.heartbeat`.

```bash
# Basic usage
python -m video_effects.test_infographic --text "transcript..." --skip-validate

# From file, filter to one type
python -m video_effects.test_infographic --file transcript.txt --type pie_chart

# Skip planning, provide specs directly
python -m video_effects.test_infographic --spec specs.json

# Override model
python -m video_effects.test_infographic --text "..." --model claude-sonnet-4-6
```

## Style Integration

The creative designer workflow picks a style preset and optionally adjusts it. The `style_config` (palette, fonts) flows into:

1. **A1 planner**: via `_build_style_guide()` — tells the LLM what colors/fonts to consider
2. **A2 codegen**: passed as context so generated components use `useStyle()` hooks
3. **A4 registry**: passed through for downstream rendering

The palette format is `list[str]` with 3 hex colors: `[text, secondary, accent]`. If the creative designer LLM returns a dict palette (e.g. `{"accent": "#FF4040"}`), the `design_style` activity maps named keys to list positions.
