# Infographics

The infographics pipeline generates custom React/Remotion components from transcript analysis. An LLM writes TSX code that is validated, retried on failure, and falls back to existing templates as a safety net.

## Pipeline Overview

```
A0: Cleanup generated/
 │
 ▼
A1: Plan Infographics (LLM)
 │  "What data moments deserve custom visualizations?"
 │
 ├──► For each spec:
 │
 │    A2: Generate TSX Code (LLM)
 │     │  "Write a React component for this visualization"
 │     │
 │     ▼
 │    A3: Validate
 │     │  1. Write to generated/{id}.tsx
 │     │  2. npx tsc --noEmit (type-check)
 │     │  3. Remotion test render (frame 30 of 90)
 │     │
 │     ├── Valid? ──► Collect component
 │     │
 │     └── Invalid? ──► Retry (up to 3 attempts)
 │                       │  Pass errors + previous code back to LLM
 │                       │
 │                       └── All retries exhausted?
 │                            └── Fall back to template
 │
 ▼
A4: Build Generated Registry
    Write _registry.ts with all successful components
```

**Key files:**
- Workflow: `infographic_workflow.py`
- Activities: `activities/infographic.py`
- Schemas: `schemas/infographic.py`
- Prompts: `prompts/plan_infographics.md`, `prompts/generate_infographic_code.md`, `prompts/infographic_api_reference.md`
- Output: `remotion/src/components/generated/`

## A1: Plan Infographics

**Activity:** `vfx_plan_infographics`
**Prompt:** `prompts/plan_infographics.md`
**Model:** `VFX_INFOGRAPHIC_LLM_MODEL` (default: claude-opus-4-6)

The LLM analyzes the transcript for data-rich moments and decides *what* to visualize — not *how* to code it.

**Constraints:**
- Max 4 infographics per video
- 3–8 seconds visibility each
- 2+ seconds apart (no temporal overlap)
- Only concrete data (numbers, stats, comparisons) — not vague opinions
- Must fit in face-tracking safe regions

### Infographic Types

```python
class InfographicType(str, Enum):
    PIE_CHART = "pie_chart"
    BAR_CHART = "bar_chart"
    LINE_CHART = "line_chart"
    FLOWCHART = "flowchart"
    TIMELINE = "timeline"
    COMPARISON = "comparison"
    PROCESS = "process"
    STAT_DASHBOARD = "stat_dashboard"
    CUSTOM = "custom"
```

### Data Formats

Each type expects specific JSON in the `data` field:

| Type | Data Shape |
|------|-----------|
| Charts (pie, bar, line) | `{"items": [{"label": "...", "value": N}], "unit": "%"}` |
| Flowchart | `{"nodes": [{"id": "1", "text": "..."}], "edges": [{"from": "1", "to": "2"}]}` |
| Timeline | `{"events": [{"label": "...", "year": "2020"}]}` |
| Comparison | `{"left": {"title": "A", "items": [...]}, "right": {"title": "B", "items": [...]}}` |
| Process | `{"steps": [{"number": 1, "title": "...", "detail": "..."}]}` |
| Stat Dashboard | `{"stats": [{"label": "...", "value": N, "suffix": "%"}]}` |

### Output Schema

```python
class InfographicSpec(BaseModel):
    id: str
    type: InfographicType
    title: str                              # 2-5 words
    description: str
    data: dict
    start_time: float
    end_time: float
    bounds: dict                            # {x, y, w, h} normalized 0-1
    anchor: str = "static"                  # or face-relative
```

## A2: Generate TSX Code

**Activity:** `vfx_generate_infographic_code`
**Prompt:** `prompts/generate_infographic_code.md` + `prompts/infographic_api_reference.md`
**Model:** `VFX_INFOGRAPHIC_LLM_MODEL`

The system prompt is assembled from:
1. Base prompt with hard constraints
2. API reference (allowed imports, hooks, utilities)
3. Real examples (`DataAnimation.tsx`, `AnimatedTitle.tsx`)

### Hard Constraints

1. Single named export: `export const ComponentName: React.FC<...>`
2. Must accept `position: NormalizedRect` and `anchor?: AnchorMode` props
3. Must call `useFaceAwareLayout(position, anchor)` and use returned values
4. Must fade to 0 opacity in last 0.5 seconds
5. Must call `useStyle()` and use its palette/fonts
6. Inline styles only (no CSS files, no styled-components)
7. SVG for data visualization, `<div>` for text layout
8. No external imports beyond the API reference
9. No `fetch()`, no `async` — all data via props
10. TypeScript strict — no `any`, proper types

### Allowed Imports

From `remotion`: `useCurrentFrame`, `useVideoConfig`, `interpolate`, `spring`, `Easing`

From `../../lib/spatial`: `useFaceAwareLayout`

From `../../lib/styles`: `useStyle`

From `../../lib/easing`: `SPRING_GENTLE`, `SPRING_BOUNCY`, `SPRING_SNAPPY`, `SPRING_SMOOTH`

From `../../lib/infographic-utils`: `polarToCartesian`, `describeArc`, `generateTicks`, `linearScale`, `colorWithOpacity`, `lerpColor`

From `../../types`: `NormalizedRect`, `AnchorMode`

### Retry Behavior

On retry, the user message includes:
- `## RETRY (attempt N)`
- Previous error messages
- Previous code for the LLM to fix

Component IDs are derived from the spec ID with a workflow prefix (e.g., `cca9c8_focus_stats`).

## A3: Validate

**Activity:** `vfx_validate_infographic`

Two-stage validation:

### 1. TypeScript Type-Check

```bash
npx tsc --noEmit --pretty false
```

- 60-second timeout
- Filters errors to component-specific messages
- Returns first ~10 errors on failure

### 2. Test Render

If type-check passes:
- Creates a temporary `_registry.ts` with just this component
- Renders frame 30 of a 90-frame composition via Remotion
- Detects runtime errors
- Generates preview PNG

On failure: deletes the broken `.tsx` file and temporary registry.

## A4: Build Generated Registry

**Activity:** `vfx_build_generated_registry`

Writes the final `remotion/src/components/generated/_registry.ts`:

```typescript
import { Cca9c8FocusStats } from "./cca9c8_focus_stats";
import { Cca9c8AttentionSpanDecline } from "./cca9c8_attention_span_decline";

export const GeneratedRegistry: ComponentMap = {
  "cca9c8_focus_stats": Cca9c8FocusStats,
  "cca9c8_attention_span_decline": Cca9c8AttentionSpanDecline,
};
```

This registry is auto-merged into the main `ComponentRegistry` at runtime:

```typescript
// components/index.ts
try {
  const { GeneratedRegistry } = require("./generated/_registry");
  Object.assign(ComponentRegistry, GeneratedRegistry);
} catch {
  // No generated components — fine
}
```

Converts time-based specs to frame-based `ComponentSpec` dicts. Generated components get `zIndex: 10` (above template components).

## Fallback System

When all retries are exhausted for a spec, it falls back to an existing template:

```python
FALLBACK_MAP = {
    PIE_CHART:      ("data_animation", "bar"),
    BAR_CHART:      ("data_animation", "bar"),
    LINE_CHART:     ("data_animation", "bar"),
    COMPARISON:     ("listicle", "pop"),
    TIMELINE:       ("listicle", "slide"),
    PROCESS:        ("listicle", "slide"),
    FLOWCHART:      ("listicle", "slide"),
    STAT_DASHBOARD: ("data_animation", "stat-callout"),
    CUSTOM:         ("animated_title", "fade"),
}
```

Fallback components get `zIndex: 5` (below generated components).

## Merge into MG Plan

After the infographic workflow completes, its components are merged into the main MG plan:

1. Generated components (frame-domain) are added directly
2. Fallback components (template-based) are added with template registry lookup
3. The merged plan is re-validated via `vfx_validate_merged_plan` (same 8-pass validation)
4. Subtitles are injected last (`zIndex: 100`)

## Generated Component Patterns

Successful generated components typically use:

- **SVG** for charts (bars, pies, axes, grid lines)
- **Spring animations** with staggered delays per data item
- **`linearScale()`** for mapping data values to pixel dimensions
- **`colorWithOpacity()`** and **`lerpColor()`** for themed colors
- **Counting animations** for numeric values (interpolate from 0 to target)
- **Dark semi-transparent background** panels with accent borders
- **`useFaceAwareLayout()`** for responsive scaling based on face proximity

## Test CLI

```bash
# Full pipeline test
python -m video_effects.test_infographic --text "transcript here..."

# From file
python -m video_effects.test_infographic --file transcript.txt

# Skip validation for fast iteration
python -m video_effects.test_infographic --spec spec.json --skip-validate

# Override model and retries
python -m video_effects.test_infographic --text "..." --model claude-sonnet-4-6 --retries 5
```
