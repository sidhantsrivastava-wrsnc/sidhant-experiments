# Styles

The style system controls typography, color palette, animation preferences, and effect density across the entire pipeline. Seven presets are available, and the LLM creative designer can auto-detect the best fit from the transcript.

**Key files:**
- Presets: `schemas/styles.py`
- Creative designer activity: `activities/creative.py`
- Creative designer workflow: `creative_workflow.py`
- Design prompt: `prompts/design_style.md`
- Font loading: `remotion/src/lib/fonts.ts`
- Style context: `remotion/src/lib/styles.ts`

## Style Presets

### clean-minimal
Elegant, restrained. Best for educational content, interviews, thoughtful pacing.

| Property | Value |
|----------|-------|
| Font | Inter [400, 600, 700] |
| Palette | `#F5F0EB` (text), `#2D2D2D` (secondary), `#C49A6C` (accent) |
| Text shadow | `0 1px 4px rgba(0,0,0,0.3)` |
| Animations | fade, slide-in |
| Density | Sparse: 3–4 overlays / 60s |
| Color grading | warm @ 0.3 |
| Avoids | whip, speed_ramp |

### bold-energy
High-impact. Best for hype videos, sports, high-energy speakers.

| Property | Value |
|----------|-------|
| Font | Bebas Neue [400] |
| Palette | `#FFFFFF` (text), `#000000` (secondary), `#39FF14` (neon green) |
| Text shadow | `0 4px 12px rgba(0,0,0,0.8)` |
| Animations | bounce, pop |
| Density | High: 6–10 overlays / 60s |
| Color grading | dramatic @ 0.6 |
| Prefers | whip, speed_ramp |

### tech-sleek
Modern tech aesthetic. Best for tech reviews, tutorials, product demos.

| Property | Value |
|----------|-------|
| Font | DM Sans [400, 500, 700] |
| Palette | `#FFFFFF` (text), `#1A1A1A` (secondary), `#FF3333` (red accent) |
| Text shadow | `0 2px 6px rgba(0,0,0,0.5)` |
| Animations | slide-in |
| Density | Moderate: 4–6 overlays / 60s |
| Color grading | cool @ 0.3 |
| Prefers | whip, vignette |

### casual-vlog
Friendly and warm. Best for vlogs, casual conversations, behind-the-scenes.

| Property | Value |
|----------|-------|
| Font | Oswald [400, 500, 700] |
| Palette | `#FFFFFF` (text), `#1C1C1C` (secondary), `#FFB347` (orange) |
| Text shadow | `0 2px 8px rgba(0,0,0,0.6)` |
| Animations | typewriter, slide-in |
| Density | Sparse: 3–5 overlays / 60s |
| Color grading | warm @ 0.5 |
| Prefers | vignette |

### podcast-pro
Clean interview style. Best for podcast clips, interviews, panel discussions.

| Property | Value |
|----------|-------|
| Font | Source Sans 3 [400, 600, 700] |
| Palette | `#FFFFFF` (text), `#222222` (secondary), `#3B82F6` (blue) |
| Text shadow | `0 1px 4px rgba(0,0,0,0.4)` |
| Animations | fade, slide-in |
| Density | Very sparse: 2–3 overlays / 60s |
| Color grading | none (0.0) |
| Template preference | lower_third |
| Avoids | whip, speed_ramp |

### tiktok-native
Short-form social. Best for TikTok/Reels/Shorts, fast-paced content.

| Property | Value |
|----------|-------|
| Font | Poppins [400, 600, 700, 800] |
| Palette | `#FFFFFF` (text), `#000000` (secondary), `#FF2D55` (hot pink) |
| Text shadow | `0 3px 10px rgba(0,0,0,0.7)` |
| Animations | pop, bounce |
| Density | High: 6–10 overlays / 60s |
| Color grading | dramatic @ 0.4 |
| Prefers | whip, speed_ramp |

### default
Balanced fallback when no style is specified and auto-detection isn't used.

| Property | Value |
|----------|-------|
| Font | sans-serif |
| Palette | `#FFFFFF`, `#000000`, `#FFD700` (gold) |
| Density | Moderate: 3–6 overlays / 60s |

## Data Structures

### StyleConfig

Serialized to Remotion as part of `CompositionPlan`:

```python
class StyleConfig(BaseModel):
    font_family: str
    font_import: str                    # @remotion/google-fonts identifier
    font_weights_to_load: list[str]     # e.g., ["400", "700"]
    palette: list[str]                  # [text, secondary, accent]
    text_shadow: str
    font_weights: FontWeights

class FontWeights(BaseModel):
    heading: str        # e.g., "700"
    body: str           # e.g., "400"
    emphasis: str       # e.g., "800"
    marker: str         # e.g., "700"
```

### StylePreset

Full definition including LLM guidance:

```python
class StylePreset(BaseModel):
    name: str
    display_name: str
    description: str
    config: StyleConfig
    preferred_animations: list[str]
    avoided_animations: list[str]
    density_range: tuple[int, int]      # (min, max) per 60s
    density_label: str
    template_preferences: list[str]
    color_grading_preset: str           # "warm", "cool", "dramatic", ""
    color_grading_intensity: float      # 0.0–1.0
    preferred_effects: list[str]
    avoided_effects: list[str]
```

## Palette Conventions

The 3-color palette follows a consistent semantic pattern:

| Index | Role | Usage |
|-------|------|-------|
| 0 | Text | Primary text, headings |
| 1 | Secondary | Backgrounds, subtle elements, secondary text |
| 2 | Accent | Highlights, active states, accent bars, emphasis |

Components access palette via `useStyle().palette[index]`.

## Creative Designer Workflow

**Workflow:** `CreativeDesignerWorkflow` (child workflow)
**Activity:** `vfx_design_style`
**Prompt:** `prompts/design_style.md`

When the user doesn't specify `--style`, the LLM analyzes the transcript and video metadata to auto-select a preset.

**Analysis factors:**
- Speaker energy (calm / moderate / high)
- Content type (educational / entertainment / interview / tutorial / vlog / podcast / social)
- Tone (professional / casual / energetic / thoughtful)
- Video duration context

**Output:** `StyleDesignResponse`
```python
class StyleDesignResponse(BaseModel):
    preset: str            # Chosen preset name
    adjustments: dict      # Optional tweaks (palette, density_range, font_weights)
    reasoning: str
```

**Adjustment merging:** Starts with base preset config, applies LLM adjustments as a flat merge. Palette adjustments can be keyed as `{"text": "#...", "secondary": "#...", "accent": "#..."}`.

If `--style` is passed on the CLI, the creative designer is skipped and the preset is used directly.

## Font Loading

**File:** `remotion/src/lib/fonts.ts`

`loadStyleFont(fontImport, weights)` uses `@remotion/google-fonts` to load fonts at render time.

**Available fonts:**

| Font Import | CSS Family | Style |
|-------------|-----------|-------|
| `Inter` | Inter | clean-minimal |
| `BebasNeue` | Bebas Neue | bold-energy |
| `DMSans` | DM Sans | tech-sleek |
| `Oswald` | Oswald | casual-vlog |
| `SourceSans3` | Source Sans 3 | podcast-pro |
| `Poppins` | Poppins | tiktok-native |

## Style Application Points

Style influences the pipeline at multiple stages:

1. **Effect parsing (G3):** Style config + preset name passed to LLM for density and effect type guidance
2. **Color grading injection:** After timeline approval, a `color_change` effect is injected from `color_grading_preset` + `color_grading_intensity` (if > 0)
3. **MG planning (G8b):** Style guide section dynamically injected into prompt (preferred/avoided animations, density targets, template preferences, color palette)
4. **Remotion rendering (G8e):** `StyleConfig` passed as `CompositionPlan` prop, available to all components via `useStyle()`
5. **Infographic code generation:** Style config provided to LLM for palette/font consistency
