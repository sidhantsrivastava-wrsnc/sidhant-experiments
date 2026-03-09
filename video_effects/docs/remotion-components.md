# Remotion Components

The Remotion layer renders transparent animated overlays using React components. Five built-in components cover common overlay patterns; LLM-generated components extend the system for custom infographics.

**Remotion version:** 4.0.242 | **React:** 18.3.1

## DynamicComposition Engine

**File:** `remotion/src/DynamicComposition.tsx`

The composition engine receives a `CompositionPlan` and renders all components as `<Sequence>` elements:

```typescript
interface CompositionPlan {
  components: ComponentSpec[];
  colorPalette: string[];
  includeBaseVideo: boolean;
  baseVideoPath?: string;
  faceDataPath?: string;       // face_tracking_zoom.json
  zoomStatePath?: string;      // zoom_state.json
  styleConfig?: StyleConfig;
  durationInFrames?: number;
  fps?: number;
  width?: number;
  height?: number;
}
```

**Rendering flow:**
1. Load face tracking and zoom state JSON asynchronously (with `delayRender`)
2. Wrap content in `StyleProvider` → `FaceDataProvider` → `ZoomDataProvider`
3. Sort components by `zIndex`
4. Look up each `comp.template` in `ComponentRegistry`
5. Render: `<Sequence from={startFrame} durationInFrames={dur}><Component {...props} position={bounds} anchor={anchor} /></Sequence>`
6. Transparent background (for ProRes 4444 alpha)

### ComponentSpec Interface

```typescript
interface ComponentSpec {
  template: string;
  startFrame: number;
  durationInFrames: number;
  props: Record<string, unknown>;
  bounds: NormalizedRect;        // {x, y, w, h} normalized 0-1
  zIndex: number;
  anchor?: AnchorMode;
}

type AnchorMode = "static" | "face-right" | "face-left"
               | "face-below" | "face-above" | "face-beside";
```

## ComponentRegistry

**File:** `remotion/src/components/index.ts`

```typescript
export const ComponentRegistry: ComponentMap = {
  animated_title: AnimatedTitle,
  lower_third: LowerThird,
  listicle: Listicle,
  data_animation: DataAnimation,
  subtitles: Subtitles,
};

// Auto-merge generated infographic components
try {
  const { GeneratedRegistry } = require("./generated/_registry");
  Object.assign(ComponentRegistry, GeneratedRegistry);
} catch { /* no generated components */ }
```

---

## Built-in Components

### AnimatedTitle

**File:** `remotion/src/components/AnimatedTitle.tsx`

```typescript
interface AnimatedTitleProps {
  text: string;
  style: "fade" | "slide-in" | "typewriter" | "bounce";
  position: NormalizedRect;
  anchor?: AnchorMode;
  fontSize?: number;       // default 64
  color?: string;          // default "#FFFFFF"
  fontWeight?: string;
}
```

**Animation by style:**

| Style | Entrance | Spring | Details |
|-------|----------|--------|---------|
| `fade` | Opacity 0→1 over 0.5s | — | Linear interpolation |
| `slide-in` | translateX -100→0 | `SPRING_GENTLE` | Slides from left |
| `typewriter` | Char-by-char reveal | — | Blinking cursor at `frame % (fps/2) < fps/4` |
| `bounce` | Scale 0.3→1 | `SPRING_BOUNCY` | Scale transform |

Exit: All styles fade opacity to 0 over the last 0.5 seconds.

### LowerThird

**File:** `remotion/src/components/LowerThird.tsx`

```typescript
interface LowerThirdProps {
  name: string;
  title?: string;
  accentColor?: string;    // default "#FFD700"
  style?: "slide" | "fade"; // default "slide"
  position: NormalizedRect;
  anchor?: AnchorMode;
  fontSize?: number;       // default 36
  color?: string;          // default "#FFFFFF"
}
```

**Slide style animation (staggered):**
1. Accent bar grows width (0→4px) via `SPRING_GENTLE`
2. Name appears at 30–60% of bar progress with translateX
3. Title appears at 60–90% of bar progress

**Fade style:** Full bar + text fade in together via `SPRING_SMOOTH`.

Title renders at 65% of name font size with reduced opacity.

### Listicle

**File:** `remotion/src/components/Listicle.tsx`

```typescript
interface ListicleProps {
  items: string[];
  style?: "pop" | "slide";          // default "pop"
  listStyle?: "numbered" | "bullet" | "none"; // default "numbered"
  position: NormalizedRect;
  anchor?: AnchorMode;
  staggerDelay?: number;            // default 10 frames
  fontSize?: number;                // default 32
  color?: string;                   // default "#FFFFFF"
  accentColor?: string;             // default "#FFD700"
}
```

**Animation:**
- Items appear one by one with `staggerDelay` between each
- `pop`: Scale 0→1 per item using `SPRING_BOUNCY`
- `slide`: TranslateX -40→0 per item using `SPRING_GENTLE`

Markers: `"1."`, `"2."`, etc. for numbered; `"•"` for bullet; none.

### DataAnimation

**File:** `remotion/src/components/DataAnimation.tsx`

```typescript
interface DataAnimationProps {
  style: "counter" | "stat-callout" | "bar";
  value: number;
  label: string;
  position: NormalizedRect;
  anchor?: AnchorMode;
  startValue?: number;      // default 0
  suffix?: string;          // default ""
  prefix?: string;          // default ""
  delta?: number;           // for stat-callout
  items?: { label: string; value: number }[];  // for bar
  fontSize?: number;        // default 48
  color?: string;           // default "#FFFFFF"
  accentColor?: string;     // default "#FFD700"
}
```

**Styles:**

| Style | Behavior |
|-------|----------|
| `counter` | Animated number count-up via `SPRING_SNAPPY`. Formats as int or decimal. Label at 40% font size. |
| `stat-callout` | Same counter + delta indicator: `↑ 25%` (green #4ADE80) or `↓ 15%` (red #F87171) |
| `bar` | Horizontal bar chart (1–4 items). 8-frame stagger per bar. Animated width proportional to value. `SPRING_GENTLE`. |

### Subtitles

**File:** `remotion/src/components/Subtitles.tsx`

```typescript
interface SubtitlesProps {
  words: SubtitleWord[];
  position: NormalizedRect;
  anchor?: AnchorMode;
  fontSize?: number;              // default 48
  color?: string;                 // default "#FFFFFF"
  highlightColor?: string;        // defaults to palette[2]
  backgroundColor?: string;       // default "rgba(0,0,0,0.6)"
}

interface SubtitleWord {
  text: string;
  startFrame: number;
  endFrame: number;
}
```

**Features:**
- Dynamic word wrapping with max 2 lines per page
- Page-based rendering — only active page visible
- 0.15s fade transitions between pages
- Word highlighting: active word gets highlight color + 1.08× scale + emphasis weight
- Past words: normal color, 1.0× scale, body weight
- Future words: faded color (lower opacity)

---

## Hooks API

### `useFaceAwareLayout(staticBounds, anchor?)`

**File:** `remotion/src/lib/spatial.ts`

Returns `{ left, top, scale, maxWidth, maxHeight }` in pixels.

See [Face Tracking → Face-Aware Layout Hooks](face-tracking.md#face-aware-layout-hooks) for full details.

### `useStyle()`

**File:** `remotion/src/lib/styles.ts`

Returns `StyleConfig`:
```typescript
{
  font_family: string;
  font_import: string;
  font_weights_to_load: string[];
  palette: string[];        // [text, secondary, accent]
  text_shadow: string;
  font_weights: {
    heading: string;        // e.g., "700"
    body: string;           // e.g., "400"
    emphasis: string;       // e.g., "800"
    marker: string;
  };
}
```

### `useFaceFrame()`

**File:** `remotion/src/lib/context.ts`

Returns `FaceFrame | null` for the current frame:
```typescript
{ cx: number, cy: number, fw: number, fh: number }  // normalized 0-1
```

### `useZoomFrame()`

**File:** `remotion/src/lib/zoom-context.ts`

Returns `ZoomFrame` for the current frame:
```typescript
{ zoom: number, tx: number, ty: number }  // defaults to {1, 0.5, 0.5}
```

## Animation Patterns

### Spring Configs

**File:** `remotion/src/lib/easing.ts`

| Config | Damping | Mass | Stiffness | Character |
|--------|---------|------|-----------|-----------|
| `SPRING_GENTLE` | 15 | 0.8 | 80 | Smooth, elegant |
| `SPRING_BOUNCY` | 10 | 0.6 | 120 | Playful, overshoot |
| `SPRING_SNAPPY` | 20 | 0.5 | 200 | Fast, crisp |
| `SPRING_SMOOTH` | 200 | 1 | 100 | Very smooth, slow |

### Common Patterns

**Exit opacity** (used by every component):
```typescript
const fadeOutStart = durationInFrames - Math.round(fps * 0.5);
const exitOpacity = interpolate(frame, [fadeOutStart, durationInFrames], [1, 0], {
  extrapolateLeft: "clamp", extrapolateRight: "clamp",
});
```

**Responsive scaling** (multiply sizes by `scale` from `useFaceAwareLayout`):
```typescript
const { left, top, scale } = useFaceAwareLayout(position, anchor);
const scaledFontSize = fontSize * scale;
```

**Staggered animation**:
```typescript
items.map((item, i) => {
  const delay = i * staggerDelay;
  const progress = spring({ frame: Math.max(0, frame - delay), fps, config: SPRING_BOUNCY });
  // ...
});
```

## Font Loading

**File:** `remotion/src/lib/fonts.ts`

`loadStyleFont(fontImport, weights)` loads Google Fonts at render time via `@remotion/google-fonts`.

Available fonts: Inter, BebasNeue, DMSans, Oswald, SourceSans3, Poppins.

## Infographic Utilities

**File:** `remotion/src/lib/infographic-utils.ts`

| Function | Purpose |
|----------|---------|
| `polarToCartesian(cx, cy, r, angleDeg)` | Polar → cartesian (for pie/gauge) |
| `describeArc(cx, cy, r, start, end)` | SVG arc path string |
| `generateTicks(min, max, count)` | Evenly-spaced axis tick values |
| `linearScale(value, dMin, dMax, rMin, rMax)` | Map value between ranges |
| `colorWithOpacity(hex, opacity)` | Hex → `rgba(r,g,b,a)` |
| `lerpColor(hex1, hex2, t)` | Linear color interpolation |

## Type Definitions

**File:** `remotion/src/types.ts`

All shared interfaces: `CompositionPlan`, `ComponentSpec`, `NormalizedRect`, `AnchorMode`, `FaceFrame`, `ZoomFrame`, `StyleConfig`, `FontWeights`, and per-component props interfaces.
