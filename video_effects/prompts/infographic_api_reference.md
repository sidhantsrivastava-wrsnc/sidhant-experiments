# Infographic Component API Reference

You may ONLY use these imports. No other packages, no fetch(), no async operations.

## From `remotion`

```typescript
import { useCurrentFrame, useVideoConfig, interpolate, spring, Easing } from "remotion";
```

- `useCurrentFrame()` -> `number` — current frame index (starts at 0)
- `useVideoConfig()` -> `{ fps, durationInFrames, width, height }` — video metadata
- `interpolate(value, inputRange, outputRange, options?)` -> `number` — map value between ranges
  - options: `{ extrapolateLeft?: "clamp"|"extend", extrapolateRight?: "clamp"|"extend" }`
- `spring({ frame, fps, config?, durationInFrames? })` -> `number` (0 to ~1) — physics spring
- `Easing.bezier(x1, y1, x2, y2)` -> easing function

## From `../../lib/spatial`

```typescript
import { useFaceAwareLayout } from "../../lib/spatial";
```

- `useFaceAwareLayout(position: NormalizedRect, anchor?: AnchorMode)` -> `{ left, top, scale, maxWidth, maxHeight }`
  - REQUIRED in every component. Returns pixel values for absolute positioning.

## From `../../lib/styles`

```typescript
import { useStyle } from "../../lib/styles";
```

- `useStyle()` -> `StyleConfig`
  - `font_family: string` — CSS font family
  - `palette: string[]` — [text, secondary, accent] hex colors
  - `text_shadow: string` — CSS text-shadow
  - `font_weights: { heading, body, emphasis, marker }` — CSS font-weight strings

## From `../../lib/easing`

```typescript
import { SPRING_GENTLE, SPRING_BOUNCY, SPRING_SNAPPY, SPRING_SMOOTH, SPRING_ELASTIC, SPRING_WOBBLY } from "../../lib/easing";
```

- `SPRING_GENTLE` — `{ damping: 15, mass: 0.8, stiffness: 80 }` — smooth entrance
- `SPRING_BOUNCY` — `{ damping: 10, mass: 0.6, stiffness: 120 }` — playful bounce
- `SPRING_SNAPPY` — `{ damping: 20, mass: 0.5, stiffness: 200 }` — quick snap
- `SPRING_SMOOTH` — `{ damping: 200, mass: 1, stiffness: 100 }` — no overshoot
- `SPRING_ELASTIC` — `{ damping: 5, mass: 0.4, stiffness: 300 }` — heavy overshoot, dramatic entrance
- `SPRING_WOBBLY` — `{ damping: 8, mass: 1.0, stiffness: 150 }` — slow wobble settle, playful

## From `../../lib/infographic-utils`

```typescript
import {
  polarToCartesian,
  describeArc,
  generateTicks,
  linearScale,
  colorWithOpacity,
  lerpColor,
} from "../../lib/infographic-utils";
```

- `polarToCartesian(cx, cy, r, angleDeg)` -> `{ x, y }` — for pie slices, gauges
- `describeArc(cx, cy, r, startAngle, endAngle)` -> `string` — SVG arc `d` attribute
- `generateTicks(min, max, count)` -> `number[]` — evenly-spaced axis ticks
- `linearScale(value, domainMin, domainMax, rangeMin, rangeMax)` -> `number` — map domains
- `colorWithOpacity(hex, opacity)` -> `string` — `"rgba(r,g,b,a)"` from hex + alpha
- `lerpColor(hex1, hex2, t)` -> `string` — blend two hex colors

## From `../../lib/component-utils`

```typescript
import { drawConnector, distributeEvenly, tokenize } from "../../lib/component-utils";
```

- `drawConnector(x1, y1, x2, y2, curve?)` -> `string` — SVG path `d` string for a curved connector between two points. `curve` defaults to 0.5 (0 = straight line)
- `distributeEvenly(count, start, end)` -> `number[]` — compute evenly-spaced positions along a range
- `tokenize(code, language)` -> `{ text: string, type: "keyword" | "string" | "comment" | "number" | "plain" }[]` — simple regex tokenizer for syntax highlighting (supports `"javascript"`, `"typescript"`, `"python"`)

## Types

```typescript
import type { NormalizedRect, AnchorMode } from "../../types";
```

- `NormalizedRect` — `{ x: number, y: number, w: number, h: number, label?: string }` (0-1 normalized)
- `AnchorMode` — `"static" | "face-right" | "face-left" | "face-below" | "face-above" | "face-beside"`

## CSS 3D Transforms (Chromium-Supported)

Components render in Remotion's Chromium renderer, so all CSS 3D transforms work:

```typescript
// 3D card flip — needs two "faces" with backfaceVisibility: "hidden"
style={{ transform: `perspective(800px) rotateY(${angle}deg)`, backfaceVisibility: "hidden" }}

// Perspective tilt entrance — spring from 15° to 0°
const tilt = interpolate(spring({ frame, fps, config: SPRING_GENTLE }), [0, 1], [15, 0]);
style={{ transform: `perspective(1000px) rotateX(${tilt}deg)` }}

// Scale zoom emphasis — burst then settle
const s = interpolate(spring({ frame, fps, config: SPRING_ELASTIC }), [0, 0.5, 1], [1, 2.5, 1]);
style={{ transform: `scale(${s})` }}

// Skew entrance
const skew = interpolate(spring({ frame, fps, config: SPRING_SNAPPY }), [0, 1], [-10, 0]);
style={{ transform: `skewX(${skew}deg)` }}
```

## Animated Masking & Reveals

```typescript
// Circle reveal — radius expands from center
const r = interpolate(spring({ frame, fps, config: SPRING_SMOOTH }), [0, 1], [0, 75]);
style={{ clipPath: `circle(${r}% at 50% 50%)` }}

// Directional wipe — polygon sweeps left to right
const p = interpolate(frame, [0, 30], [0, 100], { extrapolateRight: "clamp" });
style={{ clipPath: `polygon(0 0, ${p}% 0, ${p}% 100%, 0 100%)` }}

// Gradient mask — soft edge fade into base video
style={{ maskImage: "linear-gradient(to bottom, black 70%, transparent)" }}

// Radial gradient mask — vignette-style soft focus
style={{ maskImage: "radial-gradient(ellipse at center, black 50%, transparent 100%)" }}
```

## Full-Screen Patterns

Use `AbsoluteFill` from remotion for full-screen effects:

```typescript
import { AbsoluteFill } from "remotion";

// Color flash — full-screen color that fades out over 6 frames
const opacity = interpolate(frame, [0, 6], [1, 0], { extrapolateRight: "clamp" });
<AbsoluteFill style={{ backgroundColor: accentColor, opacity }} />

// Letterbox bars
const barHeight = spring({ frame, fps, config: SPRING_GENTLE }) * 80;
<div style={{ position: "absolute", top: 0, left: 0, right: 0, height: barHeight, background: "black" }} />
<div style={{ position: "absolute", bottom: 0, left: 0, right: 0, height: barHeight, background: "black" }} />
```

## Video Cutout Patterns

Create "windows" in the overlay where the base video shows through. The overlay is alpha-composited onto the base video — opaque regions show the overlay, transparent regions show the video.

**SVG mask technique:** `fill="white"` = opaque (shows overlay), `fill="black"` = transparent (shows video).

```typescript
// Framed video cutout — rounded-rect window with decorative border
const { width, height } = useVideoConfig();
const winX = width * 0.1, winY = height * 0.08;
const winW = width * 0.8, winH = height * 0.84;
const cornerR = 24;

<svg width={width} height={height} style={{ position: "absolute", top: 0, left: 0 }}>
  <defs>
    <mask id="cutout-mask">
      <rect width={width} height={height} fill="white" />
      <rect x={winX} y={winY} width={winW} height={winH} rx={cornerR} fill="black" />
    </mask>
  </defs>
  {/* Opaque background — video is hidden here */}
  <rect width={width} height={height} fill={bgColor} mask="url(#cutout-mask)" />
  {/* Decorative border around the window */}
  <rect x={winX} y={winY} width={winW} height={winH} rx={cornerR}
        fill="none" stroke={accentColor} strokeWidth={3} />
</svg>

// Circle cutout — video visible inside a circle
<defs>
  <mask id="circle-mask">
    <rect width={width} height={height} fill="white" />
    <circle cx={width / 2} cy={height / 2} r={radius} fill="black" />
  </mask>
</defs>

// Animated cutout reveal — window expands from center
const progress = spring({ frame, fps, config: SPRING_SMOOTH });
const animR = interpolate(progress, [0, 1], [0, width * 0.4]);
<circle cx={width / 2} cy={height / 2} r={animR} fill="black" />

// Hexagon cutout
const hexPoints = Array.from({ length: 6 }, (_, i) => {
  const angle = (Math.PI / 3) * i - Math.PI / 2;
  return `${cx + r * Math.cos(angle)},${cy + r * Math.sin(angle)}`;
}).join(" ");
<polygon points={hexPoints} fill="black" />

// Multi-window cutout — two side-by-side windows
<mask id="multi-mask">
  <rect width={width} height={height} fill="white" />
  <rect x={width * 0.03} y={height * 0.1} width={width * 0.44} height={height * 0.8} rx={16} fill="black" />
  <rect x={width * 0.53} y={height * 0.1} width={width * 0.44} height={height * 0.8} rx={16} fill="black" />
</mask>

// Device mockup — phone-shaped cutout (simplified path)
const phoneW = 280, phoneH = 560;
const px = (width - phoneW) / 2, py = (height - phoneH) / 2;
const bezelR = 36, screenInset = 12;
// Outer bezel (opaque device frame)
<rect x={px} y={py} width={phoneW} height={phoneH} rx={bezelR}
      fill="#1a1a1a" stroke="#333" strokeWidth={2} />
// Inner screen (transparent window)  — in the mask:
<rect x={px + screenInset} y={py + screenInset * 2}
      width={phoneW - screenInset * 2} height={phoneH - screenInset * 3}
      rx={bezelR - screenInset} fill="black" />
```

**Important:** Cutout components must use full-screen bounds `{x:0, y:0, w:1, h:1}` and `z_index: 2` so they render below infographics and subtitles. The SVG must use pixel dimensions from `useVideoConfig()`, not percentages.

## Particle Systems

```typescript
// Generate particles with randomized physics
const particles = Array.from({ length: 40 }, (_, i) => ({
  angle: (i / 40) * Math.PI * 2 + Math.random() * 0.3,
  speed: 200 + Math.random() * 300,
  size: 4 + Math.random() * 8,
  color: palette[i % palette.length],
}));

// Per-particle position with gravity
const t = frame / fps;
const x = centerX + Math.cos(p.angle) * p.speed * t;
const y = centerY + Math.sin(p.angle) * p.speed * t + 0.5 * 800 * t * t; // gravity
const opacity = interpolate(frame, [0, 45], [1, 0], { extrapolateRight: "clamp" });
```
