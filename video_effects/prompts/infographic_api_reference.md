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
import { SPRING_GENTLE, SPRING_BOUNCY, SPRING_SNAPPY, SPRING_SMOOTH } from "../../lib/easing";
```

- `SPRING_GENTLE` — `{ damping: 15, mass: 0.8, stiffness: 80 }` — smooth entrance
- `SPRING_BOUNCY` — `{ damping: 10, mass: 0.6, stiffness: 120 }` — playful bounce
- `SPRING_SNAPPY` — `{ damping: 20, mass: 0.5, stiffness: 200 }` — quick snap
- `SPRING_SMOOTH` — `{ damping: 200, mass: 1, stiffness: 100 }` — no overshoot

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

## Types

```typescript
import type { NormalizedRect, AnchorMode } from "../../types";
```

- `NormalizedRect` — `{ x: number, y: number, w: number, h: number, label?: string }` (0-1 normalized)
- `AnchorMode` — `"static" | "face-right" | "face-left" | "face-below" | "face-above" | "face-beside"`
