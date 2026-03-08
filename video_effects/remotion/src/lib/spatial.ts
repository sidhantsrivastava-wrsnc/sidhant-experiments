import { spring, useCurrentFrame, useVideoConfig } from "remotion";
import { useFaceFrame } from "./context";
import { useZoomFrame } from "./zoom-context";
import type { AnchorMode, FaceFrame, NormalizedRect } from "../types";

interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

function faceToRect(face: FaceFrame): Rect {
  return {
    x: face.cx - face.fw / 2,
    y: face.cy - face.fh / 2,
    w: face.fw,
    h: face.fh,
  };
}

export function computeOverlap(a: Rect, b: Rect): number {
  const overlapX = Math.max(
    0,
    Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x),
  );
  const overlapY = Math.max(
    0,
    Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y),
  );
  const overlapArea = overlapX * overlapY;
  const aArea = a.w * a.h;
  return aArea > 0 ? overlapArea / aArea : 0;
}

const PADDING = 0.03;
const EDGE_MIN = 0.02;
const EDGE_MAX = 0.98;

function clampBounds(
  x: number,
  y: number,
  w: number,
  h: number,
): { x: number; y: number; w: number; h: number } {
  let cx = Math.max(EDGE_MIN, x);
  let cy = Math.max(EDGE_MIN, y);
  if (cx + w > EDGE_MAX) cx = EDGE_MAX - w;
  if (cy + h > EDGE_MAX) cy = EDGE_MAX - h;
  cx = Math.max(EDGE_MIN, cx);
  cy = Math.max(EDGE_MIN, cy);
  return { x: cx, y: cy, w, h };
}

export function useFaceAvoidance(myBounds: NormalizedRect): {
  offsetX: number;
  offsetY: number;
} {
  const face = useFaceFrame();
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  if (!face) return { offsetX: 0, offsetY: 0 };

  const faceRect = faceToRect(face);
  const myRect: Rect = {
    x: myBounds.x,
    y: myBounds.y,
    w: myBounds.w,
    h: myBounds.h,
  };

  const overlap = computeOverlap(myRect, faceRect);
  if (overlap < 0.01) return { offsetX: 0, offsetY: 0 };

  const myCenterX = myRect.x + myRect.w / 2;
  const myCenterY = myRect.y + myRect.h / 2;

  const pushX = myCenterX - face.cx;
  const pushY = myCenterY - face.cy;
  const dist = Math.sqrt(pushX * pushX + pushY * pushY) || 1;
  const strength = overlap * 0.3;

  const rawOffsetX = (pushX / dist) * strength;
  const rawOffsetY = (pushY / dist) * strength;

  const springProgress = spring({ frame, fps, durationInFrames: 15 });

  return {
    offsetX: rawOffsetX * springProgress,
    offsetY: rawOffsetY * springProgress,
  };
}

export function useZoomCompensation(
  normX: number,
  normY: number,
): { adjustedX: number; adjustedY: number; adjustedScale: number } {
  const { zoom, tx, ty } = useZoomFrame();
  if (zoom <= 1.001)
    return { adjustedX: normX, adjustedY: normY, adjustedScale: 1 };

  // Replicate OpenCV affine: mapped_x = zoom * normX + (0.5 - tx * zoom)
  const sx = 0.5 - tx * zoom;
  const sy = 0.5 - ty * zoom;
  return {
    adjustedX: zoom * normX + sx,
    adjustedY: zoom * normY + sy,
    adjustedScale: zoom,
  };
}

export function useFaceAwareLayout(
  staticBounds: NormalizedRect,
  anchor: AnchorMode = "static",
): { left: number; top: number; scale: number; maxWidth: number; maxHeight: number } {
  const face = useFaceFrame();
  const frame = useCurrentFrame();
  const { fps, width, height } = useVideoConfig();

  const compW = staticBounds.w;
  const compH = staticBounds.h;

  // Proportional scale relative to face width (0.25 = baseline "normal" face)
  const faceScale = face
    ? Math.min(1.4, Math.max(0.6, face.fw / 0.25))
    : 1;

  const springProgress = spring({ frame, fps, durationInFrames: 15 });

  let finalNormX: number;
  let finalNormY: number;
  let currentScale: number;

  // Fallback: static bounds with strengthened face avoidance + clamping
  if (!face || anchor === "static") {
    const { offsetX, offsetY } = useFaceAvoidanceRaw(
      staticBounds,
      face,
      springProgress,
    );
    const clamped = clampBounds(
      staticBounds.x + offsetX,
      staticBounds.y + offsetY,
      compW,
      compH,
    );
    finalNormX = clamped.x;
    finalNormY = clamped.y;
    currentScale = face ? faceScale : 1;
  } else {
    let nx: number;
    let ny: number;

    if (anchor === "face-right") {
      nx = face.cx + face.fw / 2 + PADDING;
      ny = face.cy - compH / 2;
    } else if (anchor === "face-left") {
      nx = face.cx - face.fw / 2 - PADDING - compW;
      ny = face.cy - compH / 2;
    } else if (anchor === "face-below") {
      nx = face.cx - compW / 2;
      ny = face.cy + face.fh / 2 + PADDING;
    } else if (anchor === "face-above") {
      nx = face.cx - compW / 2;
      ny = face.cy - face.fh / 2 - PADDING - compH;
    } else {
      // face-beside: pick the side with more room
      const spaceRight = 1 - (face.cx + face.fw / 2);
      const spaceLeft = face.cx - face.fw / 2;
      if (spaceRight >= spaceLeft) {
        nx = face.cx + face.fw / 2 + PADDING;
      } else {
        nx = face.cx - face.fw / 2 - PADDING - compW;
      }
      ny = face.cy - compH / 2;
    }

    const clamped = clampBounds(nx, ny, compW, compH);

    // Animate from static bounds toward face-relative position
    finalNormX =
      staticBounds.x + (clamped.x - staticBounds.x) * springProgress;
    finalNormY =
      staticBounds.y + (clamped.y - staticBounds.y) * springProgress;
    currentScale = faceScale;
  }

  // Apply zoom compensation so overlays track with zoomed content
  const { adjustedX, adjustedY, adjustedScale } = useZoomCompensation(
    finalNormX,
    finalNormY,
  );

  return {
    left: adjustedX * width,
    top: adjustedY * height,
    scale: currentScale * adjustedScale,
    maxWidth: (EDGE_MAX - adjustedX) * width,
    maxHeight: (EDGE_MAX - adjustedY) * height,
  };
}

function useFaceAvoidanceRaw(
  myBounds: NormalizedRect,
  face: FaceFrame | null,
  springProgress: number,
): { offsetX: number; offsetY: number } {
  if (!face) return { offsetX: 0, offsetY: 0 };

  const faceRect = faceToRect(face);
  const myRect: Rect = {
    x: myBounds.x,
    y: myBounds.y,
    w: myBounds.w,
    h: myBounds.h,
  };

  const overlap = computeOverlap(myRect, faceRect);
  if (overlap < 0.01) return { offsetX: 0, offsetY: 0 };

  const myCenterX = myRect.x + myRect.w / 2;
  const myCenterY = myRect.y + myRect.h / 2;

  const pushX = myCenterX - face.cx;
  const pushY = myCenterY - face.cy;
  const dist = Math.sqrt(pushX * pushX + pushY * pushY) || 1;
  const strength = overlap * 0.3;

  return {
    offsetX: (pushX / dist) * strength * springProgress,
    offsetY: (pushY / dist) * strength * springProgress,
  };
}
