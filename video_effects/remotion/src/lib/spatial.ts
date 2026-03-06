import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { useFaceFrame } from "./context";
import type { FaceFrame, NormalizedRect } from "../types";

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
  const strength = overlap * 0.1;

  const rawOffsetX = (pushX / dist) * strength;
  const rawOffsetY = (pushY / dist) * strength;

  const springProgress = spring({ frame, fps, durationInFrames: 15 });

  return {
    offsetX: rawOffsetX * springProgress,
    offsetY: rawOffsetY * springProgress,
  };
}
