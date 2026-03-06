import { spring } from "remotion";

export const SPRING_GENTLE = {
  damping: 15,
  mass: 0.8,
  stiffness: 80,
} as const;

export const SPRING_BOUNCY = {
  damping: 10,
  mass: 0.6,
  stiffness: 120,
} as const;

export const SPRING_SNAPPY = {
  damping: 20,
  mass: 0.5,
  stiffness: 200,
} as const;

export function gentleSpring(frame: number, fps: number): number {
  return spring({ frame, fps, config: SPRING_GENTLE });
}

export function bouncySpring(frame: number, fps: number): number {
  return spring({ frame, fps, config: SPRING_BOUNCY });
}

export function snappySpring(frame: number, fps: number): number {
  return spring({ frame, fps, config: SPRING_SNAPPY });
}
