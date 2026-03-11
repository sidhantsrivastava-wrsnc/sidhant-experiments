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

export const SPRING_SMOOTH = {
  damping: 200,
  mass: 1,
  stiffness: 100,
} as const;

export const SPRING_ELASTIC = {
  damping: 5,
  mass: 0.4,
  stiffness: 300,
} as const;

export const SPRING_WOBBLY = {
  damping: 8,
  mass: 1.0,
  stiffness: 150,
} as const;

export function snappySpring(frame: number, fps: number): number {
  return spring({ frame, fps, config: SPRING_SNAPPY });
}

export function smoothSpring(frame: number, fps: number): number {
  return spring({ frame, fps, config: SPRING_SMOOTH });
}

export function elasticSpring(frame: number, fps: number): number {
  return spring({ frame, fps, config: SPRING_ELASTIC });
}

export function wobblySpring(frame: number, fps: number): number {
  return spring({ frame, fps, config: SPRING_WOBBLY });
}
