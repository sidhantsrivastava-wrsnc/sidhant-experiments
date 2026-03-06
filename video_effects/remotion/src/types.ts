export interface FaceFrame {
  cx: number;
  cy: number;
  fw: number;
  fh: number;
}

export interface NormalizedRect {
  x: number;
  y: number;
  w: number;
  h: number;
  label?: string;
}

export interface ComponentSpec {
  template: string;
  startFrame: number;
  durationInFrames: number;
  props: Record<string, unknown>;
  bounds: NormalizedRect;
  zIndex: number;
}

export interface CompositionPlan {
  components: ComponentSpec[];
  colorPalette: string[];
  includeBaseVideo: boolean;
  baseVideoPath?: string;
  faceDataPath?: string;
  /** Passed from Python to size the composition dynamically. */
  durationInFrames?: number;
  fps?: number;
  width?: number;
  height?: number;
}

export interface AnimatedTitleProps {
  text: string;
  style: "fade" | "slide-in" | "typewriter" | "bounce";
  position: NormalizedRect;
  fontSize?: number;
  color?: string;
  fontWeight?: string;
}
