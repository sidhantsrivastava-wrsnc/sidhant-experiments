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

export type AnchorMode = "static" | "face-right" | "face-left" | "face-below" | "face-above" | "face-beside";

export interface ComponentSpec {
  template: string;
  startFrame: number;
  durationInFrames: number;
  props: Record<string, unknown>;
  bounds: NormalizedRect;
  zIndex: number;
  anchor?: AnchorMode;
}

export interface FontWeights {
  heading: string;
  body: string;
  emphasis: string;
  marker: string;
}

export interface StyleConfig {
  font_family: string;
  font_import: string;
  font_weights_to_load: string[];
  palette: string[];
  text_shadow: string;
  font_weights: FontWeights;
}

export interface ZoomFrame {
  zoom: number;     // current zoom level (1.0 = no zoom)
  tx: number;       // target X normalized (0-1)
  ty: number;       // target Y normalized (0-1)
}

export interface CompositionPlan {
  components: ComponentSpec[];
  colorPalette: string[];
  includeBaseVideo: boolean;
  baseVideoPath?: string;
  faceDataPath?: string;
  zoomStatePath?: string;
  styleConfig?: StyleConfig;
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
  anchor?: AnchorMode;
  fontSize?: number;
  color?: string;
  fontWeight?: string;
}

export interface LowerThirdProps {
  name: string;
  title?: string;
  accentColor?: string;
  style?: "slide" | "fade";
  position: NormalizedRect;
  anchor?: AnchorMode;
  fontSize?: number;
  color?: string;
}

export interface ListicleProps {
  items: string[];
  style?: "pop" | "slide";
  listStyle?: "numbered" | "bullet" | "none";
  position: NormalizedRect;
  anchor?: AnchorMode;
  staggerDelay?: number;
  fontSize?: number;
  color?: string;
  accentColor?: string;
}

export interface SubtitleWord {
  text: string;
  startFrame: number;
  endFrame: number;
}

export interface SubtitlesProps {
  words: SubtitleWord[];
  position: NormalizedRect;
  anchor?: AnchorMode;
  fontSize?: number;
  color?: string;
  highlightColor?: string;
  backgroundColor?: string;
}

export interface DataAnimationProps {
  style: "counter" | "stat-callout" | "bar";
  value: number;
  label: string;
  position: NormalizedRect;
  anchor?: AnchorMode;
  startValue?: number;
  suffix?: string;
  prefix?: string;
  delta?: number;
  items?: { label: string; value: number }[];
  fontSize?: number;
  color?: string;
  accentColor?: string;
}
