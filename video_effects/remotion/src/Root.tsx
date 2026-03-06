import React from "react";
import { Composition } from "remotion";
import { DynamicComposition } from "./DynamicComposition";
import type { CompositionPlan } from "./types";

const defaultProps: CompositionPlan = {
  components: [],
  colorPalette: [],
  includeBaseVideo: false,
};

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="MotionOverlay"
        component={DynamicComposition as unknown as React.FC<Record<string, unknown>>}
        durationInFrames={300}
        fps={30}
        width={1920}
        height={1080}
        defaultProps={defaultProps as unknown as Record<string, unknown>}
        calculateMetadata={async ({ props }) => {
          const p = props as unknown as CompositionPlan;
          return {
            ...(p.durationInFrames ? { durationInFrames: p.durationInFrames } : {}),
            ...(p.fps ? { fps: p.fps } : {}),
            ...(p.width ? { width: p.width } : {}),
            ...(p.height ? { height: p.height } : {}),
          };
        }}
      />
    </>
  );
};
