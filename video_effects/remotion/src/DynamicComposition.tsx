import React, { useEffect, useState } from "react";
import {
  AbsoluteFill,
  Sequence,
  OffthreadVideo,
  staticFile,
  continueRender,
  delayRender,
} from "remotion";
import { FaceDataProvider } from "./lib/context";
import { ComponentRegistry } from "./components";
import type { CompositionPlan, FaceFrame } from "./types";

export const DynamicComposition: React.FC<CompositionPlan> = ({
  components,
  colorPalette,
  includeBaseVideo,
  baseVideoPath,
  faceDataPath,
}) => {
  const [faceData, setFaceData] = useState<FaceFrame[]>([]);
  const [handle] = useState(() => delayRender());

  useEffect(() => {
    if (faceDataPath) {
      fetch(faceDataPath)
        .then((r) => r.json())
        .then((data: number[][]) => {
          const frames: FaceFrame[] = data.map((d) => ({
            cx: d[0],
            cy: d[1],
            fw: d[2],
            fh: d[3],
          }));
          setFaceData(frames);
          continueRender(handle);
        })
        .catch(() => {
          continueRender(handle);
        });
    } else {
      continueRender(handle);
    }
  }, [faceDataPath, handle]);

  const sorted = [...components].sort((a, b) => a.zIndex - b.zIndex);

  return (
    <FaceDataProvider faceData={faceData}>
      <AbsoluteFill style={{ backgroundColor: "transparent" }}>
        {includeBaseVideo && baseVideoPath && (
          <OffthreadVideo src={baseVideoPath} />
        )}
        {sorted.map((comp, i) => {
          const Component = ComponentRegistry[comp.template];
          if (!Component) return null;
          return (
            <Sequence
              key={i}
              from={comp.startFrame}
              durationInFrames={comp.durationInFrames}
            >
              <Component {...comp.props} position={comp.bounds} />
            </Sequence>
          );
        })}
      </AbsoluteFill>
    </FaceDataProvider>
  );
};
