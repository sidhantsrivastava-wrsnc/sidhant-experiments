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
import { ZoomDataProvider } from "./lib/zoom-context";
import { loadStyleFont } from "./lib/fonts";
import { StyleProvider } from "./lib/styles";
import { ComponentRegistry } from "./components";
import type { CompositionPlan, FaceFrame, StyleConfig, ZoomFrame } from "./types";

export const DynamicComposition: React.FC<CompositionPlan> = ({
  components,
  colorPalette,
  includeBaseVideo,
  baseVideoPath,
  faceDataPath,
  zoomStatePath,
  styleConfig,
}) => {
  const [faceData, setFaceData] = useState<FaceFrame[]>([]);
  const [zoomData, setZoomData] = useState<Map<number, ZoomFrame>>(new Map());
  const [handle] = useState(() => delayRender());

  useEffect(() => {
    let pending = 0;
    const maybeFinish = () => {
      pending--;
      if (pending <= 0) continueRender(handle);
    };

    // Load face data
    if (faceDataPath) {
      pending++;
      fetch(faceDataPath)
        .then((r) => r.json())
        .then((raw: unknown) => {
          // Handle both dict format and old bare-array format
          let faceArray: number[][];
          let normW = 1920;
          let normH = 1080;
          if (Array.isArray(raw)) {
            faceArray = raw;
          } else {
            const obj = raw as Record<string, unknown>;
            faceArray = (obj.face_data as number[][]) || [];
            const dims = obj.dimensions as number[] | undefined;
            if (dims && dims.length >= 2) {
              normW = dims[0];
              normH = dims[1];
            }
          }

          const frames: FaceFrame[] = faceArray.map((d) => ({
            cx: d[0] / normW,
            cy: d[1] / normH,
            fw: d[2] / normW,
            fh: d[3] / normH,
          }));
          setFaceData(frames);
          maybeFinish();
        })
        .catch(() => maybeFinish());
    }

    // Load zoom state
    if (zoomStatePath) {
      pending++;
      fetch(zoomStatePath)
        .then((r) => r.json())
        .then((data: { frames: Record<string, number[]> }) => {
          const map = new Map<number, ZoomFrame>();
          for (const [key, val] of Object.entries(data.frames)) {
            map.set(Number(key), { zoom: val[0], tx: val[1], ty: val[2] });
          }
          setZoomData(map);
          maybeFinish();
        })
        .catch(() => maybeFinish());
    }

    if (pending === 0) continueRender(handle);
  }, [faceDataPath, zoomStatePath, handle]);

  // Resolve font and build final style config
  const resolvedStyle = React.useMemo<StyleConfig | undefined>(() => {
    if (!styleConfig) return undefined;
    if (styleConfig.font_import) {
      const fontFamily = loadStyleFont(
        styleConfig.font_import,
        styleConfig.font_weights_to_load,
      );
      return { ...styleConfig, font_family: fontFamily };
    }
    return styleConfig;
  }, [styleConfig]);

  const sorted = [...components].sort((a, b) => a.zIndex - b.zIndex);

  return (
    <StyleProvider styleConfig={resolvedStyle}>
      <FaceDataProvider faceData={faceData}>
        <ZoomDataProvider zoomData={zoomData}>
          <AbsoluteFill style={{ backgroundColor: "transparent" }}>
            {includeBaseVideo && baseVideoPath && (
              <OffthreadVideo src={baseVideoPath} />
            )}
            {sorted.map((comp, i) => {
              const Component = ComponentRegistry[comp.template];
              if (!Component) return null;
              const b = comp.bounds;
              const clipTop = b.y * 100;
              const clipRight = (1 - b.x - b.w) * 100;
              const clipBottom = (1 - b.y - b.h) * 100;
              const clipLeft = b.x * 100;
              return (
                <Sequence
                  key={i}
                  from={comp.startFrame}
                  durationInFrames={comp.durationInFrames}
                >
                  <div
                    style={{
                      position: "absolute",
                      inset: 0,
                      clipPath: `inset(${clipTop}% ${clipRight}% ${clipBottom}% ${clipLeft}%)`,
                      zIndex: comp.zIndex,
                      ...(comp.shadow ? { filter: `drop-shadow(${comp.shadow})` } : {}),
                    }}
                  >
                    <Component {...comp.props} position={comp.bounds} anchor={comp.anchor} />
                  </div>
                </Sequence>
              );
            })}
          </AbsoluteFill>
        </ZoomDataProvider>
      </FaceDataProvider>
    </StyleProvider>
  );
};
