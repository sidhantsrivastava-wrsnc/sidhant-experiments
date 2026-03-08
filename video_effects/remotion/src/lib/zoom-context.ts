import React, { createContext, useContext, useMemo } from "react";
import { useCurrentFrame } from "remotion";
import type { ZoomFrame } from "../types";

interface ZoomDataContextValue {
  frames: Map<number, ZoomFrame>;
}

const ZoomDataContext = createContext<ZoomDataContextValue>({
  frames: new Map(),
});

export const ZoomDataProvider: React.FC<{
  zoomData: Map<number, ZoomFrame>;
  children: React.ReactNode;
}> = ({ zoomData, children }) => {
  const value = useMemo(() => ({ frames: zoomData }), [zoomData]);
  return React.createElement(ZoomDataContext.Provider, { value }, children);
};

const DEFAULT_ZOOM_FRAME: ZoomFrame = { zoom: 1, tx: 0.5, ty: 0.5 };

export function useZoomFrame(): ZoomFrame {
  const { frames } = useContext(ZoomDataContext);
  const frame = useCurrentFrame();
  return frames.get(frame) ?? DEFAULT_ZOOM_FRAME;
}
