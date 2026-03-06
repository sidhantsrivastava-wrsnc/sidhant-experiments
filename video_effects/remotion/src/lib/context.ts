import React, { createContext, useContext, useMemo } from "react";
import { useCurrentFrame } from "remotion";
import type { FaceFrame } from "../types";

interface FaceDataContextValue {
  frames: FaceFrame[];
}

const FaceDataContext = createContext<FaceDataContextValue>({ frames: [] });

export const FaceDataProvider: React.FC<{
  faceData: FaceFrame[];
  children: React.ReactNode;
}> = ({ faceData, children }) => {
  const value = useMemo(() => ({ frames: faceData }), [faceData]);
  return React.createElement(
    FaceDataContext.Provider,
    { value },
    children,
  );
};

export function useFaceFrame(): FaceFrame | null {
  const { frames } = useContext(FaceDataContext);
  const frame = useCurrentFrame();
  if (frames.length === 0) return null;
  const idx = Math.min(frame, frames.length - 1);
  return frames[idx];
}
