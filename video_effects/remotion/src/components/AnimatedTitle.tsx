import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
  Easing,
} from "remotion";
import type { AnimatedTitleProps, NormalizedRect } from "../types";
import { useFaceAvoidance } from "../lib/spatial";
import { SPRING_GENTLE, SPRING_BOUNCY } from "../lib/easing";

export const AnimatedTitle: React.FC<AnimatedTitleProps> = ({
  text,
  style,
  position,
  fontSize = 64,
  color = "#FFFFFF",
  fontWeight = "700",
}) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames, width, height } = useVideoConfig();
  const { offsetX, offsetY } = useFaceAvoidance(position);

  const fadeOutStart = durationInFrames - Math.round(fps * 0.5);

  const exitOpacity = interpolate(
    frame,
    [fadeOutStart, durationInFrames],
    [1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" },
  );

  let opacity = 1;
  let translateX = 0;
  let translateY = 0;

  if (style === "fade") {
    const enterOpacity = interpolate(frame, [0, Math.round(fps * 0.5)], [0, 1], {
      extrapolateRight: "clamp",
    });
    opacity = Math.min(enterOpacity, exitOpacity);
  } else if (style === "slide-in") {
    const progress = spring({ frame, fps, config: SPRING_GENTLE });
    translateX = interpolate(progress, [0, 1], [-100, 0]);
    opacity = exitOpacity * progress;
  } else if (style === "typewriter") {
    const charsToShow = Math.floor(
      interpolate(frame, [0, Math.min(text.length * 2, durationInFrames * 0.6)], [0, text.length], {
        extrapolateRight: "clamp",
      }),
    );
    const displayText = text.slice(0, charsToShow);
    opacity = exitOpacity;

    const left = position.x * width + offsetX * width;
    const top = position.y * height + offsetY * height;

    return (
      <div
        style={{
          position: "absolute",
          left,
          top,
          fontSize,
          color,
          fontWeight,
          fontFamily: "sans-serif",
          opacity,
          whiteSpace: "nowrap",
          textShadow: "0 2px 8px rgba(0,0,0,0.6)",
        }}
      >
        {displayText}
        <span style={{ opacity: frame % (fps / 2) < fps / 4 ? 1 : 0 }}>|</span>
      </div>
    );
  } else if (style === "bounce") {
    const progress = spring({ frame, fps, config: SPRING_BOUNCY });
    const scale = interpolate(progress, [0, 1], [0.3, 1]);
    opacity = exitOpacity * progress;

    const left = position.x * width + offsetX * width;
    const top = position.y * height + offsetY * height;

    return (
      <div
        style={{
          position: "absolute",
          left,
          top,
          fontSize,
          color,
          fontWeight,
          fontFamily: "sans-serif",
          opacity,
          transform: `scale(${scale})`,
          transformOrigin: "left top",
          whiteSpace: "nowrap",
          textShadow: "0 2px 8px rgba(0,0,0,0.6)",
        }}
      >
        {text}
      </div>
    );
  }

  const left = position.x * width + (translateX + offsetX * width);
  const top = position.y * height + (translateY + offsetY * height);

  return (
    <div
      style={{
        position: "absolute",
        left,
        top,
        fontSize,
        color,
        fontWeight,
        fontFamily: "sans-serif",
        opacity,
        whiteSpace: "nowrap",
        textShadow: "0 2px 8px rgba(0,0,0,0.6)",
      }}
    >
      {text}
    </div>
  );
};
