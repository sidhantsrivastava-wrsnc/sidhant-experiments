import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
  Easing,
} from "remotion";
import type { AnimatedTitleProps } from "../types";
import { useFaceAwareLayout } from "../lib/spatial";
import { useStyle } from "../lib/styles";
import { SPRING_GENTLE, SPRING_BOUNCY } from "../lib/easing";

export const AnimatedTitle: React.FC<AnimatedTitleProps> = ({
  text,
  style,
  position,
  anchor,
  fontSize = 64,
  color = "#FFFFFF",
  fontWeight,
}) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames, width, height } = useVideoConfig();
  const { left, top, scale, maxWidth } = useFaceAwareLayout(position, anchor);
  const scaledFontSize = fontSize * scale;
  const style_ = useStyle();
  const resolvedWeight = fontWeight ?? style_.font_weights.heading;

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

    return (
      <div
        style={{
          position: "absolute",
          left,
          top,
          maxWidth,
          fontSize: scaledFontSize,
          color,
          fontWeight: resolvedWeight,
          fontFamily: style_.font_family,
          opacity,
          overflow: "hidden",
          textShadow: style_.text_shadow,
        }}
      >
        {displayText}
        <span style={{ opacity: frame % (fps / 2) < fps / 4 ? 1 : 0 }}>|</span>
      </div>
    );
  } else if (style === "bounce") {
    const progress = spring({ frame, fps, config: SPRING_BOUNCY });
    const bounceScale = interpolate(progress, [0, 1], [0.3, 1]);
    opacity = exitOpacity * progress;

    return (
      <div
        style={{
          position: "absolute",
          left,
          top,
          maxWidth,
          fontSize: scaledFontSize,
          color,
          fontWeight: resolvedWeight,
          fontFamily: style_.font_family,
          opacity,
          transform: `scale(${bounceScale})`,
          transformOrigin: "left top",
          overflow: "hidden",
          textShadow: style_.text_shadow,
        }}
      >
        {text}
      </div>
    );
  }

  return (
    <div
      style={{
        position: "absolute",
        left: left + translateX,
        top: top + translateY,
        maxWidth,
        fontSize: scaledFontSize,
        color,
        fontWeight: resolvedWeight,
        fontFamily: style_.font_family,
        opacity,
        overflow: "hidden",
        textShadow: style_.text_shadow,
      }}
    >
      {text}
    </div>
  );
};
