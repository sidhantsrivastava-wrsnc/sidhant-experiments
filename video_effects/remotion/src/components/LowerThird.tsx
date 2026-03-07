import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import type { LowerThirdProps } from "../types";
import { useFaceAwareLayout } from "../lib/spatial";
import { useStyle } from "../lib/styles";
import { SPRING_GENTLE, SPRING_SMOOTH } from "../lib/easing";

export const LowerThird: React.FC<LowerThirdProps> = ({
  name,
  title,
  accentColor = "#FFD700",
  style = "slide",
  position,
  anchor,
  fontSize = 36,
  color = "#FFFFFF",
}) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames, width, height } = useVideoConfig();
  const { left, top, scale, maxWidth } = useFaceAwareLayout(position, anchor);
  const scaledFontSize = fontSize * scale;
  const s = useStyle();

  const fadeOutStart = durationInFrames - Math.round(fps * 0.5);
  const exitOpacity = interpolate(
    frame,
    [fadeOutStart, durationInFrames],
    [1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" },
  );

  if (style === "slide") {
    const barProgress = spring({ frame, fps, config: SPRING_GENTLE });
    const barWidth = interpolate(barProgress, [0, 1], [0, 4]);

    const nameOpacity = interpolate(barProgress, [0.3, 0.6], [0, 1], {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    });
    const titleOpacity = interpolate(barProgress, [0.6, 0.9], [0, 1], {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    });
    const textTranslateX = interpolate(barProgress, [0.3, 1], [20, 0], {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    });

    return (
      <div
        style={{
          position: "absolute",
          left,
          top,
          maxWidth,
          display: "flex",
          flexDirection: "row",
          alignItems: "stretch",
          overflow: "hidden",
          opacity: exitOpacity,
        }}
      >
        <div
          style={{
            width: barWidth,
            backgroundColor: accentColor,
            borderRadius: 2,
            minHeight: title ? scaledFontSize * 2.2 : scaledFontSize * 1.4,
          }}
        />
        <div
          style={{
            marginLeft: 12,
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
          }}
        >
          <div
            style={{
              fontSize: scaledFontSize,
              color,
              fontWeight: s.font_weights.heading,
              fontFamily: s.font_family,
              opacity: nameOpacity,
              transform: `translateX(${textTranslateX}px)`,
              whiteSpace: "nowrap",
              textShadow: s.text_shadow,
            }}
          >
            {name}
          </div>
          {title && (
            <div
              style={{
                fontSize: scaledFontSize * 0.65,
                color,
                fontWeight: s.font_weights.body,
                fontFamily: s.font_family,
                opacity: titleOpacity * 0.75,
                transform: `translateX(${textTranslateX}px)`,
                whiteSpace: "nowrap",
                textShadow: s.text_shadow,
                marginTop: 2,
              }}
            >
              {title}
            </div>
          )}
        </div>
      </div>
    );
  }

  // fade style
  const fadeProgress = spring({ frame, fps, config: SPRING_SMOOTH });

  return (
    <div
      style={{
        position: "absolute",
        left,
        top,
        maxWidth,
        display: "flex",
        flexDirection: "row",
        alignItems: "stretch",
        overflow: "hidden",
        opacity: fadeProgress * exitOpacity,
      }}
    >
      <div
        style={{
          width: 4,
          backgroundColor: accentColor,
          borderRadius: 2,
          minHeight: title ? scaledFontSize * 2.2 : scaledFontSize * 1.4,
        }}
      />
      <div
        style={{
          marginLeft: 12,
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
        }}
      >
        <div
          style={{
            fontSize: scaledFontSize,
            color,
            fontWeight: s.font_weights.heading,
            fontFamily: s.font_family,
            whiteSpace: "nowrap",
            textShadow: s.text_shadow,
          }}
        >
          {name}
        </div>
        {title && (
          <div
            style={{
              fontSize: scaledFontSize * 0.65,
              color,
              fontWeight: s.font_weights.body,
              fontFamily: s.font_family,
              opacity: 0.75,
              whiteSpace: "nowrap",
              textShadow: s.text_shadow,
              marginTop: 2,
            }}
          >
            {title}
          </div>
        )}
      </div>
    </div>
  );
};
