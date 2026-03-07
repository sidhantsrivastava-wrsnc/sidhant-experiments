import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import type { ListicleProps } from "../types";
import { useFaceAwareLayout } from "../lib/spatial";
import { useStyle } from "../lib/styles";
import { SPRING_BOUNCY, SPRING_GENTLE } from "../lib/easing";

export const Listicle: React.FC<ListicleProps> = ({
  items,
  style = "pop",
  listStyle = "numbered",
  position,
  anchor,
  staggerDelay = 10,
  fontSize = 32,
  color = "#FFFFFF",
  accentColor = "#FFD700",
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

  const springConfig = style === "pop" ? SPRING_BOUNCY : SPRING_GENTLE;

  const getMarker = (index: number): string => {
    if (listStyle === "numbered") return `${index + 1}.`;
    if (listStyle === "bullet") return "\u2022";
    return "";
  };

  return (
    <div
      style={{
        position: "absolute",
        left,
        top,
        maxWidth,
        display: "flex",
        flexDirection: "column",
        gap: scaledFontSize * 0.5,
        overflow: "hidden",
        opacity: exitOpacity,
      }}
    >
      {items.map((item, index) => {
        const delay = index * staggerDelay;
        const progress = spring({
          frame: Math.max(0, frame - delay),
          fps,
          config: springConfig,
        });

        const itemOpacity = frame >= delay ? progress : 0;
        const itemScale = style === "pop"
          ? interpolate(progress, [0, 1], [0, 1])
          : 1;
        const itemTranslateX = style === "slide"
          ? interpolate(progress, [0, 1], [-40, 0])
          : 0;

        const marker = getMarker(index);

        return (
          <div
            key={index}
            style={{
              display: "flex",
              flexDirection: "row",
              alignItems: "baseline",
              gap: scaledFontSize * 0.4,
              opacity: itemOpacity,
              transform:
                style === "pop"
                  ? `scale(${itemScale})`
                  : `translateX(${itemTranslateX}px)`,
              transformOrigin: "left center",
            }}
          >
            {marker && (
              <span
                style={{
                  fontSize: scaledFontSize * 0.85,
                  color: accentColor,
                  fontWeight: s.font_weights.marker,
                  fontFamily: s.font_family,
                  textShadow: s.text_shadow,
                }}
              >
                {marker}
              </span>
            )}
            <span
              style={{
                fontSize: scaledFontSize,
                color,
                fontWeight: s.font_weights.body,
                fontFamily: s.font_family,
                whiteSpace: "nowrap",
                textShadow: s.text_shadow,
              }}
            >
              {item}
            </span>
          </div>
        );
      })}
    </div>
  );
};
