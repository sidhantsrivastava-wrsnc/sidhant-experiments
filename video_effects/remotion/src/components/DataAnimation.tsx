import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import type { DataAnimationProps } from "../types";
import { useFaceAwareLayout } from "../lib/spatial";
import { useStyle } from "../lib/styles";
import { SPRING_SNAPPY, SPRING_GENTLE } from "../lib/easing";

export const DataAnimation: React.FC<DataAnimationProps> = ({
  style,
  value,
  label,
  position,
  anchor,
  startValue = 0,
  suffix = "",
  prefix = "",
  delta,
  items,
  fontSize = 48,
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

  const progress = spring({ frame, fps, config: SPRING_SNAPPY });

  const animatedValue = interpolate(progress, [0, 1], [startValue, value]);
  const isInteger = Number.isInteger(value) && Number.isInteger(startValue);
  const displayValue = isInteger
    ? Math.round(animatedValue).toLocaleString()
    : animatedValue.toFixed(1);

  if (style === "counter") {
    return (
      <div
        style={{
          position: "absolute",
          left,
          top,
          maxWidth,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          overflow: "hidden",
          opacity: progress * exitOpacity,
        }}
      >
        <div
          style={{
            fontSize: scaledFontSize,
            color,
            fontWeight: s.font_weights.emphasis,
            fontFamily: s.font_family,
            textShadow: s.text_shadow,
            whiteSpace: "nowrap",
          }}
        >
          {prefix}{displayValue}{suffix}
        </div>
        <div
          style={{
            fontSize: scaledFontSize * 0.4,
            color,
            fontWeight: s.font_weights.body,
            fontFamily: s.font_family,
            opacity: 0.7,
            textShadow: s.text_shadow,
            marginTop: 4,
          }}
        >
          {label}
        </div>
      </div>
    );
  }

  if (style === "stat-callout") {
    const deltaColor =
      delta !== undefined && delta !== null
        ? delta >= 0
          ? "#4ADE80"
          : "#F87171"
        : undefined;
    const deltaArrow =
      delta !== undefined && delta !== null
        ? delta >= 0
          ? "\u2191"
          : "\u2193"
        : "";
    const deltaText =
      delta !== undefined && delta !== null
        ? `${deltaArrow} ${Math.abs(delta)}%`
        : "";

    return (
      <div
        style={{
          position: "absolute",
          left,
          top,
          maxWidth,
          display: "flex",
          flexDirection: "column",
          alignItems: "flex-start",
          overflow: "hidden",
          opacity: progress * exitOpacity,
        }}
      >
        <div
          style={{
            fontSize: scaledFontSize,
            color,
            fontWeight: s.font_weights.emphasis,
            fontFamily: s.font_family,
            textShadow: s.text_shadow,
            whiteSpace: "nowrap",
            display: "flex",
            alignItems: "baseline",
            gap: 12,
          }}
        >
          <span>
            {prefix}{displayValue}{suffix}
          </span>
          {deltaText && (
            <span
              style={{
                fontSize: scaledFontSize * 0.4,
                color: deltaColor,
                fontWeight: s.font_weights.emphasis,
              }}
            >
              {deltaText}
            </span>
          )}
        </div>
        <div
          style={{
            fontSize: scaledFontSize * 0.4,
            color,
            fontWeight: s.font_weights.body,
            fontFamily: s.font_family,
            opacity: 0.7,
            textShadow: s.text_shadow,
            marginTop: 4,
          }}
        >
          {label}
        </div>
      </div>
    );
  }

  // bar style
  const barItems = items ?? [{ label, value }];
  const maxBarValue = Math.max(...barItems.map((it) => it.value), 1);
  const barHeight = scaledFontSize * 0.7;
  const maxBarWidth = position.w * width * 0.8;

  return (
    <div
      style={{
        position: "absolute",
        left,
        top,
        maxWidth,
        display: "flex",
        flexDirection: "column",
        gap: barHeight * 0.6,
        overflow: "hidden",
        opacity: exitOpacity,
      }}
    >
      {barItems.slice(0, 4).map((item, index) => {
        const delay = index * 8;
        const barProgress = spring({
          frame: Math.max(0, frame - delay),
          fps,
          config: SPRING_GENTLE,
        });
        const barWidth = interpolate(
          barProgress,
          [0, 1],
          [0, (item.value / maxBarValue) * maxBarWidth],
        );

        return (
          <div
            key={index}
            style={{
              display: "flex",
              flexDirection: "column",
              gap: 4,
              opacity: frame >= delay ? barProgress : 0,
            }}
          >
            <div
              style={{
                fontSize: scaledFontSize * 0.35,
                color,
                fontWeight: s.font_weights.body,
                fontFamily: s.font_family,
                textShadow: s.text_shadow,
                whiteSpace: "nowrap",
              }}
            >
              {item.label}
            </div>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
              }}
            >
              <div
                style={{
                  width: barWidth,
                  height: barHeight,
                  backgroundColor: accentColor,
                  borderRadius: barHeight / 4,
                }}
              />
              <span
                style={{
                  fontSize: scaledFontSize * 0.35,
                  color,
                  fontWeight: s.font_weights.marker,
                  fontFamily: s.font_family,
                  textShadow: s.text_shadow,
                  opacity: barProgress,
                }}
              >
                {prefix}{Math.round(interpolate(barProgress, [0, 1], [0, item.value]))}{suffix}
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
};
