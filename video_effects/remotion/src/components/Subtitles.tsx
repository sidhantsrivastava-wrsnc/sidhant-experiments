import React, { useMemo } from "react";
import { useCurrentFrame, useVideoConfig, interpolate } from "remotion";
import type { SubtitlesProps, SubtitleWord } from "../types";
import { useFaceAwareLayout } from "../lib/spatial";
import { useStyle } from "../lib/styles";

interface Page {
  words: SubtitleWord[];
  startFrame: number;
  endFrame: number;
}

/**
 * Estimate rendered width of a string in pixels.
 * Average character width is ~0.55 * fontSize for proportional fonts.
 */
function estimateTextWidth(text: string, fontSize: number): number {
  return text.length * fontSize * 0.6;
}

/**
 * Build pages dynamically — pack as many words as fit on one line,
 * allowing up to `maxLines` lines per page.
 */
function buildDynamicPages(
  words: SubtitleWord[],
  availableWidth: number,
  fontSize: number,
  maxLines: number,
): Page[] {
  const pages: Page[] = [];
  const wordGap = fontSize * 0.3; // gap between words
  let pageWords: SubtitleWord[] = [];
  let lineWidth = 0;
  let lineCount = 1;

  for (const word of words) {
    const wordWidth = estimateTextWidth(word.text, fontSize);
    const neededWidth = lineWidth > 0 ? wordGap + wordWidth : wordWidth;

    if (lineWidth + neededWidth > availableWidth && lineWidth > 0) {
      // Current line is full — try a new line
      if (lineCount < maxLines) {
        lineCount++;
        lineWidth = wordWidth;
        pageWords.push(word);
      } else {
        // Page is full — flush and start a new page
        if (pageWords.length > 0) {
          pages.push({
            words: pageWords,
            startFrame: pageWords[0].startFrame,
            endFrame: pageWords[pageWords.length - 1].endFrame,
          });
        }
        pageWords = [word];
        lineWidth = wordWidth;
        lineCount = 1;
      }
    } else {
      lineWidth += neededWidth;
      pageWords.push(word);
    }
  }

  // Flush remaining
  if (pageWords.length > 0) {
    pages.push({
      words: pageWords,
      startFrame: pageWords[0].startFrame,
      endFrame: pageWords[pageWords.length - 1].endFrame,
    });
  }

  return pages;
}

export const Subtitles: React.FC<SubtitlesProps> = ({
  words,
  position,
  anchor,
  fontSize = 48,
  color = "#FFFFFF",
  highlightColor,
  backgroundColor = "rgba(0, 0, 0, 0.6)",
}) => {
  const frame = useCurrentFrame();
  const { fps, width, height } = useVideoConfig();
  const { left, top, scale, maxWidth } = useFaceAwareLayout(position, anchor);
  const style_ = useStyle();
  const scaledFontSize = fontSize * scale;

  const resolvedHighlight =
    highlightColor ?? style_.palette?.[2] ?? "#FFD700";

  // Available width for text (account for padding)
  const containerPadding = scaledFontSize * 0.4 * 2;
  const availableTextWidth =
    Math.min(maxWidth, width * 0.85) - containerPadding;

  const pages = useMemo(
    () => buildDynamicPages(words, availableTextWidth, scaledFontSize, 2),
    [words, availableTextWidth, scaledFontSize],
  );

  // Find the active page for this frame
  const activePage = pages.find(
    (p) => frame >= p.startFrame && frame <= p.endFrame,
  );

  if (!activePage) return null;

  // Fade in/out for the page
  const fadeInFrames = Math.round(fps * 0.15);
  const fadeOutFrames = Math.round(fps * 0.15);

  const enterOpacity = interpolate(
    frame,
    [activePage.startFrame, activePage.startFrame + fadeInFrames],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" },
  );
  const exitOpacity = interpolate(
    frame,
    [activePage.endFrame - fadeOutFrames, activePage.endFrame],
    [1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" },
  );
  const opacity = Math.min(enterOpacity, exitOpacity);

  // Subtle scale entrance
  const scaleIn = interpolate(
    frame,
    [activePage.startFrame, activePage.startFrame + fadeInFrames],
    [0.95, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" },
  );

  return (
    <div
      style={{
        position: "absolute",
        left,
        top,
        maxWidth: Math.min(maxWidth, width * 0.85),
        maxHeight: position.h * height,
        overflow: "hidden",
        opacity,
        transform: `scale(${scaleIn})`,
        transformOrigin: "center center",
        display: "flex",
        flexWrap: "wrap",
        justifyContent: "center",
        gap: `${scaledFontSize * 0.3}px`,
        padding: `${scaledFontSize * 0.25}px ${scaledFontSize * 0.4}px`,
        borderRadius: scaledFontSize * 0.2,
        backgroundColor,
      }}
    >
      {activePage.words.map((word, i) => {
        const isActive = frame >= word.startFrame && frame <= word.endFrame;
        const isPast = frame > word.endFrame;

        const wordColor = isActive
          ? resolvedHighlight
          : isPast
            ? color
            : `${color}99`;

        const wordScale = isActive ? 1.08 : 1;
        const wordWeight = isActive
          ? (style_.font_weights?.emphasis ?? "700")
          : (style_.font_weights?.body ?? "400");

        return (
          <span
            key={i}
            style={{
              fontSize: scaledFontSize,
              color: wordColor,
              fontWeight: wordWeight,
              fontFamily: style_.font_family,
              textShadow: style_.text_shadow,
              transform: `scale(${wordScale})`,
              transition: "color 0.1s, transform 0.1s",
              whiteSpace: "pre",
            }}
          >
            {word.text}
          </span>
        );
      })}
    </div>
  );
};
