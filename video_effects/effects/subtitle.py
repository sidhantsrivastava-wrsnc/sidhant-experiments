import numpy as np
import cv2

from video_effects.effects.base import BaseEffect, EffectContext
from video_effects.schemas.effects import EffectCue, VideoInfo


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color (#RRGGBB or #RRGGBBAA) to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b)


def _hex_to_rgba(hex_color: str) -> tuple[int, int, int, int]:
    """Convert hex color to RGBA tuple."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    a = int(hex_color[6:8], 16) if len(hex_color) >= 8 else 255
    return (r, g, b, a)


class SubtitleEffect(BaseEffect):
    """Subtitle burn-in / overlay effect."""

    def __init__(self):
        super().__init__()
        self._video_info: VideoInfo | None = None

    def setup(self, video_info: VideoInfo, effect_cues: list[EffectCue],
              *, cache_dir: str | None = None, video_path: str | None = None) -> None:
        self._cues = effect_cues
        self._video_info = video_info

    def apply_frame(
        self, frame: np.ndarray, timestamp: float, context: EffectContext
    ) -> np.ndarray:
        active_cues = self.get_active_cues(timestamp)
        if not active_cues:
            return frame

        result = frame.copy()

        for cue in active_cues:
            params = cue.subtitle_params
            if params is None or not params.text:
                continue

            result = self._render_subtitle(result, params)

        return result

    def _render_subtitle(self, frame: np.ndarray, params) -> np.ndarray:
        """Render subtitle text onto frame with background box."""
        h, w = frame.shape[:2]

        text = params.text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = params.font_size / 30.0  # Normalize to cv2 scale
        thickness = 2 if params.bold else 1
        color = _hex_to_rgb(params.color)

        # Measure text
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Position
        padding = 12
        if params.position == "bottom":
            text_y = h - 40 - baseline
        elif params.position == "top":
            text_y = 40 + text_h
        else:  # center
            text_y = h // 2 + text_h // 2

        text_x = (w - text_w) // 2

        # Background rectangle
        if params.background_color:
            bg_color = _hex_to_rgba(params.background_color)
            alpha = bg_color[3] / 255.0
            x1 = text_x - padding
            y1 = text_y - text_h - padding
            x2 = text_x + text_w + padding
            y2 = text_y + baseline + padding

            # Clamp to frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            overlay = frame[y1:y2, x1:x2].copy()
            cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color[:3], -1)
            frame[y1:y2, x1:x2] = cv2.addWeighted(
                frame[y1:y2, x1:x2], alpha, overlay, 1.0 - alpha, 0
            )

        # Draw text
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)

        return frame
