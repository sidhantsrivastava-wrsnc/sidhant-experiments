import math

import numpy as np
import cv2

from video_effects.effects.base import BaseEffect, EffectContext
from video_effects.schemas.effects import EffectCue, VideoInfo


class ZoomEffect(BaseEffect):
    """Zoom effect with face-tracked, center, or point tracking."""

    def __init__(self):
        super().__init__()
        self._face_data: list[tuple[float, float, float, float]] | None = None
        self._video_info: VideoInfo | None = None
        self._hold_intervals: list[tuple[float, float, float, str]] = []

    def setup(self, video_info: VideoInfo, effect_cues: list[EffectCue]) -> None:
        self._cues = effect_cues
        self._video_info = video_info

        # Pre-compute face tracking for face-tracked zoom cues
        face_cues = [
            c for c in effect_cues
            if c.zoom_params and c.zoom_params.tracking == "face"
        ]
        if face_cues:
            self._setup_face_tracking(video_info, face_cues)

        self._build_hold_intervals()

    def _build_hold_intervals(self) -> None:
        """Pair each 'in' cue with the next 'out' cue to create hold intervals."""
        zoom_cues = [
            c for c in self._cues
            if c.zoom_params is not None
        ]
        zoom_cues.sort(key=lambda c: c.start_time)

        self._hold_intervals = []
        pending_in: EffectCue | None = None

        for cue in zoom_cues:
            action = cue.zoom_params.action
            if action == "in":
                pending_in = cue
            elif action == "out" and pending_in is not None:
                # Hold between end of "in" and start of "out"
                if pending_in.end_time < cue.start_time:
                    self._hold_intervals.append((
                        pending_in.end_time,
                        cue.start_time,
                        pending_in.zoom_params.zoom_level,
                        pending_in.zoom_params.tracking,
                    ))
                pending_in = None

    def get_active_ranges(self) -> list[tuple[float, float]]:
        """Include hold intervals in active ranges."""
        ranges = super().get_active_ranges()
        for hold_start, hold_end, _, _ in self._hold_intervals:
            ranges.append((hold_start, hold_end))
        return ranges

    def _setup_face_tracking(
        self, video_info: VideoInfo, face_cues: list[EffectCue]
    ) -> None:
        """Run face detection on active ranges."""
        from video_effects.helpers.face_tracking import detect_faces

        active_ranges = [
            (
                int(c.start_time * video_info.fps),
                int(c.end_time * video_info.fps),
            )
            for c in face_cues
        ]
        # face_data will be populated by the helper
        self._face_data = None  # Will be set when we have video path access

    def apply_frame(
        self, frame: np.ndarray, timestamp: float, context: EffectContext
    ) -> np.ndarray:
        # Check hold intervals first (constant zoom between in/out pairs)
        for hold_start, hold_end, hold_zoom, hold_tracking in self._hold_intervals:
            if hold_start <= timestamp <= hold_end:
                h, w = frame.shape[:2]
                if hold_tracking == "center":
                    tx, ty = w / 2, h / 2
                else:
                    tx, ty = w / 2, h / 2
                sx = w / 2 - tx * hold_zoom
                sy = h / 2 - ty * hold_zoom
                M = np.float32([[hold_zoom, 0, sx], [0, hold_zoom, sy]])
                return cv2.warpAffine(
                    frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE
                )

        active_cues = self.get_active_cues(timestamp)
        if not active_cues:
            return frame

        h, w = frame.shape[:2]
        result = frame

        for cue in active_cues:
            params = cue.zoom_params
            if params is None:
                continue

            z = params.zoom_level

            # Compute easing intensity based on position within the cue
            duration = cue.end_time - cue.start_time
            if duration > 0:
                progress = (timestamp - cue.start_time) / duration
            else:
                progress = 1.0

            action = params.action
            if action == "in":
                intensity = self._ease_in(progress, params.easing)
            elif action == "out":
                intensity = self._ease_out(progress, params.easing)
            else:  # bounce
                intensity = self._ease_bounce(progress, params.easing)

            current_zoom = 1.0 + (z - 1.0) * intensity

            if params.tracking == "face" and self._face_data is not None:
                fx, fy, _, _ = self._face_data[context.frame_index]
                tx = self._lerp(w / 2, fx, intensity)
                ty = self._lerp(h / 2, fy, intensity)
            elif params.tracking == "center":
                tx, ty = w / 2, h / 2
            else:
                # point tracking — default to center
                tx, ty = w / 2, h / 2

            # Build affine matrix: zoom centered on (tx, ty)
            sx = w / 2 - tx * current_zoom
            sy = h / 2 - ty * current_zoom
            M = np.float32([[current_zoom, 0, sx], [0, current_zoom, sy]])

            result = cv2.warpAffine(
                result, M, (w, h), borderMode=cv2.BORDER_REPLICATE
            )

        return result

    def _ease_bounce(self, t: float, easing: str) -> float:
        """Symmetric 0 -> 1 -> 0 easing for bounce action."""
        t = max(0.0, min(1.0, t))
        if easing == "snap":
            if t < 0.25:
                return (t / 0.25) ** 2
            elif t > 0.75:
                r = (t - 0.75) / 0.25
                return 1.0 - r ** 2
            return 1.0
        elif easing == "overshoot":
            base = math.sin(math.pi * t)
            overshoot = 0.15 * math.sin(2.0 * math.pi * t)
            return max(0.0, min(1.15, base + overshoot))
        else:  # smooth
            return math.sin(math.pi * t)

    def _ease_in(self, t: float, easing: str) -> float:
        """Monotonic 0 -> 1 easing for zoom-in action."""
        t = max(0.0, min(1.0, t))
        if easing == "snap":
            if t < 0.5:
                return (t / 0.5) ** 2
            return 1.0
        elif easing == "overshoot":
            base = math.sin(math.pi / 2 * t)
            overshoot = 0.15 * math.sin(math.pi * t)
            return max(0.0, min(1.15, base + overshoot))
        else:  # smooth
            return math.sin(math.pi / 2 * t)

    def _ease_out(self, t: float, easing: str) -> float:
        """Monotonic 1 -> 0 easing for zoom-out action."""
        t = max(0.0, min(1.0, t))
        if easing == "snap":
            return max(0.0, 1.0 - t ** 2)
        elif easing == "overshoot":
            base = math.cos(math.pi / 2 * t)
            undershoot = -0.15 * math.sin(math.pi * t)
            return max(0.0, min(1.15, base + undershoot))
        else:  # smooth
            return math.cos(math.pi / 2 * t)
