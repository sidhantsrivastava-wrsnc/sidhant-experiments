import math

import cv2
import numpy as np

from video_effects.effects.base import BaseEffect, EffectContext
from video_effects.schemas.effects import EffectCue, VideoInfo


class WhipEffect(BaseEffect):
    """Directional motion blur transition between sections."""

    def setup(self, video_info: VideoInfo, effect_cues: list[EffectCue],
              *, cache_dir: str | None = None, video_path: str | None = None) -> None:
        self._cues = effect_cues
        self._video_info = video_info

    def apply_frame(self, frame: np.ndarray, timestamp: float, context: EffectContext) -> np.ndarray:
        active_cues = self.get_active_cues(timestamp)
        if not active_cues:
            return frame

        cue = active_cues[0]
        params = cue.whip_params
        if params is None:
            return frame

        duration = cue.end_time - cue.start_time
        progress = (timestamp - cue.start_time) / max(duration, 0.001)
        progress = max(0.0, min(1.0, progress))

        # Bell curve intensity: 0 -> peak -> 0
        intensity = math.sin(math.pi * progress) * params.intensity

        h, w = frame.shape[:2]
        # Kernel size proportional to intensity and frame dimension
        k = int(intensity * max(w, h) * 0.15)
        if k < 2:
            return frame

        # Create directional motion blur kernel
        kernel = np.zeros((k, k), dtype=np.float32)
        if params.direction in ("left", "right"):
            kernel[k // 2, :] = 1.0 / k
        else:  # up/down
            kernel[:, k // 2] = 1.0 / k

        blurred = cv2.filter2D(frame, -1, kernel)

        # Blend original and blurred based on intensity
        alpha = min(1.0, intensity)
        return cv2.addWeighted(frame, 1.0 - alpha, blurred, alpha, 0)
