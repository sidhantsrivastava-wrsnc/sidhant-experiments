import math

import cv2
import numpy as np

from video_effects.effects.base import BaseEffect, EffectContext
from video_effects.schemas.effects import EffectCue, VideoInfo


class SpeedRampEffect(BaseEffect):
    """Visual speed ramp — creates fast-forward appearance via motion blur.

    v1: visual-only effect (no actual time remapping). Creates a speed-up
    appearance using directional motion blur proportional to the speed setting.
    """

    def setup(self, video_info: VideoInfo, effect_cues: list[EffectCue],
              *, cache_dir: str | None = None, video_path: str | None = None) -> None:
        self._cues = effect_cues
        self._video_info = video_info

    def apply_frame(self, frame: np.ndarray, timestamp: float, context: EffectContext) -> np.ndarray:
        active_cues = self.get_active_cues(timestamp)
        if not active_cues:
            return frame

        cue = active_cues[0]
        params = cue.speed_ramp_params
        if not params:
            return frame

        duration = cue.end_time - cue.start_time
        progress = (timestamp - cue.start_time) / max(duration, 0.001)

        # Eased speed: bell curve for smooth, constant for snap
        if params.easing == "smooth":
            t = math.sin(math.pi * progress)
            speed = 1.0 + (params.speed - 1.0) * t
        else:
            speed = params.speed

        # Visual effect: horizontal motion blur proportional to speed
        blur_amount = (speed - 1.0) / (params.speed - 1.0 + 0.001)
        k = int(blur_amount * 30)
        if k < 2:
            return frame

        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0 / k  # horizontal motion blur
        blurred = cv2.filter2D(frame, -1, kernel)

        alpha = min(0.6, blur_amount * 0.4)
        return cv2.addWeighted(frame, 1.0 - alpha, blurred, alpha, 0)
