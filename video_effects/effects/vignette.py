import numpy as np
import cv2

from video_effects.effects.base import BaseEffect, EffectContext
from video_effects.schemas.effects import EffectCue, VideoInfo


class VignetteEffect(BaseEffect):
    """Darkened edges for cinematic focus."""

    def __init__(self):
        super().__init__()
        self._mask_cache: dict[tuple[int, int], np.ndarray] = {}

    def setup(self, video_info: VideoInfo, effect_cues: list[EffectCue],
              *, cache_dir: str | None = None, video_path: str | None = None) -> None:
        self._cues = effect_cues
        self._video_info = video_info

    def apply_frame(self, frame: np.ndarray, timestamp: float, context: EffectContext) -> np.ndarray:
        active_cues = self.get_active_cues(timestamp)
        if not active_cues:
            return frame

        cue = active_cues[0]
        params = cue.vignette_params
        if params is None:
            return frame

        h, w = frame.shape[:2]
        key = (w, h)

        # Cache the base distance mask (same for all frames at same resolution)
        if key not in self._mask_cache:
            Y, X = np.ogrid[:h, :w]
            cx, cy = w / 2, h / 2
            dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
            self._mask_cache[key] = dist.astype(np.float32)

        dist = self._mask_cache[key]

        # Map distance to darkness: below radius=1.0, above radius=darken
        falloff = np.clip((dist - params.radius) / (1.4 - params.radius), 0.0, 1.0)
        darkness = 1.0 - falloff * params.strength

        # Apply: multiply each channel
        result = (frame.astype(np.float32) * darkness[:, :, np.newaxis]).clip(0, 255).astype(np.uint8)

        # Ease in/out at cue boundaries (first/last 0.5s)
        duration = cue.end_time - cue.start_time
        fade_dur = min(0.5, duration / 4)
        if timestamp < cue.start_time + fade_dur:
            t = (timestamp - cue.start_time) / fade_dur
            return cv2.addWeighted(frame, 1.0 - t, result, t, 0)
        elif timestamp > cue.end_time - fade_dur:
            t = (cue.end_time - timestamp) / fade_dur
            return cv2.addWeighted(frame, 1.0 - t, result, t, 0)
        return result
