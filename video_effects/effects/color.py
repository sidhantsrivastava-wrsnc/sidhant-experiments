import numpy as np
import cv2

from video_effects.effects.base import BaseEffect, EffectContext
from video_effects.schemas.effects import EffectCue, VideoInfo


# Pre-defined color grading LUTs (RGB adjustments)
COLOR_PRESETS = {
    "warm": (10, -5, -15),     # R=+10, G=-5, B=-15 (boost red, reduce blue)
    "cool": (-15, 0, 15),      # R=-15, G=0,  B=+15 (boost blue, reduce red)
    "dramatic": (-10, -20, 0), # deepen shadows
    "sepia": None,              # special handling
    "bw": None,                 # special handling
}


class ColorEffect(BaseEffect):
    """Color grading effect with presets and custom adjustments."""

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
            params = cue.color_params
            if params is None:
                continue

            intensity = params.intensity

            if params.preset == "bw":
                gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
                bw = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                result = cv2.addWeighted(result, 1.0 - intensity, bw, intensity, 0)
            elif params.preset == "sepia":
                result = self._apply_sepia(result, intensity)
            elif params.preset == "custom":
                adjustments = np.array(
                    [params.r_adjust, params.g_adjust, params.b_adjust],
                    dtype=np.float32,
                ) * intensity
                result = np.clip(
                    result.astype(np.float32) + adjustments, 0, 255
                ).astype(np.uint8)
            else:
                # Preset lookup
                adj = COLOR_PRESETS.get(params.preset)
                if adj:
                    adjustments = np.array(adj, dtype=np.float32) * intensity
                    result = np.clip(
                        result.astype(np.float32) + adjustments, 0, 255
                    ).astype(np.uint8)

        return result

    def _apply_sepia(self, frame: np.ndarray, intensity: float) -> np.ndarray:
        """Apply sepia tone using standard sepia matrix."""
        sepia_kernel = np.array([
            [0.393, 0.769, 0.189],  # out_R = 0.393*R + 0.769*G + 0.189*B
            [0.349, 0.686, 0.168],  # out_G = 0.349*R + 0.686*G + 0.168*B
            [0.272, 0.534, 0.131],  # out_B = 0.272*R + 0.534*G + 0.131*B
        ], dtype=np.float32)
        sepia = cv2.transform(frame, sepia_kernel)
        sepia = np.clip(sepia, 0, 255).astype(np.uint8)
        return cv2.addWeighted(frame, 1.0 - intensity, sepia, intensity, 0)
