from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from video_effects.schemas.effects import EffectCue, VideoInfo


@dataclass
class EffectContext:
    """Runtime context passed to each effect during frame processing."""
    video_info: VideoInfo
    frame_index: int
    timestamp: float
    total_frames: int


class BaseEffect(ABC):
    """All effects implement this interface."""

    def __init__(self):
        self._cues: list[EffectCue] = []

    @abstractmethod
    def setup(self, video_info: VideoInfo, effect_cues: list[EffectCue]) -> None:
        """Pre-compute any data needed (face tracking, masks, etc.).

        Called once per phase before frame iteration begins.
        """
        ...

    @abstractmethod
    def apply_frame(
        self, frame: np.ndarray, timestamp: float, context: EffectContext
    ) -> np.ndarray:
        """Apply this effect to a single frame. Returns modified frame."""
        ...

    def is_active(self, timestamp: float) -> bool:
        """Check if any cue is active at this timestamp."""
        return any(
            cue.start_time <= timestamp <= cue.end_time for cue in self._cues
        )

    def get_active_cues(self, timestamp: float) -> list[EffectCue]:
        """Get all cues active at this timestamp."""
        return [
            cue for cue in self._cues
            if cue.start_time <= timestamp <= cue.end_time
        ]

    def get_active_ranges(self) -> list[tuple[float, float]]:
        """Return time ranges where this effect is active."""
        return [(cue.start_time, cue.end_time) for cue in self._cues]

    def _lerp(self, a: float, b: float, t: float) -> float:
        return a + (b - a) * t
