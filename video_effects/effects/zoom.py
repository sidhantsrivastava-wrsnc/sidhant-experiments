import json
import math
import os

import numpy as np
import cv2

from video_effects.effects.base import BaseEffect, EffectContext
from video_effects.schemas.effects import EffectCue, VideoInfo


def export_zoom_state(
    effects: list[dict],
    face_data: list | None,
    total_frames: int,
    fps: float,
    width: int,
    height: int,
    output_path: str,
) -> str:
    """Pre-compute per-frame [zoomLevel, targetNormX, targetNormY] and write to JSON.

    Output format: {"frames": {<frame_idx>: [zoom, tx_norm, ty_norm], ...}}
    Only includes frames where zoom != 1.0 (sparse).
    """
    zoom_cues = [e for e in effects if e.get("effect_type") == "zoom"]
    if not zoom_cues:
        return ""

    # Build hold intervals (same logic as ZoomEffect._build_hold_intervals)
    sorted_cues = sorted(zoom_cues, key=lambda c: c.get("start_time", 0))
    hold_intervals: list[tuple[float, float, float, str]] = []
    pending_in = None
    for cue in sorted_cues:
        params = cue.get("zoom_params", {})
        action = params.get("action", "bounce")
        if action == "in":
            pending_in = cue
        elif action == "out" and pending_in is not None:
            p_end = pending_in.get("end_time", 0)
            c_start = cue.get("start_time", 0)
            if p_end < c_start:
                hold_intervals.append((
                    p_end, c_start,
                    pending_in.get("zoom_params", {}).get("zoom_level", 1.5),
                    pending_in.get("zoom_params", {}).get("tracking", "center"),
                ))
            pending_in = None

    frames_dict: dict[str, list[float]] = {}

    # Dampened face tracking state
    damp_fx, damp_fy = float(width) / 2, float(height) / 2
    damp_initialized = False

    for frame_idx in range(total_frames):
        timestamp = frame_idx / fps
        current_zoom = 1.0
        tx, ty = float(width) / 2, float(height) / 2

        # Check hold intervals
        in_hold = False
        for hold_start, hold_end, hold_zoom, hold_tracking in hold_intervals:
            if hold_start <= timestamp <= hold_end:
                current_zoom = hold_zoom
                if hold_tracking == "face" and face_data and frame_idx < len(face_data):
                    fd = face_data[frame_idx]
                    fx, fy = float(fd[0]), float(fd[1])
                    dampen = 1.0 / max(hold_zoom, 1.001)
                    if not damp_initialized:
                        damp_fx, damp_fy = fx, fy
                        damp_initialized = True
                    else:
                        damp_fx = dampen * fx + (1.0 - dampen) * damp_fx
                        damp_fy = dampen * fy + (1.0 - dampen) * damp_fy
                    tx, ty = damp_fx, damp_fy
                in_hold = True
                break

        if not in_hold:
            # Check active zoom cues
            for cue in zoom_cues:
                cue_start = cue.get("start_time", 0)
                cue_end = cue.get("end_time", 0)
                if not (cue_start <= timestamp <= cue_end):
                    continue

                params = cue.get("zoom_params", {})
                z = params.get("zoom_level", 1.5)
                duration = cue_end - cue_start
                progress = (timestamp - cue_start) / duration if duration > 0 else 1.0

                action = params.get("action", "bounce")
                easing = params.get("easing", "smooth")
                if action == "in":
                    intensity = _ease_in_standalone(progress, easing)
                elif action == "out":
                    intensity = _ease_out_standalone(progress, easing)
                else:
                    intensity = _ease_bounce_standalone(progress, easing)

                current_zoom = 1.0 + (z - 1.0) * intensity

                tracking = params.get("tracking", "center")
                if tracking == "face" and face_data and frame_idx < len(face_data):
                    fd = face_data[frame_idx]
                    fx, fy = float(fd[0]), float(fd[1])
                    dampen = 1.0 / max(current_zoom, 1.001)
                    if not damp_initialized:
                        damp_fx, damp_fy = fx, fy
                        damp_initialized = True
                    else:
                        damp_fx = dampen * fx + (1.0 - dampen) * damp_fx
                        damp_fy = dampen * fy + (1.0 - dampen) * damp_fy
                    # Lerp between center and face position
                    tx = width / 2 + (damp_fx - width / 2) * intensity
                    ty = height / 2 + (damp_fy - height / 2) * intensity
                break  # Only apply first matching cue

        if current_zoom > 1.001:
            tx_norm = tx / width
            ty_norm = ty / height
            frames_dict[str(frame_idx)] = [
                round(current_zoom, 4),
                round(tx_norm, 4),
                round(ty_norm, 4),
            ]

    if not frames_dict:
        return ""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"frames": frames_dict}, f)

    return output_path


def _ease_bounce_standalone(t: float, easing: str) -> float:
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
    else:
        return math.sin(math.pi * t)


def _ease_in_standalone(t: float, easing: str) -> float:
    t = max(0.0, min(1.0, t))
    if easing == "snap":
        if t < 0.5:
            return (t / 0.5) ** 2
        return 1.0
    elif easing == "overshoot":
        base = math.sin(math.pi / 2 * t)
        overshoot = 0.15 * math.sin(math.pi * t)
        return max(0.0, min(1.15, base + overshoot))
    else:
        return math.sin(math.pi / 2 * t)


def _ease_out_standalone(t: float, easing: str) -> float:
    t = max(0.0, min(1.0, t))
    if easing == "snap":
        return max(0.0, 1.0 - t ** 2)
    elif easing == "overshoot":
        base = math.cos(math.pi / 2 * t)
        undershoot = -0.15 * math.sin(math.pi * t)
        return max(0.0, min(1.15, base + undershoot))
    else:
        return math.cos(math.pi / 2 * t)


class ZoomEffect(BaseEffect):
    """Zoom effect with face-tracked, center, or point tracking."""

    def __init__(self):
        super().__init__()
        self._face_data: list[tuple[float, float, float, float]] | None = None
        self._video_info: VideoInfo | None = None
        self._hold_intervals: list[tuple[float, float, float, str]] = []

    def setup(self, video_info: VideoInfo, effect_cues: list[EffectCue],
              *, cache_dir: str | None = None, video_path: str | None = None) -> None:
        self._cues = effect_cues
        self._video_info = video_info

        # Pre-compute face tracking for face-tracked zoom cues
        face_cues = [
            c for c in effect_cues
            if c.zoom_params and c.zoom_params.tracking == "face"
        ]
        if face_cues:
            self._setup_face_tracking(video_info, face_cues,
                                      cache_dir=cache_dir, video_path=video_path)

        self._build_hold_intervals()

    def set_cues(self, effect_cues: list[EffectCue]) -> None:
        super().set_cues(effect_cues)
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
        self, video_info: VideoInfo, face_cues: list[EffectCue],
        *, cache_dir: str | None = None, video_path: str | None = None,
    ) -> None:
        """Run face detection on active ranges, with optional disk cache."""
        cache_file = os.path.join(cache_dir, "face_tracking_zoom.json") if cache_dir else None

        from video_effects.helpers.face_tracking import (
            detect_faces, smooth_data, _probe_decoded_size,
        )

        # Get decoded dimensions for cache validation
        if video_path:
            dec_w, dec_h = _probe_decoded_size(video_path)
        else:
            dec_w, dec_h = video_info.width, video_info.height

        # Try loading from cache — validate dimensions match
        if cache_file and os.path.exists(cache_file):
            with open(cache_file) as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                cached_dims = raw.get("dimensions")
                if cached_dims == [dec_w, dec_h]:
                    self._face_data = [tuple(row) for row in raw["face_data"]]
                    return
                # Dimension mismatch — stale cache, regenerate
            # Old list format — regenerate with correct dimensions

        if video_path is None:
            self._face_data = None
            return

        active_ranges = [
            (
                int(c.start_time * video_info.fps),
                int(c.end_time * video_info.fps),
            )
            for c in face_cues
        ]

        raw_data = detect_faces(
            video_path, active_ranges, video_info.total_frames
        )
        smoothed = smooth_data(raw_data)
        self._face_data = [tuple(row) for row in smoothed.tolist()]

        # Write cache with dimensions metadata
        if cache_file:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file, "w") as f:
                json.dump({
                    "dimensions": [dec_w, dec_h],
                    "face_data": self._face_data,
                }, f)

    def apply_frame(
        self, frame: np.ndarray, timestamp: float, context: EffectContext
    ) -> np.ndarray:
        # Check hold intervals first (constant zoom between in/out pairs)
        for hold_start, hold_end, hold_zoom, hold_tracking in self._hold_intervals:
            if hold_start <= timestamp <= hold_end:
                h, w = frame.shape[:2]
                if hold_tracking == "face" and self._face_data is not None:
                    fx, fy, _, _ = self._face_data[context.frame_index]
                    dampen = 1.0 / max(hold_zoom, 1.001)
                    if not hasattr(self, '_damp_fx'):
                        self._damp_fx, self._damp_fy = float(fx), float(fy)
                    else:
                        self._damp_fx = dampen * fx + (1.0 - dampen) * self._damp_fx
                        self._damp_fy = dampen * fy + (1.0 - dampen) * self._damp_fy
                    tx, ty = self._damp_fx, self._damp_fy
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
                # Dampen face tracking more at higher zoom levels
                # (prevents amplified jitter — matches reference implementation)
                dampen = 1.0 / max(current_zoom, 1.001)
                if not hasattr(self, '_damp_fx'):
                    self._damp_fx, self._damp_fy = float(fx), float(fy)
                else:
                    self._damp_fx = dampen * fx + (1.0 - dampen) * self._damp_fx
                    self._damp_fy = dampen * fy + (1.0 - dampen) * self._damp_fy
                tx = self._lerp(w / 2, self._damp_fx, intensity)
                ty = self._lerp(h / 2, self._damp_fy, intensity)
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

            # Radial motion blur (zoom blur effect)
            if params.motion_blur > 0 and intensity > 0.05:
                blur_strength = params.motion_blur * intensity
                extra_zoom = 1.0 + blur_strength * 0.08
                # Apply extra zoom relative to output frame center (result is already centered)
                M_blur = np.float32([
                    [extra_zoom, 0, (w / 2) * (1 - extra_zoom)],
                    [0, extra_zoom, (h / 2) * (1 - extra_zoom)],
                ])
                blurred = cv2.warpAffine(result, M_blur, (w, h), borderMode=cv2.BORDER_REPLICATE)
                alpha = min(0.7, blur_strength * 0.5)
                result = cv2.addWeighted(result, 1.0 - alpha, blurred, alpha, 0)

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
