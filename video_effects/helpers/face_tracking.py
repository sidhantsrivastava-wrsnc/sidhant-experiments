"""Standalone face detection for the video effects pipeline.

Single-process, ffmpeg-pipe-based face detection using MediaPipe FaceLandmarker.
No dependency on cv_experiments/.
"""

import logging
import os
import subprocess

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from video_effects.config import settings

logger = logging.getLogger(__name__)

_DEFAULT_FACE = (0, 0, 100, 100)  # placeholder, overwritten per-video


def _probe_video(video_path: str) -> tuple[int, int, float]:
    """Get width, height, fps via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate",
            "-of", "csv=p=0", video_path,
        ],
        capture_output=True, text=True,
    )
    parts = result.stdout.strip().split(",")
    w, h = int(parts[0]), int(parts[1])
    fps_str = parts[2]
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / float(den)
    else:
        fps = float(fps_str)
    return w, h, fps


def detect_faces(
    video_path: str,
    active_ranges: list[tuple[int, int]],
    total_frames: int,
    stride: int | None = None,
) -> list[tuple[int, int, int, int]]:
    """Run face detection on active frame ranges.

    Decodes frames via ffmpeg pipe, runs MediaPipe FaceLandmarker at stride
    intervals, interpolates in between. Single-process to avoid fork issues
    with MediaPipe on macOS.

    Args:
        video_path: Path to video file.
        active_ranges: List of (start_frame, end_frame) ranges to detect in.
        total_frames: Total number of frames in the video.
        stride: Detect every N frames (default: from settings).

    Returns:
        List of (center_x, center_y, face_width, face_height) per frame.
        Length == total_frames.
    """
    if stride is None:
        stride = settings.FACE_DETECTION_STRIDE

    w, h, fps = _probe_video(video_path)
    default = (w // 2, h // 2, 100, 100)
    data: list[tuple[int, int, int, int]] = [default] * total_frames
    frame_size = w * h * 3

    model_path = settings.FACE_LANDMARKER_PATH
    if not os.path.isabs(model_path):
        # Resolve relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Go up one more to get to sidhant-experiments
        project_root = os.path.dirname(project_root)
        model_path = os.path.join(project_root, model_path)

    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.FaceLandmarker.create_from_options(opts)
    last_detected = default
    total_infer = 0
    last_ts_ms = -1

    for rng_start, rng_end in active_ranges:
        n_frames = rng_end - rng_start + 1
        t_start = rng_start / fps

        cmd = [
            "ffmpeg", "-ss", str(t_start), "-i", video_path,
            "-frames:v", str(n_frames),
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "pipe:1",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        key_indices: list[int] = []
        key_vals: list[tuple[int, int, int, int]] = []

        for idx in range(rng_start, rng_end + 1):
            raw = proc.stdout.read(frame_size)
            if len(raw) != frame_size:
                break

            run_detect = (idx - rng_start) % stride == 0 or idx == rng_end
            if run_detect:
                rgb = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
                ts_ms = max(int(idx * 1000 / fps), last_ts_ms + 1)
                last_ts_ms = ts_ms
                res = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                    ts_ms,
                )
                if res.face_landmarks:
                    f = res.face_landmarks[0]
                    detected = (
                        int(f[4].x * w),      # nose tip x
                        int(f[4].y * h),      # nose tip y
                        int(abs(f[454].x - f[234].x) * w),  # face width
                        int(abs(f[152].y - f[10].y) * h),   # face height
                    )
                    last_detected = detected

                key_indices.append(idx)
                key_vals.append(last_detected)
                total_infer += 1

                if total_infer % 50 == 0:
                    logger.info("Face detection: %d inferences done", total_infer)

        proc.stdout.close()
        proc.wait()

        # Interpolate skipped frames within this range
        if len(key_indices) >= 2:
            ki = np.array(key_indices, dtype=np.float64)
            kv = np.array(key_vals, dtype=np.float64)
            all_idx = np.arange(rng_start, rng_end + 1, dtype=np.float64)
            interped = np.column_stack([
                np.interp(all_idx, ki, kv[:, c]) for c in range(4)
            ]).astype(int)
            for j, frame_idx in enumerate(range(rng_start, rng_end + 1)):
                data[frame_idx] = tuple(interped[j])
        elif len(key_indices) == 1:
            data[key_indices[0]] = key_vals[0]

    landmarker.close()

    # Fill gaps between ranges with last detected from preceding range
    last_fill = default
    for ri, (rng_start, rng_end) in enumerate(active_ranges):
        # Fill gap before this range
        prev_end = active_ranges[ri - 1][1] + 1 if ri > 0 else 0
        for fill_idx in range(prev_end, rng_start):
            data[fill_idx] = last_fill
        # Update last_fill from this range's last frame
        last_fill = data[rng_end]
    # Fill after last range
    if active_ranges:
        last_end = active_ranges[-1][1]
        for fill_idx in range(last_end + 1, total_frames):
            data[fill_idx] = data[last_end]

    logger.info(
        "Face detection complete: %d inferences, %d ranges, %d total frames",
        total_infer, len(active_ranges), total_frames,
    )
    return data


def smooth_data(data: list, alpha: float | None = None) -> np.ndarray:
    """Exponential moving average filter for tracking data.

    Args:
        data: List of (cx, cy, fw, fh) tuples.
        alpha: Smoothing factor (0-1). Lower = smoother. Default from settings.

    Returns:
        Smoothed numpy array of same shape (int32).
    """
    if alpha is None:
        alpha = settings.SMOOTHING_ALPHA

    a = np.array(data, dtype=np.float64)
    o = np.empty_like(a)
    o[0] = a[0]
    inv = 1.0 - alpha
    for i in range(1, len(a)):
        o[i] = alpha * a[i] + inv * o[i - 1]
    return o.astype(np.int32)
