"""Activity: detect hard cuts (jump cuts) in video via frame histogram comparison."""

import cv2
from temporalio import activity

from video_effects.schemas.effects import VideoInfo


@activity.defn(name="vfx_detect_jump_cuts")
def detect_jump_cuts(input_data: dict) -> dict:
    """Detect hard cuts in video by comparing adjacent frame histograms.

    Input: {"video_path": str, "video_info": dict, "threshold": float}
    Output: {"jump_cuts": [{"time": float, "confidence": float}]}
    """
    video_path = input_data["video_path"]
    video_info = VideoInfo(**input_data["video_info"])
    threshold = input_data.get("threshold", 0.35)

    cap = cv2.VideoCapture(video_path)
    prev_hist = None
    jump_cuts = []

    for frame_idx in range(video_info.total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Downsample for speed
        small = cv2.resize(frame, (160, 90))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist)

        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if diff > threshold:
                time = frame_idx / video_info.fps
                jump_cuts.append({
                    "time": time,
                    "confidence": min(1.0, diff / threshold),
                })

        prev_hist = hist

        if frame_idx % 500 == 0:
            activity.heartbeat(f"frame {frame_idx}/{video_info.total_frames}")

    cap.release()

    filtered = _filter_jump_cuts(jump_cuts, video_info.duration)
    return {"jump_cuts": filtered}


def _filter_jump_cuts(cuts: list[dict], duration: float) -> list[dict]:
    """Remove edge cuts and merge nearby ones."""
    result = []
    for cut in cuts:
        if cut["time"] < 1.0 or cut["time"] > duration - 1.0:
            continue
        if result and cut["time"] - result[-1]["time"] < 0.5:
            continue  # too close to previous
        result.append(cut)
    return result
