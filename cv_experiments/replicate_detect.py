"""
Replicate API-based face detection — standalone PoC module.

Offloads face detection to Replicate's chigozienri/mediapipe-face model
instead of running MediaPipe locally. Same return signature as
zoom_bounce.get_face_data_seek(): (data, fps, (w, h))

Usage:
    from replicate_detect import get_face_data_replicate
    data, fps, (w, h) = get_face_data_replicate(video_path, active_ranges, n_frames)
"""

import os
import time
import json
import subprocess
import tempfile
import zipfile
import asyncio
from pathlib import Path

import cv2
import numpy as np


REPLICATE_MODEL = "chigozienri/mediapipe-face"
# Will be resolved on first call; set to None to auto-detect latest version
REPLICATE_VERSION = None

DOWNSCALE_HEIGHT = 720  # Downscale to 720p for detection (saves bandwidth)


def _probe_video(video_path):
    """Probe video metadata via ffprobe. Returns (fps, w, h, n_frames)."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
         "-of", "csv=p=0", video_path],
        capture_output=True, text=True,
    )
    parts = probe.stdout.strip().split(",")
    w, h = int(parts[0]), int(parts[1])
    fps_str = parts[2]
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / float(den)
    else:
        fps = float(fps_str)
    # nb_frames may be "N/A" for some containers
    n_frames_str = parts[3] if len(parts) > 3 else "N/A"
    n_frames = int(n_frames_str) if n_frames_str != "N/A" else None
    return fps, w, h, n_frames


def _extract_detection_frames(video_path, active_ranges, fps, w, h, stride, tmpdir):
    """
    Extract detection frames as 720p JPEGs to tmpdir.
    Returns list of (frame_index, jpeg_path) pairs.
    """
    t0 = time.time()

    # Compute downscale dimensions
    if h > DOWNSCALE_HEIGHT:
        scale = DOWNSCALE_HEIGHT / h
        ds_w = int(w * scale) // 2 * 2  # ensure even
        ds_h = DOWNSCALE_HEIGHT
    else:
        ds_w, ds_h = w, h

    frame_size = ds_w * ds_h * 3
    key_frames = []  # (frame_index, jpeg_path)

    for rng_start, rng_end in active_ranges:
        n_frames_range = rng_end - rng_start + 1
        t_start = rng_start / fps

        cmd = [
            "ffmpeg", "-ss", str(t_start), "-i", video_path,
            "-frames:v", str(n_frames_range),
            "-vf", f"scale={ds_w}:{ds_h}",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "pipe:1",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        for idx in range(rng_start, rng_end + 1):
            data = proc.stdout.read(frame_size)
            if len(data) != frame_size:
                break
            run_detect = (idx - rng_start) % stride == 0 or idx == rng_end
            if run_detect:
                rgb = np.frombuffer(data, dtype=np.uint8).reshape(ds_h, ds_w, 3)
                jpeg_path = os.path.join(tmpdir, f"frame_{idx:06d}.jpg")
                cv2.imwrite(jpeg_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 85])
                key_frames.append((idx, jpeg_path))

        proc.stdout.close()
        proc.wait()

    elapsed = time.time() - t0
    print(f"[replicate] Frame extraction: {elapsed:.1f}s ({len(key_frames)} frames, {ds_h}p JPEG)")
    return key_frames, ds_w, ds_h


def _create_zip(key_frames, tmpdir):
    """Zip all extracted JPEGs (ZIP_STORED — already compressed)."""
    t0 = time.time()
    zip_path = os.path.join(tmpdir, "frames.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for _, jpeg_path in key_frames:
            zf.write(jpeg_path, os.path.basename(jpeg_path))
    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    elapsed = time.time() - t0
    print(f"[replicate] Zip creation: {elapsed:.1f}s ({size_mb:.1f} MB)")
    return zip_path


def _call_replicate_batch(zip_path):
    """Call Replicate API with zip archive. Returns parsed output."""
    import replicate

    t0 = time.time()
    output = replicate.run(
        REPLICATE_MODEL,
        input={"images": open(zip_path, "rb")},
    )
    elapsed = time.time() - t0
    print(f"[replicate] API call: {elapsed:.1f}s (batch zip)")
    return output


async def _call_replicate_parallel(key_frames, max_concurrent=10):
    """
    Fallback: parallel async predictions for single-image API.
    Rate limit: 600 predictions/min (10/sec burst).
    """
    import replicate

    t0 = time.time()
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _detect_one(jpeg_path):
        async with semaphore:
            output = await replicate.async_run(
                REPLICATE_MODEL,
                input={"image": open(jpeg_path, "rb")},
            )
            return output

    tasks = [_detect_one(jpeg_path) for _, jpeg_path in key_frames]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.time() - t0
    n_ok = sum(1 for r in results if not isinstance(r, Exception))
    n_err = sum(1 for r in results if isinstance(r, Exception))
    print(f"[replicate] Parallel API: {elapsed:.1f}s ({n_ok} ok, {n_err} errors, {len(key_frames)} total)")
    return results


def _parse_output_entry(entry, ds_w, ds_h, orig_w, orig_h):
    """
    Parse a single Replicate output entry into (cx, cy, fw, fh) in original resolution.

    Handles two possible formats:
    - Landmarks dict with nose_tip, ear_tragion_left/right, forehead, chin
    - Mask image (binary) → boundingRect
    """
    scale_x = orig_w / ds_w
    scale_y = orig_h / ds_h

    # Format 1: landmarks / keypoints dict
    if isinstance(entry, dict):
        # Try landmark-based parsing
        if "nose_tip" in entry:
            cx = entry["nose_tip"]["x"]
            cy = entry["nose_tip"]["y"]
            if "ear_tragion_left" in entry and "ear_tragion_right" in entry:
                fw = abs(entry["ear_tragion_right"]["x"] - entry["ear_tragion_left"]["x"])
            else:
                fw = 100 / orig_w  # fallback normalized
            if "forehead" in entry and "chin" in entry:
                fh = abs(entry["chin"]["y"] - entry["forehead"]["y"])
            else:
                fh = 100 / orig_h
            return (
                int(cx * scale_x) if cx <= 1.0 else int(cx * scale_x / ds_w * orig_w),
                int(cy * scale_y) if cy <= 1.0 else int(cy * scale_y / ds_h * orig_h),
                int(fw * orig_w) if fw <= 1.0 else int(fw * scale_x),
                int(fh * orig_h) if fh <= 1.0 else int(fh * scale_y),
            )

        # Could be bounding box format
        if "x" in entry and "y" in entry and "width" in entry:
            return (
                int((entry["x"] + entry["width"] / 2) * scale_x),
                int((entry["y"] + entry["height"] / 2) * scale_y),
                int(entry["width"] * scale_x),
                int(entry["height"] * scale_y),
            )

        # Face landmarks array (MediaPipe-style indexed list)
        if "landmarks" in entry or "face_landmarks" in entry:
            landmarks = entry.get("landmarks") or entry.get("face_landmarks")
            if isinstance(landmarks, list) and len(landmarks) > 454:
                # Same indices as local MediaPipe: 4=nose, 234/454=ears, 10/152=forehead/chin
                nose = landmarks[4]
                cx = nose["x"] if isinstance(nose, dict) else nose[0]
                cy = nose["y"] if isinstance(nose, dict) else nose[1]
                ear_l = landmarks[234]
                ear_r = landmarks[454]
                lx = ear_l["x"] if isinstance(ear_l, dict) else ear_l[0]
                rx = ear_r["x"] if isinstance(ear_r, dict) else ear_r[0]
                top = landmarks[10]
                bot = landmarks[152]
                ty = top["y"] if isinstance(top, dict) else top[1]
                by = bot["y"] if isinstance(bot, dict) else bot[1]
                return (
                    int(cx * orig_w) if cx <= 1.0 else int(cx * scale_x),
                    int(cy * orig_h) if cy <= 1.0 else int(cy * scale_y),
                    int(abs(rx - lx) * orig_w) if rx <= 1.0 else int(abs(rx - lx) * scale_x),
                    int(abs(by - ty) * orig_h) if by <= 1.0 else int(abs(by - ty) * scale_y),
                )

    # Format 2: mask image URL or base64
    if isinstance(entry, str):
        # Could be a URL to a mask image or base64 data
        if entry.startswith("http"):
            import urllib.request
            resp = urllib.request.urlopen(entry)
            mask_data = np.frombuffer(resp.read(), dtype=np.uint8)
            mask = cv2.imdecode(mask_data, cv2.IMREAD_GRAYSCALE)
        elif entry.startswith("data:"):
            import base64
            b64 = entry.split(",", 1)[1]
            mask_data = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
            mask = cv2.imdecode(mask_data, cv2.IMREAD_GRAYSCALE)
        else:
            # Try reading as file path
            mask = cv2.imread(entry, cv2.IMREAD_GRAYSCALE)

        if mask is not None:
            _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            coords = cv2.findNonZero(thresh)
            if coords is not None:
                x, y, bw, bh = cv2.boundingRect(coords)
                return (
                    int((x + bw / 2) * scale_x),
                    int((y + bh / 2) * scale_y),
                    int(bw * scale_x),
                    int(bh * scale_y),
                )

    # Format 3: list of detections (take first face)
    if isinstance(entry, list) and len(entry) > 0:
        return _parse_output_entry(entry[0], ds_w, ds_h, orig_w, orig_h)

    return None


def _parse_replicate_output(output, key_frames, ds_w, ds_h, orig_w, orig_h):
    """
    Parse Replicate output into {frame_index: (cx, cy, fw, fh)} dict.
    Output could be a list (one per image) or a dict keyed by filename.
    """
    t0 = time.time()
    detections = {}
    default = (orig_w // 2, orig_h // 2, 100, 100)

    if isinstance(output, list):
        # Assume same order as sorted filenames (frame_{idx:06d}.jpg)
        sorted_frames = sorted(key_frames, key=lambda x: x[1])  # sort by filename
        for i, entry in enumerate(output):
            if i >= len(sorted_frames):
                break
            frame_idx = sorted_frames[i][0]
            parsed = _parse_output_entry(entry, ds_w, ds_h, orig_w, orig_h)
            detections[frame_idx] = parsed if parsed else default

    elif isinstance(output, dict):
        # Keyed by filename
        idx_map = {os.path.basename(p): idx for idx, p in key_frames}
        for filename, entry in output.items():
            frame_idx = idx_map.get(filename)
            if frame_idx is not None:
                parsed = _parse_output_entry(entry, ds_w, ds_h, orig_w, orig_h)
                detections[frame_idx] = parsed if parsed else default

    else:
        # Single output — try parsing directly
        print(f"[replicate] Unexpected output type: {type(output)}")
        print(f"[replicate] Output preview: {str(output)[:500]}")

    elapsed = time.time() - t0
    print(f"[replicate] Output parsing: {elapsed:.1f}s ({len(detections)} detections)")
    return detections


def _interpolate_and_fill(detections, active_ranges, n_frames, w, h, stride):
    """
    Interpolate skipped frames within ranges and fill gaps between ranges.
    Same logic as zoom_bounce._detect_range_worker + _apply_worker_results.
    """
    t0 = time.time()
    default = (w // 2, h // 2, 100, 100)
    data = [default] * n_frames
    last_detected = default

    for rng_start, rng_end in active_ranges:
        # Collect key detections within this range
        key_indices = []
        key_vals = []
        for idx in range(rng_start, rng_end + 1):
            if idx in detections:
                key_indices.append(idx)
                key_vals.append(detections[idx])
                last_detected = detections[idx]

        # Interpolate within range
        if len(key_indices) >= 2:
            ki = np.array(key_indices, dtype=np.float64)
            kv = np.array(key_vals, dtype=np.float64)
            all_idx = np.arange(rng_start, rng_end + 1, dtype=np.float64)
            interped = np.column_stack([
                np.interp(all_idx, ki, kv[:, c]) for c in range(4)
            ]).astype(int)
            for j, idx in enumerate(range(rng_start, rng_end + 1)):
                data[idx] = tuple(interped[j])
        elif len(key_indices) == 1:
            val = key_vals[0]
            for idx in range(rng_start, rng_end + 1):
                data[idx] = val

        # Fill gap after this range until next range
        next_start = n_frames
        for ns, _ in active_ranges:
            if ns > rng_end:
                next_start = ns
                break
        for fill_idx in range(rng_end + 1, next_start):
            data[fill_idx] = last_detected

    elapsed = time.time() - t0
    print(f"[replicate] Interpolation + fill: {elapsed:.1f}s")
    return data


def get_face_data_replicate(video_path, active_ranges, n_frames, stride=3):
    """
    Replicate API-based face detection. Drop-in replacement for get_face_data_seek().

    Returns: (data, fps, (w, h))
        data: list of (cx, cy, fw, fh) tuples, one per frame
        fps: float
        (w, h): original video dimensions
    """
    total_t0 = time.time()

    # 1. Probe video
    fps, w, h, _ = _probe_video(video_path)

    with tempfile.TemporaryDirectory(prefix="replicate_detect_") as tmpdir:
        # 2. Extract detection frames as JPEGs
        key_frames, ds_w, ds_h = _extract_detection_frames(
            video_path, active_ranges, fps, w, h, stride, tmpdir
        )

        if not key_frames:
            print("[replicate] No frames extracted — returning defaults")
            default = (w // 2, h // 2, 100, 100)
            return [default] * n_frames, fps, (w, h)

        # 3. Create zip archive
        zip_path = _create_zip(key_frames, tmpdir)

        # 4. Call Replicate API (try batch first, fall back to parallel)
        try:
            output = _call_replicate_batch(zip_path)
        except Exception as e:
            print(f"[replicate] Batch call failed ({e}), falling back to parallel async")
            results = asyncio.run(_call_replicate_parallel(key_frames))
            output = results

        # 5. Parse output
        detections = _parse_replicate_output(output, key_frames, ds_w, ds_h, w, h)

    # 6. Interpolate + fill gaps
    data = _interpolate_and_fill(detections, active_ranges, n_frames, w, h, stride)

    total_elapsed = time.time() - total_t0
    n_infer = len([kf for kf in key_frames])
    print(f"[replicate] Total detection: {total_elapsed:.1f}s for {n_infer} inference frames")

    return data, fps, (w, h)
