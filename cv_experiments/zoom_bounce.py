"""
Zoom-Bounce — Dramatic Punch-In / Punch-Out Face Tracking Effect
=================================================================
Face-tracking zoom that punches IN then back OUT (bell curve),
with multiple bounce windows and selectable easing modes.

Based on opte.py pipeline: MediaPipe face tracking → affine warp →
per-row edge-fade → overlay composite → FFmpeg encode.
"""

import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
cv2.setNumThreads(1)
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from moviepy.editor import TextClip, VideoFileClip

MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")


def lerp(a, b, t):
    return a + (b - a) * t


# ─── Bounce easing functions ────────────────────────────────────────────────
# All take an array of normalized time [0,1] and return intensity [0,1] (bell)


def ease_smooth(t):
    """sin(pi*t) — clean symmetric punch in then out."""
    return np.sin(np.pi * t)


def ease_snap(t):
    """Fast attack, brief hold at peak, fast release — editorial punch."""
    out = np.zeros_like(t, dtype=np.float32)
    attack = t < 0.25
    hold = (t >= 0.25) & (t <= 0.75)
    release = t > 0.75
    out[attack] = (t[attack] / 0.25) ** 2
    out[hold] = 1.0
    r = (t[release] - 0.75) / 0.25
    out[release] = 1.0 - r**2
    return out


def ease_overshoot(t):
    """Elastic — overshoots ~15%, springs back, then releases. Playful energy."""
    base = np.sin(np.pi * t)
    overshoot = 0.15 * np.sin(2.0 * np.pi * t)
    return np.clip(base + overshoot, 0.0, 1.15)


EASE_FUNCTIONS = {
    "smooth": ease_smooth,
    "snap": ease_snap,
    "overshoot": ease_overshoot,
}

# ─── Attack-only easing (0→1) for "in" events ──────────────────────────────


def ease_in_smooth(t):
    return np.sin(np.pi / 2 * t)


def ease_in_snap(t):
    out = np.zeros_like(t, dtype=np.float32)
    fast = t < 0.5
    out[fast] = (t[fast] / 0.5) ** 2
    out[~fast] = 1.0
    return out


def ease_in_overshoot(t):
    base = np.sin(np.pi / 2 * t)
    overshoot = 0.15 * np.sin(np.pi * t)
    return np.clip(base + overshoot, 0.0, 1.15)


EASE_IN_FUNCTIONS = {
    "smooth": ease_in_smooth,
    "snap": ease_in_snap,
    "overshoot": ease_in_overshoot,
}

# ─── Release-only easing (1→0) for "out" events ────────────────────────────


def ease_out_smooth(t):
    return np.cos(np.pi / 2 * t)


def ease_out_snap(t):
    return np.clip(1.0 - t**2, 0.0, 1.0)


def ease_out_overshoot(t):
    base = np.cos(np.pi / 2 * t)
    undershoot = -0.15 * np.sin(np.pi * t)
    return np.clip(base + undershoot, 0.0, 1.15)


EASE_OUT_FUNCTIONS = {
    "smooth": ease_out_smooth,
    "snap": ease_out_snap,
    "overshoot": ease_out_overshoot,
}


def _parse_events(bounces, default_mode, default_zoom):
    """
    Normalize bounce entries (tuples and dicts) into canonical event list.

    Returns list of dicts: {"action", "start", "end", "ease", "zoom"}
    Validates: no double zoom-in, no zoom-out without prior zoom-in.
    """
    events = []
    for b in bounces:
        if isinstance(b, dict):
            ev = {**b}
            ev.setdefault("ease", default_mode)
            ev.setdefault("zoom", default_zoom)
            events.append(ev)
        else:
            # Legacy tuple: (start, end[, mode[, zoom]])
            if len(b) == 2:
                ev = {
                    "action": "bounce",
                    "start": b[0],
                    "end": b[1],
                    "ease": default_mode,
                    "zoom": default_zoom,
                }
            elif len(b) == 3:
                ev = {
                    "action": "bounce",
                    "start": b[0],
                    "end": b[1],
                    "ease": b[2],
                    "zoom": default_zoom,
                }
            else:
                ev = {
                    "action": "bounce",
                    "start": b[0],
                    "end": b[1],
                    "ease": b[2],
                    "zoom": b[3],
                }
            events.append(ev)

    # Validate in/out pairing
    zoomed_in = False
    last_zoom = default_zoom
    for ev in events:
        action = ev["action"]
        if action == "in":
            if zoomed_in:
                raise ValueError(
                    f"Double zoom-in at t={ev['start']}: already zoomed in"
                )
            zoomed_in = True
            last_zoom = ev["zoom"]
        elif action == "out":
            if not zoomed_in:
                raise ValueError(f"Zoom-out at t={ev['start']} without prior zoom-in")
            zoomed_in = False
            # Out inherits zoom from its paired in
            ev["zoom"] = last_zoom
        # "bounce" is self-contained, no state change

    return events


def build_bounce_curves(n_frames, fps, bounces, default_mode, default_zoom):
    """
    Build per-frame p (intensity 0-1) and zoom arrays.

    bounces: list of tuples or dicts (see _parse_events).
    Returns: (times, p_curve, zooms) — all shape (n_frames,) float32
    """
    times = np.arange(n_frames, dtype=np.float32) / fps
    p_curve = np.zeros(n_frames, dtype=np.float32)
    zooms = np.ones(n_frames, dtype=np.float32)

    events = _parse_events(bounces, default_mode, default_zoom)

    # Process each event
    for ev in events:
        action = ev["action"]
        bs, be = ev["start"], ev["end"]
        mode = ev["ease"]
        zm = ev["zoom"]
        dur = max(be - bs, 1e-9)
        mask = (times >= bs) & (times <= be)

        if action == "bounce":
            ease_fn = EASE_FUNCTIONS[mode]
            t_norm = np.clip((times[mask] - bs) / dur, 0.0, 1.0)
            p_vals = ease_fn(t_norm)
            new_zooms = 1.0 + (zm - 1.0) * p_vals
            better = new_zooms > zooms[mask]
            p_curve[mask] = np.where(better, p_vals, p_curve[mask])
            zooms[mask] = np.where(better, new_zooms, zooms[mask])

        elif action == "in":
            ease_fn = EASE_IN_FUNCTIONS[mode]
            t_norm = np.clip((times[mask] - bs) / dur, 0.0, 1.0)
            p_vals = ease_fn(t_norm)
            p_curve[mask] = np.maximum(p_curve[mask], p_vals)
            zooms[mask] = np.maximum(zooms[mask], 1.0 + (zm - 1.0) * p_vals)

        elif action == "out":
            ease_fn = EASE_OUT_FUNCTIONS[mode]
            t_norm = np.clip((times[mask] - bs) / dur, 0.0, 1.0)
            p_vals = ease_fn(t_norm)
            p_curve[mask] = np.maximum(p_curve[mask], p_vals)
            zooms[mask] = np.maximum(zooms[mask], 1.0 + (zm - 1.0) * p_vals)

    # Fill holds: between each in-end and its paired out-start, hold at p=1.0
    zoomed_in = False
    in_end = 0.0
    in_zoom = default_zoom
    for ev in events:
        if ev["action"] == "in":
            zoomed_in = True
            in_end = ev["end"]
            in_zoom = ev["zoom"]
        elif ev["action"] == "out" and zoomed_in:
            out_start = ev["start"]
            hold_mask = (times > in_end) & (times < out_start)
            p_curve[hold_mask] = 1.0
            zooms[hold_mask] = in_zoom
            zoomed_in = False

    return times, p_curve, zooms


def build_effect_curves(n_frames, fps, bounces, default_mode, default_zoom):
    """
    Build per-frame intensity arrays for zoom_blur and whip effects.

    Returns: (blur_strength, blur_n_samples, whip_strength, whip_direction)
        blur_strength:   float32 (n_frames,) — 0-1 radial blur intensity
        blur_n_samples:  int32   (n_frames,) — samples per frame (0 = inactive)
        whip_strength:   float32 (n_frames,) — 0-1 whip intensity
        whip_direction:  list of str (n_frames,) — "h" or "v" per frame
    """
    times = np.arange(n_frames, dtype=np.float32) / fps
    blur_strength = np.zeros(n_frames, dtype=np.float32)
    blur_n_samples = np.zeros(n_frames, dtype=np.int32)
    whip_strength = np.zeros(n_frames, dtype=np.float32)
    whip_direction = ["h"] * n_frames

    events = _parse_events(bounces, default_mode, default_zoom)

    for ev in events:
        action = ev["action"]
        if action not in ("zoom_blur", "whip"):
            continue

        bs, be = ev["start"], ev["end"]
        dur = max(be - bs, 1e-9)
        intensity = ev.get("intensity", 1.0)
        mask = (times >= bs) & (times <= be)
        t_norm = np.clip((times[mask] - bs) / dur, 0.0, 1.0)
        # Sine bell: peaks at midpoint
        strength = np.sin(np.pi * t_norm).astype(np.float32) * intensity

        if action == "zoom_blur":
            n_samp = ev.get("n_samples", 8)
            blur_strength[mask] = np.maximum(blur_strength[mask], strength)
            blur_n_samples[mask] = np.where(
                strength > blur_strength[mask] - 1e-6, n_samp, blur_n_samples[mask]
            )
        elif action == "whip":
            direction = ev.get("direction", "h")
            whip_strength[mask] = np.maximum(whip_strength[mask], strength)
            indices = np.where(mask)[0]
            for i in indices:
                whip_direction[i] = direction

    return blur_strength, blur_n_samples, whip_strength, whip_direction


def apply_zoom_blur(
    buf_warped, rgb, M, w, h, strength, n_samples, buf_accum, buf_sample
):
    """
    Radial blur via N additional warpAffine calls at micro-zoom offsets.

    Accumulates samples at slightly different zoom levels centered on frame
    center, then blends result with original by `strength`.
    """
    if strength < 0.001 or n_samples < 1:
        return
    cx, cy = w / 2.0, h / 2.0
    base_zoom = M[0, 0]
    spread = 0.05 * strength * base_zoom

    buf_accum[:] = 0.0
    for i in range(n_samples):
        t = (i / max(n_samples - 1, 1)) * 2.0 - 1.0  # -1 to +1
        dz = t * spread
        sz = base_zoom + dz
        # Adjust translation to keep center fixed
        M_sample = M.copy()
        M_sample[0, 0] = sz
        M_sample[1, 1] = sz
        M_sample[0, 2] = M[0, 2] + cx * (base_zoom - sz)
        M_sample[1, 2] = M[1, 2] + cy * (base_zoom - sz)
        cv2.warpAffine(
            rgb, M_sample, (w, h), dst=buf_sample, borderMode=cv2.BORDER_REPLICATE
        )
        buf_accum += buf_sample.astype(np.float32)

    buf_accum /= n_samples
    # Blend: result = lerp(original, blurred, strength)
    orig_f = buf_warped.astype(np.float32)
    blended = orig_f + (buf_accum - orig_f) * strength
    np.clip(blended, 0, 255, out=blended)
    np.copyto(buf_warped, blended.astype(np.uint8))


def apply_whip(buf_warped, rgb, M, w, h, strength, direction):
    """
    Directional motion blur applied directly to the warped frame.
    Simulates a fast whip-pan without shifting the actual frame content.
    """
    if strength < 0.001:
        return
    # Motion blur kernel size scales with strength (3–81px, always odd)
    ksize = max(3, min(81, int(81 * strength) | 1))
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    if direction == "v":
        kernel[:, ksize // 2] = 1.0 / ksize
    else:
        kernel[ksize // 2, :] = 1.0 / ksize
    blurred = cv2.filter2D(buf_warped, -1, kernel)
    cv2.addWeighted(buf_warped, 1.0 - strength, blurred, strength, 0, dst=buf_warped)


# ─── Encoder detection ───────────────────────────────────────────────────────


def _probe_encoder(name):
    try:
        r = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=black:s=64x64:d=0.04",
                "-c:v",
                name,
                "-f",
                "null",
                "-",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return r.returncode == 0
    except Exception:
        return False


_ENCODER_CANDIDATES = {
    "h264": ["h264_nvenc", "h264_videotoolbox", "h264_qsv", "libx264"],
    "hevc": ["hevc_nvenc", "hevc_videotoolbox", "hevc_qsv", "libx265"],
    "av1": ["av1_nvenc", "av1_videotoolbox", "av1_qsv", "libsvtav1", "libaom-av1"],
    "vp9": ["libvpx-vp9"],
}

# Cache: codec_name → best encoder
_encoder_cache = {}


def detect_best_encoder(codec="h264"):
    if codec not in _encoder_cache:
        candidates = _ENCODER_CANDIDATES.get(codec, [f"lib{codec}"])
        # Probe all candidates in parallel — first (by priority) that succeeds wins
        with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
            results = list(pool.map(_probe_encoder, candidates))
        best = None
        for e, ok in zip(candidates, results):
            if ok:
                best = e
                break
        _encoder_cache[codec] = best or (candidates[-1] if candidates else f"lib{codec}")
        print(f"   Encoder ({codec}): {_encoder_cache[codec]}")
    return _encoder_cache[codec]


# ─── Overlay ─────────────────────────────────────────────────────────────────


class Overlay:
    def get_frame(self, t):
        raise NotImplementedError


class TextOverlay(Overlay):
    def __init__(
        self,
        content,
        color="white",
        fontsize=80,
        font="Arial-Bold",
        max_width=None,
        max_height=None,
    ):
        fontsize = self._fit(content, fontsize, color, font, max_width, max_height)
        kw = dict(fontsize=fontsize, color=color, font=font)
        if max_width:
            kw.update(size=(max_width, None), method="caption")
        txt = TextClip(content, **kw)
        self.img = np.ascontiguousarray(txt.get_frame(0), dtype=np.float32)
        if txt.mask:
            m = txt.mask.get_frame(0)
            if m.max() > 1.0:
                m = m / 255.0
            self.mask = np.ascontiguousarray(m[:, :, np.newaxis], dtype=np.float32)
        else:
            self.mask = np.ones((*self.img.shape[:2], 1), dtype=np.float32)

    @staticmethod
    def _fit(content, fs, color, font, mw, mh):
        if not mw and not mh:
            return fs
        for s in range(fs, 19, -4):
            kw = dict(fontsize=s, color=color, font=font)
            if mw:
                kw.update(size=(mw, None), method="caption")
            sh = TextClip(content, **kw).get_frame(0).shape[:2]
            if (not mw or sh[1] <= mw) and (not mh or sh[0] <= mh):
                return s
        return 20

    def get_frame(self, t):
        return self.img, self.mask


class ImageOverlay(Overlay):
    def __init__(self, path):
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw.shape[2] == 4:
            raw = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGBA)
            self.img = np.ascontiguousarray(raw[:, :, :3], dtype=np.float32)
            self.mask = np.ascontiguousarray(raw[:, :, 3:4] / 255.0, dtype=np.float32)
        else:
            self.img = np.ascontiguousarray(
                cv2.cvtColor(raw, cv2.COLOR_BGR2RGB), dtype=np.float32
            )
            self.mask = np.ones((*raw.shape[:2], 1), dtype=np.float32)

    def get_frame(self, t):
        return self.img, self.mask


class ClipOverlay(Overlay):
    def __init__(self, path):
        self.clip = VideoFileClip(path, has_mask=True)

    def get_frame(self, t):
        t = min(t, self.clip.duration - 0.01)
        img = np.ascontiguousarray(self.clip.get_frame(t), dtype=np.float32)
        if self.clip.mask:
            m = self.clip.mask.get_frame(t)
            if m.max() > 1.0:
                m = m / 255.0
            mask = np.ascontiguousarray(m[:, :, np.newaxis], dtype=np.float32)
        else:
            mask = np.ones((*img.shape[:2], 1), dtype=np.float32)
        return img, mask


def create_overlay(cfg):
    t = cfg.get("type", "text")
    if t == "text":
        return TextOverlay(
            cfg.get("content", "Text"),
            cfg.get("color", "white"),
            cfg.get("fontsize", 80),
            cfg.get("font", "Arial-Bold"),
            cfg.get("_avail_w"),
            cfg.get("_avail_h"),
        )
    if t == "image":
        return ImageOverlay(cfg["path"])
    if t == "clip":
        return ClipOverlay(cfg["path"])
    raise ValueError(f"Unknown overlay type: {t}")


# ─── Threaded reader ─────────────────────────────────────────────────────────


class ThreadedVideoReader:
    def __init__(self, path, queue_size=128):
        self.cap = cv2.VideoCapture(path)
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.q = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stopped:
            ok, f = self.cap.read()
            if not ok:
                self.stopped = True
                return
            self.q.put(f)

    def read(self):
        try:
            return (True, self.q.get(timeout=1.0))
        except queue.Empty:
            return (False, None)

    def release(self):
        self.stopped = True
        self.thread.join(timeout=2)
        self.cap.release()


# ─── Selective detection helpers ──────────────────────────────────────────────


def _compute_active_frame_ranges(bounces, fps, n_frames, padding_sec=2.0,
                                  detect_holds=True):
    """
    Return sorted list of (start_frame, end_frame) ranges where face detection
    is needed.  Only bounce/in/out events use face data; zoom_blur and whip
    do not.  Paired in→out events are merged into a single range that covers
    the hold region between them.  Returns None if no face-dependent events
    exist.

    When detect_holds=False, only the transition segments (in/out) are included
    — hold regions between them are excluded.  This is faster but means face
    positions during holds are carried forward from the last transition frame,
    producing a static crop instead of drift tracking.
    """
    # First pass: pair in/out events and collect time ranges
    raw_ranges = []
    pending_in = None
    for b in bounces:
        if isinstance(b, dict):
            action = b.get("action", "bounce")
            if action in ("zoom_blur", "whip"):
                continue
            if action == "in":
                if detect_holds:
                    pending_in = b["start"]
                    continue
                else:
                    # Only detect the transition itself, not the hold after
                    raw_ranges.append((b["start"], b["end"]))
                    continue
            if action == "out" and pending_in is not None:
                # Merge entire in→hold→out span into one range
                raw_ranges.append((pending_in, b["end"]))
                pending_in = None
                continue
            if action == "out" and not detect_holds:
                # detect_holds=False: just detect the out transition
                raw_ranges.append((b["start"], b["end"]))
                continue
            start_sec, end_sec = b["start"], b["end"]
        else:
            start_sec, end_sec = b[0], b[1]
        raw_ranges.append((start_sec, end_sec))
    # Dangling "in" with no "out" — extend to end of video
    if pending_in is not None:
        raw_ranges.append((pending_in, n_frames / fps))

    if not raw_ranges:
        return None

    # Convert to frame indices with padding for EMA warmup
    frame_ranges = []
    for start_sec, end_sec in raw_ranges:
        f_start = max(0, int((start_sec - padding_sec) * fps))
        f_end = min(n_frames - 1, int(end_sec * fps) + 1)
        frame_ranges.append((f_start, f_end))

    # Sort and merge overlapping ranges
    frame_ranges.sort()
    merged = [frame_ranges[0]]
    for s, e in frame_ranges[1:]:
        if s <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    return merged


def _compute_render_ranges(bounces, fps, n_frames):
    """
    Return sorted list of (start_frame, end_frame) ranges where any effect
    (bounce, in, out, zoom_blur, whip) is active.  No padding — exact event
    boundaries only.  Returns None if no events exist.
    """
    raw_ranges = []
    pending_in = None
    for b in bounces:
        if isinstance(b, dict):
            action = b.get("action", "bounce")
            if action == "in":
                pending_in = b["start"]
                continue
            if action == "out" and pending_in is not None:
                raw_ranges.append((pending_in, b["end"]))
                pending_in = None
                continue
            start_sec, end_sec = b["start"], b["end"]
        else:
            start_sec, end_sec = b[0], b[1]
        raw_ranges.append((start_sec, end_sec))
    if pending_in is not None:
        raw_ranges.append((pending_in, n_frames / fps))

    if not raw_ranges:
        return None

    frame_ranges = []
    for start_sec, end_sec in raw_ranges:
        f_start = max(0, int(start_sec * fps))
        f_end = min(n_frames - 1, int(end_sec * fps) + 1)
        frame_ranges.append((f_start, f_end))

    frame_ranges.sort()
    merged = [frame_ranges[0]]
    for s, e in frame_ranges[1:]:
        if s <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    return merged


def _frame_in_ranges(idx, ranges):
    """Check if frame index falls within any sorted range (early exit)."""
    for s, e in ranges:
        if idx < s:
            return False
        if idx <= e:
            return True
    return False


def _detect_range_worker(args):
    """
    Worker function for parallel face detection.  Each worker opens its own
    VideoCapture and FaceLandmarker to process a batch of active ranges.
    Must be a top-level function for pickling by ProcessPoolExecutor.

    Returns: list of (frame_index, (cx, cy, fw, fh)) for every frame in
    the assigned ranges (interpolated at stride intervals).
    """
    video_path, ranges_batch, stride, worker_id = args
    import cv2 as _cv2
    _cv2.setNumThreads(1)
    import mediapipe as _mp
    from mediapipe.tasks.python import vision as _vision, BaseOptions as _BO

    opts = _vision.FaceLandmarkerOptions(
        base_options=_BO(model_asset_path=MODEL_PATH),
        running_mode=_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    lm = _vision.FaceLandmarker.create_from_options(opts)
    cap = _cv2.VideoCapture(video_path)
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(_cv2.CAP_PROP_FPS)
    default = (w // 2, h // 2, 100, 100)
    last_detected = default
    results = []  # (frame_index, value) pairs
    done_infer = 0
    total_infer = sum((e - s) // stride + 1 for s, e in ranges_batch)

    for rng_start, rng_end in ranges_batch:
        cap.set(_cv2.CAP_PROP_POS_FRAMES, rng_start)
        key_indices = []
        key_vals = []
        for idx in range(rng_start, rng_end + 1):
            ok, bgr = cap.read()
            if not ok:
                break
            run_detect = (idx - rng_start) % stride == 0 or idx == rng_end
            if run_detect:
                rgb = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2RGB)
                ts_ms = int(idx * 1000 / fps)
                res = lm.detect_for_video(
                    _mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb),
                    ts_ms,
                )
                if res.face_landmarks:
                    f = res.face_landmarks[0]
                    detected = (
                        int(f[4].x * w),
                        int(f[4].y * h),
                        int(abs(f[454].x - f[234].x) * w),
                        int(abs(f[152].y - f[10].y) * h),
                    )
                    last_detected = detected
                key_indices.append(idx)
                key_vals.append(last_detected)
                done_infer += 1
                if done_infer % 50 == 0:
                    print(f"   [worker {worker_id}] detect {done_infer}/{total_infer}", flush=True)

        # Interpolate skipped frames within this range
        if len(key_indices) >= 2:
            ki = np.array(key_indices, dtype=np.float64)
            kv = np.array(key_vals, dtype=np.float64)
            all_idx = np.arange(rng_start, rng_end + 1, dtype=np.float64)
            interped = np.column_stack([
                np.interp(all_idx, ki, kv[:, c]) for c in range(4)
            ]).astype(int)
            for j, idx in enumerate(range(rng_start, rng_end + 1)):
                results.append((idx, tuple(interped[j])))
        elif len(key_indices) == 1:
            results.append((key_indices[0], key_vals[0]))

        # Carry last_detected for gap-fill info
        results.append(("last", rng_end, last_detected))

    cap.release()
    lm.close()
    return results, done_infer


def get_face_data_seek(video_path, active_ranges, n_frames, stride=3):
    """
    Seek-based face detection: only decode + detect frames within active_ranges.
    Runs inference every `stride` frames and interpolates the rest with np.interp.
    Frames outside ranges get the last detected position (or default center).

    When there are multiple active ranges, processes them in parallel using
    separate processes (each with its own VideoCapture + FaceLandmarker).
    """
    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    default = (w // 2, h // 2, 100, 100)
    data = [default] * n_frames
    total_read = sum(e - s + 1 for s, e in active_ranges)
    total_infer = sum((e - s) // stride + 1 for s, e in active_ranges)

    n_workers = max(1, os.cpu_count() // 2)

    if n_workers <= 1:
        # Single-worker path — no process overhead
        result, done_infer = _detect_range_worker(
            (video_path, active_ranges, stride, 0)
        )
        _apply_worker_results(data, [result], active_ranges, n_frames, default)
        print(f"   Inference: {done_infer} frames (stride={stride}, {total_read} read)")
    else:
        # Split large ranges so no single worker gets stuck with a huge chunk.
        # Target: each sub-range ≈ total_frames / n_workers.
        from concurrent.futures import ProcessPoolExecutor
        target_per_worker = max(int(fps * 10), total_read // n_workers)
        split_ranges = []
        for s, e in active_ranges:
            rng_len = e - s + 1
            if rng_len <= target_per_worker:
                split_ranges.append((s, e))
            else:
                # Split into chunks of ~target_per_worker frames
                cur = s
                while cur <= e:
                    chunk_end = min(cur + target_per_worker - 1, e)
                    split_ranges.append((cur, chunk_end))
                    cur = chunk_end + 1

        # Greedy bin-pack split ranges across workers
        batches = [[] for _ in range(n_workers)]
        batch_frames = [0] * n_workers
        sorted_ranges = sorted(split_ranges, key=lambda r: r[1] - r[0], reverse=True)
        for rng in sorted_ranges:
            lightest = min(range(n_workers), key=lambda i: batch_frames[i])
            batches[lightest].append(rng)
            batch_frames[lightest] += rng[1] - rng[0] + 1
        # Sort each batch by start frame (MediaPipe requires monotonic timestamps)
        # and filter out empty batches
        worker_args = [
            (video_path, sorted(batch), stride, wi)
            for wi, batch in enumerate(batches) if batch
        ]
        print(f"   Parallel face detection: {len(worker_args)} workers, {len(split_ranges)} chunks from {len(active_ranges)} ranges")
        with ProcessPoolExecutor(max_workers=len(worker_args)) as pool:
            futures = list(pool.map(_detect_range_worker, worker_args))
        all_results = [f[0] for f in futures]
        total_done = sum(f[1] for f in futures)
        _apply_worker_results(data, all_results, active_ranges, n_frames, default)
        print(f"   Inference: {total_done} frames (stride={stride}, {total_read} read, {len(worker_args)} workers)")

    return data, fps, (w, h)


def _apply_worker_results(data, all_results, active_ranges, n_frames, default):
    """Merge worker results into the data array and fill gaps between ranges."""
    # Apply frame-level results from all workers
    last_detected_per_range = {}
    for results in all_results:
        for entry in results:
            if isinstance(entry, tuple) and len(entry) == 3 and entry[0] == "last":
                _, rng_end, last_det = entry
                last_detected_per_range[rng_end] = last_det
            else:
                idx, val = entry
                data[idx] = val

    # Fill gaps between ranges using last detected from the preceding range
    last_detected = default
    for ri, (rng_start, rng_end) in enumerate(active_ranges):
        last_detected = last_detected_per_range.get(rng_end, last_detected)
        next_fill_end = n_frames
        for ns, _ in active_ranges:
            if ns > rng_end:
                next_fill_end = ns
                break
        for fill_idx in range(rng_end + 1, next_fill_end):
            data[fill_idx] = last_detected


# ─── Face detection ──────────────────────────────────────────────────────────


def get_face_data(video_path, active_ranges=None, stride=3):
    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    lm = vision.FaceLandmarker.create_from_options(opts)
    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    default = (w // 2, h // 2, 100, 100)
    last_detected = default
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # First pass: detect at stride intervals, collect keypoints
    key_indices = []
    key_vals = []
    idx = 0
    n_infer = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        skip_range = active_ranges is not None and not _frame_in_ranges(idx, active_ranges)
        run_detect = not skip_range and (idx % stride == 0)
        if run_detect:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            res = lm.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(idx * 1000 / fps)
            )
            if res.face_landmarks:
                f = res.face_landmarks[0]
                last_detected = (
                    int(f[4].x * w),
                    int(f[4].y * h),
                    int(abs(f[454].x - f[234].x) * w),
                    int(abs(f[152].y - f[10].y) * h),
                )
            key_indices.append(idx)
            key_vals.append(last_detected)
            n_infer += 1
        elif not skip_range:
            # Non-stride frame inside active range — will be interpolated
            key_indices.append(idx)
            key_vals.append(None)  # placeholder
        else:
            # Outside active range — carry forward
            key_indices.append(idx)
            key_vals.append(last_detected)
        idx += 1
        if idx % 500 == 0:
            print(f"   detect {idx}/{n_total} ({n_infer} inferred)", flush=True)
    cap.release()
    lm.close()

    # Second pass: interpolate skipped frames
    n = len(key_indices)
    if n == 0:
        return [default] * n_total, fps, (w, h)
    # Build arrays of detected keypoints for interpolation
    det_idx = []
    det_vals = []
    for i, v in zip(key_indices, key_vals):
        if v is not None:
            det_idx.append(i)
            det_vals.append(v)
    if not det_idx:
        return [default] * n_total, fps, (w, h)

    di = np.array(det_idx, dtype=np.float64)
    dv = np.array(det_vals, dtype=np.float64)
    all_frames = np.arange(n_total, dtype=np.float64)
    interped = np.column_stack([
        np.interp(all_frames, di, dv[:, c]) for c in range(4)
    ]).astype(int)
    data = [tuple(row) for row in interped]

    print(f"   Inference: {n_infer}/{n_total} frames (stride={stride})")
    return data, fps, (w, h)


def smooth_data(data, alpha=0.1):
    a = np.array(data, dtype=np.float64)
    o = np.empty_like(a)
    o[0] = a[0]
    inv = 1.0 - alpha
    for i in range(1, len(a)):
        o[i] = alpha * a[i] + inv * o[i - 1]
    return o.astype(np.int32)


EDGE_STRIP_FRAC = 0.04
FADE_WIDTH_FRAC = 0.25


# ─── FFmpeg writer ───────────────────────────────────────────────────────────


def open_ffmpeg_writer(path, w, h, fps, enc):
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-c:v",
        enc,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]
    if enc == "libx264":
        cmd += ["-preset", "fast", "-crf", "18"]
    elif enc == "h264_nvenc":
        cmd += ["-preset", "p4", "-rc", "vbr", "-cq", "20"]
    elif enc == "h264_videotoolbox":
        cmd += ["-q:v", "65"]
    elif enc == "libx265":
        cmd += ["-preset", "fast", "-crf", "22"]
    elif enc == "hevc_videotoolbox":
        cmd += ["-q:v", "65"]
    elif enc == "hevc_nvenc":
        cmd += ["-preset", "p4", "-rc", "vbr", "-cq", "22"]
    elif enc == "libsvtav1":
        cmd += ["-preset", "6", "-crf", "28"]
    elif enc == "libaom-av1":
        cmd += ["-cpu-used", "6", "-crf", "28"]
    elif enc == "av1_videotoolbox":
        cmd += ["-q:v", "65"]
    elif enc == "av1_nvenc":
        cmd += ["-preset", "p4", "-rc", "vbr", "-cq", "28"]
    cmd.append(path)
    return subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def mux_audio(src, silent, out):
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            src,
        ],
        capture_output=True,
        text=True,
    )
    has_audio = probe.returncode == 0 and probe.stdout.strip() != ""

    if has_audio:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                silent,
                "-i",
                src,
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-shortest",
                out,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return
        print(f"   Warning: Mux with audio failed (exit {result.returncode}):")
        print(f"     {result.stderr[-300:]}")
        print(f"     Falling back to video-only ...")

    subprocess.run(
        ["ffmpeg", "-y", "-i", silent, "-c:v", "copy", out],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not has_audio:
        print("   (source has no audio track — skipped)")


# ─── Segment pipeline ────────────────────────────────────────────────────────


def _probe_source_codec(video_path):
    """Return the video codec name of the source file (e.g. 'h264')."""
    r = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name", "-of", "csv=p=0",
            video_path,
        ],
        capture_output=True, text=True,
    )
    return r.stdout.strip() if r.returncode == 0 else ""


def _probe_keyframe_times(video_path):
    """Return sorted list of keyframe timestamps (seconds) in the video."""
    r = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-skip_frame", "nokey",
            "-show_entries", "frame=pts_time",
            "-of", "csv=p=0",
            video_path,
        ],
        capture_output=True, text=True,
    )
    times = []
    for line in r.stdout.strip().split("\n"):
        line = line.strip()
        if line:
            try:
                times.append(float(line))
            except ValueError:
                pass
    return sorted(times)


def _extract_passthrough(input_path, output_path, t_start, t_end, enc):
    """Stream-copy passthrough segment (keyframe-aligned boundaries)."""
    cmd = [
        "ffmpeg", "-y", "-ss", str(t_start), "-to", str(t_end),
        "-i", input_path, "-c:v", "copy", "-an",
        output_path,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)


def _run_ffmpeg_with_progress(cmd, total_frames, fps):
    """Run an FFmpeg command, printing frame progress for long encodes."""
    if total_frames < fps * 10:  # < 10s, don't bother with progress
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        return
    import re
    proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    last_print = 0
    frame_re = re.compile(r"frame=\s*(\d+)")
    buf = ""
    for chunk in iter(lambda: proc.stderr.read(256), ""):
        buf += chunk
        # FFmpeg writes progress on the same line with \r
        lines = buf.split("\r")
        buf = lines[-1]
        for line in lines[:-1]:
            m = frame_re.search(line)
            if m:
                done = int(m.group(1))
                if done - last_print >= 500:
                    print(f"       encode {done}/{total_frames}", flush=True)
                    last_print = done
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def _render_hold_ffmpeg(input_path, output_path, frame_start, frame_end,
                        face_data_stable, z, face_side, dest_x_full,
                        fps, w, h, enc):
    """
    Render a hold region (constant zoom, slowly drifting face) entirely via
    FFmpeg crop+scale — no Python frame loop.

    If the face drifts significantly (≥2% of frame width), builds a
    piecewise-linear interpolation using FFmpeg lerp() expressions sampled
    every KEY_INTERVAL seconds.  Otherwise uses a single static crop.
    """
    KEY_INTERVAL = 5  # seconds between keyframes

    crop_w = int(w / z)
    crop_h = int(h / z)
    n = frame_end - frame_start + 1

    # Compute per-frame crop positions from stabilised face data
    cx_arr = []
    cy_arr = []
    for k in range(n):
        abs_idx = frame_start + k
        fx_st = float(face_data_stable[abs_idx][0])
        fy_st = float(face_data_stable[abs_idx][1])
        cx = fx_st - dest_x_full / z
        cy = fy_st - (h / 2) / z
        cx = max(0, min(cx, w - crop_w))
        cy = max(0, min(cy, h - crop_h))
        cx_arr.append(cx)
        cy_arr.append(cy)

    cx_min, cx_max = min(cx_arr), max(cx_arr)
    cy_min, cy_max = min(cy_arr), max(cy_arr)
    drift = max(cx_max - cx_min, cy_max - cy_min)

    t_start = frame_start / fps
    t_end = (frame_end + 1) / fps

    if drift < 0.02 * w:
        # Static path — drift is negligible, use average position
        avg_cx = int(sum(cx_arr) / n)
        avg_cy = int(sum(cy_arr) / n)
        avg_cx = max(0, min(avg_cx, w - crop_w))
        avg_cy = max(0, min(avg_cy, h - crop_h))
        vf = f"crop={crop_w}:{crop_h}:{avg_cx}:{avg_cy},scale={w}:{h}:flags=bilinear"
    else:
        # Drifting path — sample keyframes and build piecewise lerp expression
        key_interval_frames = int(KEY_INTERVAL * fps)
        keyframe_indices = list(range(0, n, key_interval_frames))
        if keyframe_indices[-1] != n - 1:
            keyframe_indices.append(n - 1)

        # Average crop position around each keyframe for stability
        def _avg_at(idx, arr):
            half_win = int(fps)  # ±1 second window
            lo = max(0, idx - half_win)
            hi = min(n, idx + half_win + 1)
            return sum(arr[lo:hi]) / (hi - lo)

        kf_cx = [_avg_at(i, cx_arr) for i in keyframe_indices]
        kf_cy = [_avg_at(i, cy_arr) for i in keyframe_indices]
        kf_t = [i / fps for i in keyframe_indices]  # time relative to segment start

        # Clamp keyframe positions
        for i in range(len(kf_cx)):
            kf_cx[i] = max(0, min(kf_cx[i], w - crop_w))
            kf_cy[i] = max(0, min(kf_cy[i], h - crop_h))

        def _build_lerp_expr(kf_vals, kf_times):
            """Build chained if(between(t,...), lerp(...), ...) expression."""
            if len(kf_vals) == 1:
                return str(int(kf_vals[0]))
            # Build from last segment backwards
            expr = str(int(kf_vals[-1]))
            for i in range(len(kf_vals) - 2, -1, -1):
                t0 = kf_times[i]
                t1 = kf_times[i + 1]
                v0 = kf_vals[i]
                v1 = kf_vals[i + 1]
                seg_dur = t1 - t0
                if seg_dur <= 0:
                    continue
                seg_expr = f"lerp({int(v0)}\\,{int(v1)}\\,(t-{t0:.3f})/{seg_dur:.3f})"
                expr = f"if(between(t\\,{t0:.3f}\\,{t1:.3f})\\,{seg_expr}\\,{expr})"
            return expr

        cx_expr = _build_lerp_expr(kf_cx, kf_t)
        cy_expr = _build_lerp_expr(kf_cy, kf_t)
        vf = f"crop={crop_w}:{crop_h}:'{cx_expr}':'{cy_expr}',scale={w}:{h}:flags=bilinear"

    cmd = [
        "ffmpeg", "-y", "-ss", str(t_start), "-to", str(t_end),
        "-i", input_path, "-vf", vf, "-c:v", enc, "-pix_fmt", "yuv420p",
        "-an", output_path,
    ]
    # Holds are simple crop+scale — use fast presets since there's no
    # complex motion or effects to preserve
    if "libx264" in enc:
        cmd[-2:-2] = ["-preset", "ultrafast", "-crf", "18"]
    elif "libsvtav1" in enc:
        cmd[-2:-2] = ["-preset", "12", "-crf", "28"]
    elif "videotoolbox" in enc:
        cmd[-2:-2] = ["-q:v", "65"]
    elif "nvenc" in enc:
        cmd[-2:-2] = ["-preset", "p4", "-rc", "vbr", "-cq", "22"]
    _run_ffmpeg_with_progress(cmd, n, fps)


def _render_active_segment(
    input_path, output_path, frame_start, frame_end,
    face_data, face_data_stable, p_curve, zooms,
    blur_strength, blur_n_samples, whip_strength, whip_direction,
    times, overlay, overlay_config, face_side, dest_x_full,
    stabilize, debug_labels, fps, w, h, enc,
):
    """Render a single active segment to its own file."""
    # Check if this is a pure hold region (constant p≈1.0, constant z, no effects)
    seg_p = p_curve[frame_start:frame_end + 1]
    seg_blur = blur_strength[frame_start:frame_end + 1]
    seg_whip = whip_strength[frame_start:frame_end + 1]
    seg_z = zooms[frame_start:frame_end + 1]
    n_seg = frame_end - frame_start + 1

    # Detect hold sub-region within this segment
    is_hold = (seg_p > 0.999) & (seg_blur < 0.001) & (seg_whip < 0.001)
    z_range = float(seg_z[is_hold].max() - seg_z[is_hold].min()) if is_hold.any() else 1.0
    is_pure_hold = is_hold.all() and z_range < 0.01 and not overlay and n_seg > int(fps)

    if is_pure_hold:
        hold_z = float(seg_z[0])
        print(f"     FFmpeg hold: {n_seg} frames at z={hold_z:.2f}", flush=True)
        _render_hold_ffmpeg(
            input_path, output_path, frame_start, frame_end,
            face_data_stable, hold_z, face_side, dest_x_full,
            fps, w, h, enc,
        )
        return

    has_zoom_blur = seg_blur.max() > 0
    has_whip = seg_whip.max() > 0

    # Overlay config lookups
    ovl_pos = overlay_config.get("position", "left") if overlay_config else "left"
    ovl_mg = overlay_config.get("margin", 1.8) if overlay_config else 1.8

    # Gradient fade setup
    need_fade = face_side != "center"
    edge_strip = max(int(w * EDGE_STRIP_FRAC), 1)
    fade_width = int(w * FADE_WIDTH_FRAC)
    base_gradient_3ch = None
    if need_fade:
        ramp = np.linspace(0, 1, fade_width).astype(np.float32)
        base_gradient = np.ones((h, w), dtype=np.float32)
        if face_side == "right":
            base_gradient[:, :fade_width] = ramp[np.newaxis, :]
        else:
            base_gradient[:, w - fade_width:] = ramp[::-1][np.newaxis, :]
        base_gradient_3ch = base_gradient[:, :, np.newaxis]

    # Allocate buffers
    buf_warped = np.empty((h, w, 3), dtype=np.uint8)
    buf_out = np.empty((h, w, 3), dtype=np.uint8)
    buf_warped_f32 = np.empty((h, w, 3), dtype=np.float32)
    buf_fade_alpha = np.empty((h, w, 1), dtype=np.float32)
    buf_blend = np.empty((h, w, 3), dtype=np.float32)
    fade_bg_buf = np.empty((h, w, 3), dtype=np.float32)
    buf_rgb = np.empty((h, w, 3), dtype=np.uint8)
    if has_zoom_blur:
        buf_blur_accum = np.empty((h, w, 3), dtype=np.float32)
        buf_blur_sample = np.empty((h, w, 3), dtype=np.uint8)
    else:
        buf_blur_accum = buf_blur_sample = None

    cap = cv2.VideoCapture(input_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    writer = open_ffmpeg_writer(output_path, w, h, fps, enc)

    for idx in range(frame_start, frame_end + 1):
        ok, bgr = cap.read()
        if not ok:
            break

        p = float(p_curve[idx])
        z = float(zooms[idx])
        local = idx - frame_start

        # Passthrough within active segment (frame has no actual effect)
        if p < 0.001 and blur_strength[idx] < 0.001 and whip_strength[idx] < 0.001:
            cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB, dst=buf_out)
            writer.stdin.write(buf_out.data)
            continue

        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB, dst=buf_rgb)
        rgb = buf_rgb
        t = times[idx]

        # Warp geometry (no-stabilize path only)
        fx_raw, fy_raw, fw_raw, fh_raw = face_data[idx]
        if face_data_stable is not None and p > 0.001:
            fx_st = float(face_data_stable[idx][0])
            fy_st = float(face_data_stable[idx][1])
            fx = lerp(float(fx_raw), fx_st, p)
            fy = lerp(float(fy_raw), fy_st, p)
        else:
            fx, fy = float(fx_raw), float(fy_raw)
        fw, fh = float(fw_raw), float(fh_raw)

        tx = lerp(w / 2, fx, p)
        ty = lerp(h / 2, fy, p)
        dx = lerp(w / 2, dest_x_full, p)
        sx = dx - tx * z
        sy = h / 2 - ty * z
        M = np.float32([[z, 0, sx], [0, z, sy]])
        sfx = fx * z + sx
        sfy = fy * z + sy
        sfw = fw * z
        sfh = fh * z

        # Warp
        cv2.warpAffine(rgb, M, (w, h), dst=buf_warped, borderMode=cv2.BORDER_REPLICATE)

        # Post-warp effects
        if has_zoom_blur and blur_strength[idx] > 0.001:
            apply_zoom_blur(
                buf_warped, rgb, M, w, h,
                float(blur_strength[idx]), int(blur_n_samples[idx]),
                buf_blur_accum, buf_blur_sample,
            )
        if has_whip and whip_strength[idx] > 0.001:
            apply_whip(buf_warped, rgb, M, w, h, float(whip_strength[idx]), whip_direction[idx])

        # Gradient fade
        if p < 0.001 or not need_fade:
            np.copyto(buf_out, buf_warped)
        else:
            if face_side == "right":
                edge_band = buf_warped[:, :edge_strip].mean(axis=1, dtype=np.float32)
            else:
                edge_band = buf_warped[:, w - edge_strip:].mean(axis=1, dtype=np.float32)
            CRUSH_H = 6
            edge_col = edge_band.reshape(h, 1, 3)
            crushed = cv2.resize(edge_col, (1, CRUSH_H), interpolation=cv2.INTER_AREA)
            edge_band = cv2.resize(crushed, (1, h), interpolation=cv2.INTER_LINEAR).reshape(h, 3)
            fade_bg_buf[:] = edge_band[:, np.newaxis, :]
            buf_warped_f32[:] = buf_warped
            np.multiply(base_gradient_3ch, p, out=buf_fade_alpha)
            buf_fade_alpha += 1.0 - p
            np.multiply(buf_warped_f32, buf_fade_alpha, out=buf_blend)
            np.subtract(1.0, buf_fade_alpha, out=buf_fade_alpha)
            np.multiply(fade_bg_buf, buf_fade_alpha, out=buf_warped_f32)
            np.add(buf_blend, buf_warped_f32, out=buf_blend)
            np.clip(buf_blend, 0, 255, out=buf_blend)
            np.copyto(buf_out, buf_blend.astype(np.uint8))

        # Overlay
        if overlay and overlay_config and p > 0.01:
            opacity = min(p * 3.0, 1.0)
            if opacity > 0:
                oi, om = overlay.get_frame(t)
                oh, ow_ = oi.shape[:2]
                if ovl_pos == "left":
                    ox, oy = int(sfx - sfw / 2 * ovl_mg - ow_), int(sfy - oh // 2)
                elif ovl_pos == "right":
                    ox, oy = int(sfx + sfw / 2 * ovl_mg), int(sfy - oh // 2)
                elif ovl_pos == "top":
                    ox, oy = int(sfx - ow_ // 2), int(sfy - sfh / 2 * ovl_mg - oh)
                else:
                    ox, oy = int(sfx - ow_ // 2), int(sfy + sfh / 2 * ovl_mg)
                x1, y1 = max(0, ox), max(0, oy)
                x2, y2 = min(w, ox + ow_), min(h, oy + oh)
                if x1 < x2 and y1 < y2:
                    s1, s2 = x1 - ox, y1 - oy
                    roi = buf_out[y1:y2, x1:x2].astype(np.float32)
                    o = oi[s2:s2 + y2 - y1, s1:s1 + x2 - x1]
                    a = om[s2:s2 + y2 - y1, s1:s1 + x2 - x1] * opacity
                    buf_out[y1:y2, x1:x2] = (o * a + roi * (1.0 - a)).astype(np.uint8)

        # Debug labels
        if debug_labels:
            labels = []
            if p > 0.01:
                labels.append("bounce")
            if has_zoom_blur and blur_strength[idx] > 0.001:
                labels.append("zoom_blur")
            if has_whip and whip_strength[idx] > 0.001:
                labels.append("whip")
            _draw_debug_label(buf_out, labels, h)

        writer.stdin.write(buf_out.data)
        if local % 100 == 0:
            print(f"     frame {local}/{n_seg}", flush=True)

    cap.release()
    writer.stdin.close()
    writer.wait()


def _concat_segments(segment_paths, output_path):
    """Concatenate segments via FFmpeg concat demuxer (stream copy)."""
    tmp_dir = os.path.dirname(segment_paths[0])
    filelist = os.path.join(tmp_dir, "filelist.txt")
    with open(filelist, "w") as f:
        for p in segment_paths:
            f.write(f"file '{p}'\n")
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", filelist, "-c", "copy", output_path,
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
    )


def _run_segment_pipeline(
    input_path, output_path, render_ranges, n_frames, fps,
    face_data, face_data_stable, p_curve, zooms,
    blur_strength, blur_n_samples, whip_strength, whip_direction,
    times, overlay, overlay_config, face_side, dest_x_full,
    stabilize, debug_labels, w, h, enc,
):
    """Orchestrate segment-based rendering: stream-copy passthrough, render active, concat."""
    import bisect
    tmp_dir = tempfile.mkdtemp(prefix="zb_seg_")
    segments = []  # (path, type, frame_start, frame_end)
    seg_idx = 0
    min_hold_frames = int(fps)  # 1 second minimum for FFmpeg hold

    # Probe source keyframes so passthrough can stream-copy between them.
    # Active segments absorb any non-keyframe-aligned frames at boundaries.
    kf_times = _probe_keyframe_times(input_path)
    kf_frames = sorted(set(int(round(t * fps)) for t in kf_times)) if kf_times else []

    def _snap_forward(frame):
        """First keyframe at or after frame."""
        i = bisect.bisect_left(kf_frames, frame)
        return kf_frames[i] if i < len(kf_frames) else None

    def _snap_backward(frame):
        """Last keyframe at or before frame."""
        i = bisect.bisect_right(kf_frames, frame) - 1
        return kf_frames[i] if i >= 0 else None

    prev_end = 0
    for rng_start, rng_end in render_ranges:
        # Passthrough before this range — snap inward to keyframe boundaries
        if rng_start > prev_end:
            pass_start = prev_end
            pass_end = rng_start - 1
            if kf_frames:
                # Snap start forward to first keyframe at or after pass_start
                snapped_start = _snap_forward(pass_start)
                # Snap end backward to last keyframe before pass_end (exclusive)
                # so the segment ends just before the next keyframe
                snapped_end_kf = _snap_backward(pass_end)
                if (snapped_start is not None and snapped_end_kf is not None
                        and snapped_start < snapped_end_kf):
                    # Re-encode the small prefix before first keyframe
                    if snapped_start > pass_start:
                        seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}_active.mp4")
                        segments.append((seg_path, "active", pass_start, snapped_start - 1))
                        seg_idx += 1
                    # Stream-copy the keyframe-aligned middle
                    seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}_pass.mp4")
                    segments.append((seg_path, "passthrough", snapped_start, snapped_end_kf - 1))
                    seg_idx += 1
                    # Re-encode the small suffix after last keyframe
                    if snapped_end_kf <= pass_end:
                        seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}_active.mp4")
                        segments.append((seg_path, "active", snapped_end_kf, pass_end))
                        seg_idx += 1
                else:
                    # Too short to fit keyframes — re-encode the whole thing
                    seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}_active.mp4")
                    segments.append((seg_path, "active", pass_start, pass_end))
                    seg_idx += 1
            else:
                # No keyframe info — fall back to re-encode
                seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}_active.mp4")
                segments.append((seg_path, "active", pass_start, pass_end))
                seg_idx += 1

        # Split active range: find hold sub-region (p≈1, no effects, constant z)
        seg_p = p_curve[rng_start:rng_end + 1]
        seg_blur = blur_strength[rng_start:rng_end + 1]
        seg_whip = whip_strength[rng_start:rng_end + 1]
        is_hold = (seg_p > 0.999) & (seg_blur < 0.001) & (seg_whip < 0.001)
        # Find first and last hold frame
        hold_indices = np.where(is_hold)[0]
        if len(hold_indices) > min_hold_frames and not overlay:
            hold_local_start = int(hold_indices[0])
            hold_local_end = int(hold_indices[-1])
            hold_abs_start = rng_start + hold_local_start
            hold_abs_end = rng_start + hold_local_end
            # Check z is constant in hold region
            hold_z = zooms[hold_abs_start:hold_abs_end + 1]
            z_const = float(hold_z.max() - hold_z.min()) < 0.01

            if z_const and (hold_local_end - hold_local_start + 1) > min_hold_frames:
                # Split into: [transition_in] [hold] [transition_out]
                if hold_abs_start > rng_start:
                    seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}_active.mp4")
                    segments.append((seg_path, "active", rng_start, hold_abs_start - 1))
                    seg_idx += 1
                seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}_active.mp4")
                segments.append((seg_path, "active", hold_abs_start, hold_abs_end))
                seg_idx += 1
                if hold_abs_end < rng_end:
                    seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}_active.mp4")
                    segments.append((seg_path, "active", hold_abs_end + 1, rng_end))
                    seg_idx += 1
                prev_end = rng_end + 1
                continue

        # No hold split — single active segment
        seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}_active.mp4")
        segments.append((seg_path, "active", rng_start, rng_end))
        seg_idx += 1
        prev_end = rng_end + 1

    # Trailing passthrough
    if prev_end < n_frames:
        pass_start = prev_end
        pass_end = n_frames - 1
        if kf_frames:
            snapped_start = _snap_forward(pass_start)
            if snapped_start is not None and snapped_start <= pass_end:
                if snapped_start > pass_start:
                    seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}_active.mp4")
                    segments.append((seg_path, "active", pass_start, snapped_start - 1))
                    seg_idx += 1
                seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}_pass.mp4")
                segments.append((seg_path, "passthrough", snapped_start, pass_end))
                seg_idx += 1
            else:
                seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}_active.mp4")
                segments.append((seg_path, "active", pass_start, pass_end))
                seg_idx += 1
        else:
            seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}_pass.mp4")
            segments.append((seg_path, "passthrough", pass_start, pass_end))
            seg_idx += 1

    n_pass = sum(1 for _, t, *_ in segments if t == "passthrough")
    n_active = sum(1 for _, t, *_ in segments if t == "active")
    pass_frames = sum(fe - fs + 1 for _, t, fs, fe in segments if t == "passthrough")
    print(f"   Segment pipeline: {len(segments)} segments ({n_active} active, {n_pass} passthrough [{pass_frames} frames stream-copy])")

    t0 = time.monotonic()

    # Extract passthrough segments in parallel (stream-copy is I/O bound, not CPU)
    pass_segs = [(s, fs, fe) for s, typ, fs, fe in segments if typ == "passthrough"]
    if pass_segs:
        total_pass_frames = sum(fe - fs + 1 for _, fs, fe in pass_segs)
        def _extract(args):
            path, fs, fe = args
            _extract_passthrough(input_path, path, fs / fps, (fe + 1) / fps, enc)
        with ThreadPoolExecutor(max_workers=min(len(pass_segs), 4)) as pool:
            list(pool.map(_extract, pass_segs))
        print(f"   Passthrough segments: {len(pass_segs)} stream-copied ({total_pass_frames} frames) in {time.monotonic() - t0:.1f}s")

    # Render active segments sequentially (encoder already saturates CPU cores)
    t1 = time.monotonic()
    active_segs = [(s, fs, fe) for s, typ, fs, fe in segments if typ == "active"]
    total_active_frames = sum(fe - fs + 1 for _, fs, fe in active_segs)

    for si, (path, fs, fe) in enumerate(active_segs):
        n_seg = fe - fs + 1
        print(f"   Rendering segment {si+1}/{len(active_segs)}: frames {fs}-{fe} ({n_seg} frames)", flush=True)
        _render_active_segment(
            input_path, path, fs, fe,
            face_data, face_data_stable, p_curve, zooms,
            blur_strength, blur_n_samples, whip_strength, whip_direction,
            times, overlay, overlay_config, face_side, dest_x_full,
            stabilize, debug_labels, fps, w, h, enc,
        )

    elapsed_render = time.monotonic() - t1
    print(f"   Active segments: {len(active_segs)} rendered ({total_active_frames} frames) in {elapsed_render:.1f}s ({total_active_frames / max(elapsed_render, 0.01):.1f} fps)")

    # Concat all segments in order
    segment_paths = [s for s, *_ in segments]
    tmp_concat = os.path.join(tmp_dir, "concat_silent.mp4")
    _concat_segments(segment_paths, tmp_concat)

    # Mux audio
    print("3. Muxing audio ...")
    mux_audio(input_path, tmp_concat, output_path)

    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)

    total = time.monotonic() - t0
    print(f"   Total segment pipeline: {total:.1f}s")
    print(f"Done -> {output_path}")


# ─── Main Effect ─────────────────────────────────────────────────────────────


def _draw_debug_label(buf_out, labels, h):
    """Draw active effect labels in bottom-left corner using cv2.putText."""
    if not labels:
        return
    text = " + ".join(labels)
    y = h - 20
    # BGR text on an RGB buffer — convert color mentally: white is fine
    cv2.putText(
        buf_out,
        text,
        (16, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.8,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        buf_out, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 2, cv2.LINE_AA
    )


def create_zoom_bounce_effect(
    input_path,
    output_path,
    zoom_max=1.4,
    bounces=None,
    bounce_mode="snap",
    face_side="center",
    overlay_config=None,
    text_config=None,
    fade_mode="band",
    stabilize=0.0,
    stabilize_alpha=0.02,
    debug_labels=False,
    detect_holds=False,
):
    """
    Dramatic punch-in / punch-out zoom that tracks a face.

    Args:
        input_path:     Source video file
        output_path:    Output video file
        zoom_max:       Peak zoom level (1.0 = none, 1.4 = 40% punch)
        bounces:        List of bounce tuples. Each can be:
                          (start, end) — uses global bounce_mode/zoom_max
                          (start, end, mode) — per-bounce easing override
                          (start, end, mode, zoom) — per-bounce easing + zoom
                        Defaults to [(1.0, 2.5)] if None.
        bounce_mode:    "smooth" (sin bell), "snap" (fast attack/release),
                        "overshoot" (elastic spring)
        face_side:      "center", "left", or "right" — where face lands on screen.
                        "center" keeps face centered (pure zoom, no lateral shift).
        overlay_config: Overlay dict (same as opte.py) or None
        text_config:    Deprecated alias for overlay_config
        fade_mode:      "band" (per-row edge color) or "average"
        detect_holds:   If False, skip face detection during hold regions
                        (between in→out pairs).  Faster but holds use a static
                        crop instead of drift tracking.
    """
    if bounces is None:
        bounces = [(1.0, 2.5)]
    if overlay_config is None and text_config is not None:
        overlay_config = text_config
    if bounce_mode not in EASE_FUNCTIONS:
        raise ValueError(
            f"Unknown bounce_mode: {bounce_mode!r}. Use: {list(EASE_FUNCTIONS)}"
        )

    print("1. Analyzing face trajectory ...")
    active_ranges = None
    if stabilize == 0:
        # Quick-probe video for fps/frame_count to compute selective ranges
        probe_cap = cv2.VideoCapture(input_path)
        probe_fps = probe_cap.get(cv2.CAP_PROP_FPS)
        probe_n = int(probe_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        probe_cap.release()
        active_ranges = _compute_active_frame_ranges(bounces, probe_fps, probe_n,
                                                       detect_holds=detect_holds)
        if active_ranges is not None:
            detect_frames = sum(e - s + 1 for s, e in active_ranges)
            print(f"   Selective detection: {detect_frames}/{probe_n} frames ({100*detect_frames/max(probe_n,1):.0f}%)")
    if active_ranges is not None:
        raw_data, fps, (w, h) = get_face_data_seek(input_path, active_ranges, probe_n)
    elif stabilize == 0:
        # No face-dependent events (only zoom_blur/whip) — skip detection entirely
        probe_cap = cv2.VideoCapture(input_path)
        fps = probe_cap.get(cv2.CAP_PROP_FPS)
        w, h = int(probe_cap.get(3)), int(probe_cap.get(4))
        probe_n = int(probe_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        probe_cap.release()
        default = (w // 2, h // 2, 100, 100)
        raw_data = [default] * probe_n
        print("   No face-dependent events — skipping detection")
    else:
        raw_data, fps, (w, h) = get_face_data(input_path, active_ranges=active_ranges)
    face_data = smooth_data(raw_data, alpha=0.05)
    n_frames = len(face_data)
    # Heavier smoothing: used for stabilization crop AND to dampen tracking jitter when zoomed
    face_data_stable = smooth_data(raw_data, alpha=stabilize_alpha)

    # Build the bounce curves (supports per-bounce mode + zoom overrides)
    times, p_curve, zooms = build_bounce_curves(
        n_frames, fps, bounces, bounce_mode, zoom_max
    )

    # Build effect curves (zoom_blur, whip)
    blur_strength, blur_n_samples, whip_strength, whip_direction = build_effect_curves(
        n_frames, fps, bounces, bounce_mode, zoom_max
    )
    has_zoom_blur = blur_strength.max() > 0
    has_whip = whip_strength.max() > 0

    # ── Segment pipeline (stabilize==0 only) ────────────────────────────
    if stabilize == 0:
        render_ranges = _compute_render_ranges(bounces, fps, n_frames)
        if render_ranges is not None:
            src_codec = _probe_source_codec(input_path)

            # Overlay prep for segment pipeline
            overlay = None
            if overlay_config:
                if overlay_config.get("type", "text") == "text":
                    mfw = float(np.median(face_data[:, 2]))
                    sfw = mfw * zoom_max
                    mg = overlay_config.get("margin", 1.8)
                    pad = int(w * 0.03)
                    if face_side == "center":
                        fcx = w * 0.5
                    elif face_side == "right":
                        fcx = w * 0.72
                    else:
                        fcx = w * 0.28
                    pos = overlay_config.get("position", "left")
                    if pos == "left":
                        aw = int(fcx - (sfw / 2 * mg) - pad)
                    elif pos == "right":
                        aw = int(w - (fcx + sfw / 2 * mg) - pad)
                    else:
                        aw = int(w * 0.5)
                    overlay_config = {
                        **overlay_config,
                        "_avail_w": max(aw, 100),
                        "_avail_h": int(h * 0.6),
                    }
                overlay = create_overlay(overlay_config)

            if face_side == "center":
                dest_x_full = w * 0.5
            elif face_side == "left":
                dest_x_full = w * 0.28
            else:
                dest_x_full = w * 0.72

            # Encode active segments in same codec as source → stream-copy concat works
            enc = detect_best_encoder(src_codec)

            render_frames = sum(e - s + 1 for s, e in render_ranges)
            print(f"   Render ranges: {len(render_ranges)} range(s), {render_frames}/{n_frames} frames ({100*render_frames/max(n_frames,1):.0f}%)")
            print(f"2. Segment pipeline ({bounce_mode} mode, {len(bounces)} bounce(s)) ...")
            _run_segment_pipeline(
                input_path, output_path, render_ranges, n_frames, fps,
                face_data, face_data_stable, p_curve, zooms,
                blur_strength, blur_n_samples, whip_strength, whip_direction,
                times, overlay, overlay_config, face_side, dest_x_full,
                stabilize, debug_labels, w, h, enc,
            )
            return

    # Precompute per-frame activity mask — when stabilize==0, frames with no
    # effects are pure passthrough (skip warp, fade, overlay, everything).
    if stabilize == 0:
        frame_active = (p_curve > 0.001) | (blur_strength > 0.001) | (whip_strength > 0.001)
        n_active = int(frame_active.sum())
        print(f"   Active frames: {n_active}/{n_frames} ({100*n_active/max(n_frames,1):.0f}%) — rest are passthrough")
    else:
        frame_active = None  # all frames need processing

    # Overlay prep
    overlay = None
    if overlay_config:
        if overlay_config.get("type", "text") == "text":
            mfw = float(np.median(face_data[:, 2]))
            sfw = mfw * zoom_max
            mg = overlay_config.get("margin", 1.8)
            pad = int(w * 0.03)
            if face_side == "center":
                fcx = w * 0.5
            elif face_side == "right":
                fcx = w * 0.72
            else:
                fcx = w * 0.28
            pos = overlay_config.get("position", "left")
            if pos == "left":
                aw = int(fcx - (sfw / 2 * mg) - pad)
            elif pos == "right":
                aw = int(w - (fcx + sfw / 2 * mg) - pad)
            else:
                aw = int(w * 0.5)
            overlay_config = {
                **overlay_config,
                "_avail_w": max(aw, 100),
                "_avail_h": int(h * 0.6),
            }
        overlay = create_overlay(overlay_config)

    # Pre-allocate buffers
    buf_warped = np.empty((h, w, 3), dtype=np.uint8)
    buf_out = np.empty((h, w, 3), dtype=np.uint8)
    buf_warped_f32 = np.empty((h, w, 3), dtype=np.float32)
    buf_fade_alpha = np.empty((h, w, 1), dtype=np.float32)
    buf_blend = np.empty((h, w, 3), dtype=np.float32)
    fade_bg_buf = np.empty((h, w, 3), dtype=np.float32)
    buf_rgb = np.empty((h, w, 3), dtype=np.uint8)
    if has_zoom_blur:
        buf_blur_accum = np.empty((h, w, 3), dtype=np.float32)
        buf_blur_sample = np.empty((h, w, 3), dtype=np.uint8)
    else:
        buf_blur_accum = None
        buf_blur_sample = None
    if face_side == "center":
        dest_x_full = w * 0.5
    elif face_side == "left":
        dest_x_full = w * 0.28
    else:
        dest_x_full = w * 0.72

    # Hoist overlay config lookups
    ovl_pos = overlay_config.get("position", "left") if overlay_config else "left"
    ovl_mg = overlay_config.get("margin", 1.8) if overlay_config else 1.8

    # Precompute which bounce window each frame belongs to (for overlay timing)
    frame_bounce_idx = np.full(n_frames, -1, dtype=np.int32)
    for bi, b in enumerate(bounces):
        if isinstance(b, dict):
            bs, be = b["start"], b["end"]
        else:
            bs, be = b[0], b[1]
        mask = (times >= bs) & (times <= be)
        frame_bounce_idx[mask] = bi

    # Gradient fade setup — not needed for "center" (zoom crops evenly)
    need_fade = face_side != "center"
    edge_strip = max(int(w * EDGE_STRIP_FRAC), 1)
    fade_width = int(w * FADE_WIDTH_FRAC)
    if need_fade:
        ramp = np.linspace(0, 1, fade_width).astype(np.float32)
        base_gradient = np.ones((h, w), dtype=np.float32)
        if face_side == "right":
            base_gradient[:, :fade_width] = ramp[np.newaxis, :]
        else:
            base_gradient[:, w - fade_width :] = ramp[::-1][np.newaxis, :]
        base_gradient_3ch = base_gradient[:, :, np.newaxis]

    # I/O
    enc = detect_best_encoder()
    tmp = output_path + ".tmp_silent.mp4"
    writer = open_ffmpeg_writer(tmp, w, h, fps, enc)
    reader = ThreadedVideoReader(input_path, queue_size=64)

    print(
        f"2. Processing {n_frames} frames ({bounce_mode} mode, {len(bounces)} bounce(s)) ..."
    )
    t0 = time.monotonic()

    for idx in range(n_frames):
        ok, bgr = reader.read()
        if not ok:
            break

        # ── Passthrough fast path ────────────────────────────────────
        if frame_active is not None and not frame_active[idx]:
            cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB, dst=buf_out)
            writer.stdin.write(buf_out.data)
            if idx % 50 == 0:
                print(f"   frame {idx}/{n_frames}", flush=True)
            continue

        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB, dst=buf_rgb)
        rgb = buf_rgb
        t = times[idx]
        p = float(p_curve[idx])
        z = float(zooms[idx])

        # ── Warp geometry ─────────────────────────────────────────────
        # Blend face position toward heavier-smoothed data as zoom
        # increases — prevents jitter amplification at high zoom
        fx_raw, fy_raw, fw_raw, fh_raw = face_data[idx]
        if face_data_stable is not None and p > 0.001:
            fx_st, fy_st = (
                float(face_data_stable[idx][0]),
                float(face_data_stable[idx][1]),
            )
            fx = lerp(float(fx_raw), fx_st, p)
            fy = lerp(float(fy_raw), fy_st, p)
        else:
            fx, fy = float(fx_raw), float(fy_raw)
        fw, fh = float(fw_raw), float(fh_raw)

        if stabilize and p < 0.001:
            # Pure stabilization: subtle crop centered on heavily-smoothed face
            sfx_s, sfy_s = (
                float(face_data_stable[idx][0]),
                float(face_data_stable[idx][1]),
            )
            sz = stabilize
            s_sx = w / 2 - sfx_s * sz
            s_sy = h / 2 - sfy_s * sz
            M = np.float32([[sz, 0, s_sx], [0, sz, s_sy]])
            sfx = fx * sz + s_sx
            sfy = fy * sz + s_sy
            sfw = fw * sz
            sfh = fh * sz
        elif stabilize and p > 0:
            # Blend between stabilization center and bounce center
            sfx_s, sfy_s = (
                float(face_data_stable[idx][0]),
                float(face_data_stable[idx][1]),
            )
            sz_stab = stabilize
            s_sx_stab = w / 2 - sfx_s * sz_stab
            s_sy_stab = h / 2 - sfy_s * sz_stab
            sx_b = dest_x_full - fx * z
            sy_b = h / 2 - fy * z
            ez = lerp(sz_stab, z, p)
            e_sx = lerp(s_sx_stab, sx_b, p)
            e_sy = lerp(s_sy_stab, sy_b, p)
            M = np.float32([[ez, 0, e_sx], [0, ez, e_sy]])
            sfx = fx * ez + e_sx
            sfy = fy * ez + e_sy
            sfw = fw * ez
            sfh = fh * ez
        else:
            # No stabilization: existing behavior
            tx = lerp(w / 2, fx, p)
            ty = lerp(h / 2, fy, p)
            dx = lerp(w / 2, dest_x_full, p)
            sx = dx - tx * z
            sy = h / 2 - ty * z
            M = np.float32([[z, 0, sx], [0, z, sy]])
            sfx = fx * z + sx
            sfy = fy * z + sy
            sfw = fw * z
            sfh = fh * z

        # ── Warp ──────────────────────────────────────────────────────
        cv2.warpAffine(rgb, M, (w, h), dst=buf_warped, borderMode=cv2.BORDER_REPLICATE)

        # ── Post-warp effects (zoom_blur, whip) ─────────────────────
        if has_zoom_blur and blur_strength[idx] > 0.001:
            apply_zoom_blur(
                buf_warped,
                rgb,
                M,
                w,
                h,
                float(blur_strength[idx]),
                int(blur_n_samples[idx]),
                buf_blur_accum,
                buf_blur_sample,
            )
        if has_whip and whip_strength[idx] > 0.001:
            apply_whip(
                buf_warped, rgb, M, w, h, float(whip_strength[idx]), whip_direction[idx]
            )

        # ── Gradient fade ─────────────────────────────────────────────
        if p < 0.001 or not need_fade:
            # No zoom active or center mode — no edge fade needed
            np.copyto(buf_out, buf_warped)
        else:
            if face_side == "right":
                edge_band = buf_warped[:, :edge_strip].mean(axis=1, dtype=np.float32)
            else:
                edge_band = buf_warped[:, w - edge_strip :].mean(
                    axis=1, dtype=np.float32
                )

            CRUSH_H = 6
            edge_col = edge_band.reshape(h, 1, 3)
            crushed = cv2.resize(edge_col, (1, CRUSH_H), interpolation=cv2.INTER_AREA)
            edge_band = cv2.resize(
                crushed, (1, h), interpolation=cv2.INTER_LINEAR
            ).reshape(h, 3)
            fade_bg_buf[:] = edge_band[:, np.newaxis, :]
            fade_bg = fade_bg_buf

            buf_warped_f32[:] = buf_warped
            np.multiply(base_gradient_3ch, p, out=buf_fade_alpha)
            buf_fade_alpha += 1.0 - p
            np.multiply(buf_warped_f32, buf_fade_alpha, out=buf_blend)
            np.subtract(1.0, buf_fade_alpha, out=buf_fade_alpha)
            np.multiply(fade_bg, buf_fade_alpha, out=buf_warped_f32)
            np.add(buf_blend, buf_warped_f32, out=buf_blend)
            np.clip(buf_blend, 0, 255, out=buf_blend)
            np.copyto(buf_out, buf_blend.astype(np.uint8))

        # ── Overlay ───────────────────────────────────────────────────
        if overlay and overlay_config and p > 0.01:
            # Overlay opacity ramps with zoom intensity
            opacity = min(p * 3.0, 1.0)
            if opacity > 0:
                oi, om = overlay.get_frame(t)
                oh, ow_ = oi.shape[:2]
                if ovl_pos == "left":
                    ox, oy = int(sfx - sfw / 2 * ovl_mg - ow_), int(sfy - oh // 2)
                elif ovl_pos == "right":
                    ox, oy = int(sfx + sfw / 2 * ovl_mg), int(sfy - oh // 2)
                elif ovl_pos == "top":
                    ox, oy = int(sfx - ow_ // 2), int(sfy - sfh / 2 * ovl_mg - oh)
                else:
                    ox, oy = int(sfx - ow_ // 2), int(sfy + sfh / 2 * ovl_mg)

                x1, y1 = max(0, ox), max(0, oy)
                x2, y2 = min(w, ox + ow_), min(h, oy + oh)
                if x1 < x2 and y1 < y2:
                    s1, s2 = x1 - ox, y1 - oy
                    roi = buf_out[y1:y2, x1:x2].astype(np.float32)
                    o = oi[s2 : s2 + y2 - y1, s1 : s1 + x2 - x1]
                    a = om[s2 : s2 + y2 - y1, s1 : s1 + x2 - x1] * opacity
                    buf_out[y1:y2, x1:x2] = (o * a + roi * (1.0 - a)).astype(np.uint8)

        # ── Debug labels ──────────────────────────────────────────────
        if debug_labels:
            labels = []
            if p > 0.01:
                labels.append("bounce")
            if has_zoom_blur and blur_strength[idx] > 0.001:
                labels.append("zoom_blur")
            if has_whip and whip_strength[idx] > 0.001:
                labels.append("whip")
            _draw_debug_label(buf_out, labels, h)

        writer.stdin.write(buf_out.data)
        if idx % 50 == 0:
            print(f"   frame {idx}/{n_frames}", flush=True)

    elapsed = time.monotonic() - t0
    actual = min(idx + 1, n_frames)
    print(
        f"   {actual} frames in {elapsed:.1f}s ({actual / max(elapsed, 0.01):.1f} fps)"
    )

    reader.release()
    writer.stdin.close()
    writer.wait()

    print("3. Muxing audio ...")
    mux_audio(input_path, tmp, output_path)
    os.remove(tmp)
    print(f"Done -> {output_path}")


# ─── Usage ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(
        "Use cv_experiments/zoom_bounce_test_runner.py to run multi-video timeline tests.\n"
        "Example: python cv_experiments/zoom_bounce_test_runner.py --all"
    )
