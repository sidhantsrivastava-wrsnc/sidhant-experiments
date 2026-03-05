"""
Zoom-Bounce GPU — Zero-copy NVDEC/NVENC render pipeline for Modal T4
=====================================================================
Drop-in replacement for zoom_bounce.create_zoom_bounce_effect that keeps
all per-frame pixel work on the GPU via CuPy + PyNvVideoCodec.

When PyNvVideoCodec is available:
  - NVDEC decodes directly to GPU memory (zero-copy via DLPack)
  - All effects (warp, zoom_blur, whip, edge_fade, overlay) run on GPU
  - RGB→NV12 conversion via custom CUDA kernel
  - NVENC encodes from GPU memory (zero PCIe transfers in hot path)

When PyNvVideoCodec is unavailable:
  - Falls back to cv2.VideoCapture + ffmpeg pipe (same as before, but
    with pre-allocated buffer pool and GPU whip kernel)
"""

import bisect
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import cv2
import cupy as cp
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from moviepy.editor import TextClip, VideoFileClip

MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

EDGE_STRIP_FRAC = 0.04
FADE_WIDTH_FRAC = 0.25


def lerp(a, b, t):
    return a + (b - a) * t


# ─── Bounce easing functions ────────────────────────────────────────────────


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


# ─── Event parsing ───────────────────────────────────────────────────────────


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
            ev["zoom"] = last_zoom

    return events


# ─── Curve builders ──────────────────────────────────────────────────────────


def build_bounce_curves(n_frames, fps, bounces, default_mode, default_zoom):
    """
    Build per-frame p (intensity 0-1) and zoom arrays.

    Returns: (times, p_curve, zooms) — all shape (n_frames,) float32
    """
    times = np.arange(n_frames, dtype=np.float32) / fps
    p_curve = np.zeros(n_frames, dtype=np.float32)
    zooms = np.ones(n_frames, dtype=np.float32)

    events = _parse_events(bounces, default_mode, default_zoom)

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


# ─── Encoder detection ───────────────────────────────────────────────────────


def _probe_encoder(name):
    try:
        r = subprocess.run(
            [
                "ffmpeg", "-y", "-f", "lavfi",
                "-i", "color=black:s=256x256:d=0.04",
                "-c:v", name, "-f", "null", "-",
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

_encoder_cache = {}


def detect_best_encoder(codec="h264"):
    if codec not in _encoder_cache:
        candidates = _ENCODER_CANDIDATES.get(codec, [f"lib{codec}"])
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


# ─── FFmpeg utilities ─────────────────────────────────────────────────────────


def open_ffmpeg_writer(path, w, h, fps, enc, pix_fmt="rgb24"):
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", pix_fmt,
        "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0", "-c:v", enc,
    ]
    if pix_fmt != "nv12":
        cmd += ["-pix_fmt", "yuv420p"]
    cmd += ["-movflags", "+faststart"]
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


def _run_ffmpeg_with_progress(cmd, total_frames, fps):
    """Run an FFmpeg command, printing frame progress for long encodes."""
    if total_frames < fps * 10:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        return
    proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    last_print = 0
    frame_re = re.compile(r"frame=\s*(\d+)")
    buf = ""
    for chunk in iter(lambda: proc.stderr.read(256), ""):
        buf += chunk
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


def _extract_passthrough(input_path, output_path, t_start, t_end, enc,
                         reencode=False):
    """Extract passthrough segment. Stream-copy when possible, re-encode when
    source codec differs from output codec."""
    if reencode:
        cmd = [
            "ffmpeg", "-y", "-ss", str(t_start), "-to", str(t_end),
            "-i", input_path, "-c:v", enc, "-an",
        ]
        if enc == "h264_nvenc":
            cmd += ["-preset", "p4", "-rc", "vbr", "-cq", "20"]
        elif enc == "h264_videotoolbox":
            cmd += ["-q:v", "65"]
        elif enc == "hevc_nvenc":
            cmd += ["-preset", "p4", "-rc", "vbr", "-cq", "22"]
        else:
            cmd += ["-preset", "fast", "-crf", "18"]
        cmd.append(output_path)
    else:
        cmd = [
            "ffmpeg", "-y", "-ss", str(t_start), "-to", str(t_end),
            "-i", input_path, "-c:v", "copy", "-an",
            output_path,
        ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)


def _render_hold_ffmpeg(input_path, output_path, frame_start, frame_end,
                        face_data_stable, z, face_side, dest_x_full,
                        fps, w, h, enc):
    """
    Render a hold region (constant zoom, slowly drifting face) entirely via
    FFmpeg crop+scale — no Python frame loop.
    """
    KEY_INTERVAL = 5

    crop_w = int(w / z)
    crop_h = int(h / z)
    n = frame_end - frame_start + 1

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
        avg_cx = int(sum(cx_arr) / n)
        avg_cy = int(sum(cy_arr) / n)
        avg_cx = max(0, min(avg_cx, w - crop_w))
        avg_cy = max(0, min(avg_cy, h - crop_h))
        vf = f"crop={crop_w}:{crop_h}:{avg_cx}:{avg_cy},scale={w}:{h}:flags=bilinear"
    else:
        key_interval_frames = int(KEY_INTERVAL * fps)
        keyframe_indices = list(range(0, n, key_interval_frames))
        if keyframe_indices[-1] != n - 1:
            keyframe_indices.append(n - 1)

        def _avg_at(idx, arr):
            half_win = int(fps)
            lo = max(0, idx - half_win)
            hi = min(n, idx + half_win + 1)
            return sum(arr[lo:hi]) / (hi - lo)

        kf_cx = [_avg_at(i, cx_arr) for i in keyframe_indices]
        kf_cy = [_avg_at(i, cy_arr) for i in keyframe_indices]
        kf_t = [i / fps for i in keyframe_indices]

        for i in range(len(kf_cx)):
            kf_cx[i] = max(0, min(kf_cx[i], w - crop_w))
            kf_cy[i] = max(0, min(kf_cy[i], h - crop_h))

        def _build_lerp_expr(kf_vals, kf_times):
            if len(kf_vals) == 1:
                return str(int(kf_vals[0]))
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
    if "libx264" in enc:
        cmd[-2:-2] = ["-preset", "ultrafast", "-crf", "18"]
    elif "libsvtav1" in enc:
        cmd[-2:-2] = ["-preset", "12", "-crf", "28"]
    elif "videotoolbox" in enc:
        cmd[-2:-2] = ["-q:v", "65"]
    elif "nvenc" in enc:
        cmd[-2:-2] = ["-preset", "p4", "-rc", "vbr", "-cq", "22"]
    _run_ffmpeg_with_progress(cmd, n, fps)


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


def mux_audio(src, silent, out):
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a",
            "-show_entries", "stream=index", "-of", "csv=p=0", src,
        ],
        capture_output=True, text=True,
    )
    has_audio = probe.returncode == 0 and probe.stdout.strip() != ""

    if has_audio:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", silent, "-i", src,
                "-c:v", "copy", "-c:a", "aac",
                "-map", "0:v:0", "-map", "1:a:0", "-shortest", out,
            ],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            return
        print(f"   Warning: Mux with audio failed (exit {result.returncode}):")
        print(f"     {result.stderr[-300:]}")
        print(f"     Falling back to video-only ...")

    subprocess.run(
        ["ffmpeg", "-y", "-i", silent, "-c:v", "copy", out],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if not has_audio:
        print("   (source has no audio track — skipped)")


# ─── Overlay classes ─────────────────────────────────────────────────────────


class Overlay:
    def get_frame(self, t):
        raise NotImplementedError


class TextOverlay(Overlay):
    def __init__(self, content, color="white", fontsize=80, font="Arial-Bold",
                 max_width=None, max_height=None):
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


# ─── Face detection ──────────────────────────────────────────────────────────


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
    ffmpeg pipe decoder and FaceLandmarker to process a batch of active ranges.
    """
    video_path, ranges_batch, stride, worker_id = args
    import subprocess as _sp
    import mediapipe as _mp
    from mediapipe.tasks.python import vision as _vision, BaseOptions as _BO

    _probe = _sp.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate",
         "-of", "csv=p=0", video_path],
        capture_output=True, text=True,
    )
    parts = _probe.stdout.strip().split(",")
    w, h = int(parts[0]), int(parts[1])
    fps_str = parts[2]
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / float(den)
    else:
        fps = float(fps_str)

    opts = _vision.FaceLandmarkerOptions(
        base_options=_BO(model_asset_path=MODEL_PATH),
        running_mode=_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    lm = _vision.FaceLandmarker.create_from_options(opts)
    default = (w // 2, h // 2, 100, 100)
    last_detected = default
    results = []
    done_infer = 0
    total_infer = sum((e - s) // stride + 1 for s, e in ranges_batch)
    frame_size = w * h * 3

    for rng_start, rng_end in ranges_batch:
        n_frames = rng_end - rng_start + 1
        t_start = rng_start / fps
        cmd = [
            "ffmpeg", "-ss", str(t_start), "-i", video_path,
            "-frames:v", str(n_frames),
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "pipe:1",
        ]
        proc = _sp.Popen(cmd, stdout=_sp.PIPE, stderr=_sp.DEVNULL)
        key_indices = []
        key_vals = []
        for idx in range(rng_start, rng_end + 1):
            data = proc.stdout.read(frame_size)
            if len(data) != frame_size:
                break
            run_detect = (idx - rng_start) % stride == 0 or idx == rng_end
            if run_detect:
                rgb = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
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

        proc.stdout.close()
        proc.wait()

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

        results.append(("last", rng_end, last_detected))

    lm.close()
    return results, done_infer


def _apply_worker_results(data, all_results, active_ranges, n_frames, default):
    """Merge worker results into the data array and fill gaps between ranges."""
    last_detected_per_range = {}
    for results in all_results:
        for entry in results:
            if isinstance(entry, tuple) and len(entry) == 3 and entry[0] == "last":
                _, rng_end, last_det = entry
                last_detected_per_range[rng_end] = last_det
            else:
                idx, val = entry
                data[idx] = val

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


def get_face_data_seek(video_path, active_ranges, n_frames, stride=3):
    """
    Seek-based face detection: only decode + detect frames within active_ranges.
    Runs inference every `stride` frames and interpolates the rest.
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
        result, done_infer = _detect_range_worker(
            (video_path, active_ranges, stride, 0)
        )
        _apply_worker_results(data, [result], active_ranges, n_frames, default)
        print(f"   Inference: {done_infer} frames (stride={stride}, {total_read} read)")
    else:
        target_per_worker = max(int(fps * 10), total_read // n_workers)
        split_ranges = []
        for s, e in active_ranges:
            rng_len = e - s + 1
            if rng_len <= target_per_worker:
                split_ranges.append((s, e))
            else:
                cur = s
                while cur <= e:
                    chunk_end = min(cur + target_per_worker - 1, e)
                    split_ranges.append((cur, chunk_end))
                    cur = chunk_end + 1

        batches = [[] for _ in range(n_workers)]
        batch_frames = [0] * n_workers
        sorted_ranges = sorted(split_ranges, key=lambda r: r[1] - r[0], reverse=True)
        for rng in sorted_ranges:
            lightest = min(range(n_workers), key=lambda i: batch_frames[i])
            batches[lightest].append(rng)
            batch_frames[lightest] += rng[1] - rng[0] + 1
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


def smooth_data(data, alpha=0.1):
    a = np.array(data, dtype=np.float64)
    o = np.empty_like(a)
    o[0] = a[0]
    inv = 1.0 - alpha
    for i in range(1, len(a)):
        o[i] = alpha * a[i] + inv * o[i - 1]
    return o.astype(np.int32)


# ─── Selective detection helpers ──────────────────────────────────────────────


def _compute_active_frame_ranges(bounces, fps, n_frames, padding_sec=2.0,
                                  detect_holds=True):
    """
    Return sorted list of (start_frame, end_frame) ranges where face detection
    is needed.  Returns None if no face-dependent events exist.
    """
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
                    raw_ranges.append((b["start"], b["end"]))
                    continue
            if action == "out" and pending_in is not None:
                raw_ranges.append((pending_in, b["end"]))
                pending_in = None
                continue
            if action == "out" and not detect_holds:
                raw_ranges.append((b["start"], b["end"]))
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
        f_start = max(0, int((start_sec - padding_sec) * fps))
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


def _compute_render_ranges(bounces, fps, n_frames):
    """
    Return sorted list of (start_frame, end_frame) ranges where any effect
    is active.  Returns None if no events exist.
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

cv2.setNumThreads(1)

_HAS_NVCODEC = False
try:
    import PyNvVideoCodec as nvc
    _HAS_NVCODEC = True
except ImportError:
    pass


_WARP_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void affine_warp(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    int w, int h,
    float inv_z, float neg_sx_over_z, float neg_sy_over_z
) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= w || dy >= h) return;

    float src_xf = dx * inv_z + neg_sx_over_z;
    float src_yf = dy * inv_z + neg_sy_over_z;

    src_xf = fminf(fmaxf(src_xf, 0.0f), (float)(w - 1));
    src_yf = fminf(fmaxf(src_yf, 0.0f), (float)(h - 1));

    int x0 = (int)floorf(src_xf);
    int y0 = (int)floorf(src_yf);
    int x1 = min(x0 + 1, w - 1);
    int y1 = min(y0 + 1, h - 1);
    float fx = src_xf - x0;
    float fy = src_yf - y0;

    int dst_idx = (dy * w + dx) * 3;
    int s00 = (y0 * w + x0) * 3;
    int s10 = (y0 * w + x1) * 3;
    int s01 = (y1 * w + x0) * 3;
    int s11 = (y1 * w + x1) * 3;

    float w00 = (1.0f - fx) * (1.0f - fy);
    float w10 = fx * (1.0f - fy);
    float w01 = (1.0f - fx) * fy;
    float w11 = fx * fy;

    for (int c = 0; c < 3; c++) {
        float val = w00 * src[s00 + c] + w10 * src[s10 + c]
                  + w01 * src[s01 + c] + w11 * src[s11 + c];
        dst[dst_idx + c] = (unsigned char)fminf(fmaxf(val + 0.5f, 0.0f), 255.0f);
    }
}
''', 'affine_warp')

_RGB_TO_NV12_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void rgb_to_nv12(
    const unsigned char* __restrict__ rgb,
    unsigned char* __restrict__ y_plane,
    unsigned char* __restrict__ uv_plane,
    int w, int h
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int rgb_idx = (y * w + x) * 3;
    float r = (float)rgb[rgb_idx];
    float g = (float)rgb[rgb_idx + 1];
    float b = (float)rgb[rgb_idx + 2];

    float yf =  0.257f * r + 0.504f * g + 0.098f * b + 16.0f;
    y_plane[y * w + x] = (unsigned char)fminf(fmaxf(yf + 0.5f, 0.0f), 255.0f);

    if ((x & 1) == 0 && (y & 1) == 0) {
        float sum_r = r, sum_g = g, sum_b = b;
        int count = 1;

        if (x + 1 < w) {
            int idx = (y * w + x + 1) * 3;
            sum_r += rgb[idx]; sum_g += rgb[idx+1]; sum_b += rgb[idx+2];
            count++;
        }
        if (y + 1 < h) {
            int idx = ((y+1) * w + x) * 3;
            sum_r += rgb[idx]; sum_g += rgb[idx+1]; sum_b += rgb[idx+2];
            count++;
        }
        if (x + 1 < w && y + 1 < h) {
            int idx = ((y+1) * w + x + 1) * 3;
            sum_r += rgb[idx]; sum_g += rgb[idx+1]; sum_b += rgb[idx+2];
            count++;
        }

        float inv_c = 1.0f / count;
        float ar = sum_r * inv_c;
        float ag = sum_g * inv_c;
        float ab = sum_b * inv_c;

        float uf = -0.148f * ar - 0.291f * ag + 0.439f * ab + 128.0f;
        float vf =  0.439f * ar - 0.368f * ag - 0.071f * ab + 128.0f;

        int uv_idx = (y / 2) * w + x;
        uv_plane[uv_idx]     = (unsigned char)fminf(fmaxf(uf + 0.5f, 0.0f), 255.0f);
        uv_plane[uv_idx + 1] = (unsigned char)fminf(fmaxf(vf + 0.5f, 0.0f), 255.0f);
    }
}
''', 'rgb_to_nv12')

_DIRECTIONAL_BLUR_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void directional_blur(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    int w, int h,
    int kernel_radius,
    float strength,
    int horizontal
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int idx = (y * w + x) * 3;
    float sum_r = 0, sum_g = 0, sum_b = 0;
    int count = 0;

    for (int k = -kernel_radius; k <= kernel_radius; k++) {
        int sx, sy;
        if (horizontal) {
            sx = x + k;
            sy = y;
        } else {
            sx = x;
            sy = y + k;
        }
        sx = max(0, min(sx, w - 1));
        sy = max(0, min(sy, h - 1));
        int si = (sy * w + sx) * 3;
        sum_r += src[si];
        sum_g += src[si + 1];
        sum_b += src[si + 2];
        count++;
    }

    float inv_c = 1.0f / count;
    float br = sum_r * inv_c;
    float bg = sum_g * inv_c;
    float bb = sum_b * inv_c;

    float orig_r = src[idx];
    float orig_g = src[idx + 1];
    float orig_b = src[idx + 2];

    dst[idx]     = (unsigned char)(orig_r + (br - orig_r) * strength + 0.5f);
    dst[idx + 1] = (unsigned char)(orig_g + (bg - orig_g) * strength + 0.5f);
    dst[idx + 2] = (unsigned char)(orig_b + (bb - orig_b) * strength + 0.5f);
}
''', 'directional_blur')

_BLOCK = (16, 16)


def _gpu_warp(g_src, g_dst, w, h, z, sx, sy):
    """Run the affine warp kernel: M = [[z,0,sx],[0,z,sy]]."""
    inv_z = 1.0 / z
    neg_sx_over_z = -sx / z
    neg_sy_over_z = -sy / z
    grid = ((w + _BLOCK[0] - 1) // _BLOCK[0],
            (h + _BLOCK[1] - 1) // _BLOCK[1])
    _WARP_KERNEL(grid, _BLOCK,
                 (g_src, g_dst, np.int32(w), np.int32(h),
                  np.float32(inv_z), np.float32(neg_sx_over_z),
                  np.float32(neg_sy_over_z)))


def _gpu_rgb_to_nv12(g_rgb, g_y, g_uv, w, h):
    """Convert RGB (h,w,3 uint8) to NV12 Y + UV planes on GPU."""
    grid = ((w + _BLOCK[0] - 1) // _BLOCK[0],
            (h + _BLOCK[1] - 1) // _BLOCK[1])
    _RGB_TO_NV12_KERNEL(grid, _BLOCK,
                        (g_rgb, g_y, g_uv, np.int32(w), np.int32(h)))


class GPUBufferPool:
    """Pre-allocate all GPU buffers for active segment rendering.

    Eliminates transient allocations in the hot loop. Double-buffered NV12
    planes allow encode of frame N to overlap with compute of frame N+1.
    """

    def __init__(self, h, w, has_zoom_blur=False, has_whip=False, need_nv12=False):
        self.h = h
        self.w = w

        self.rgb = cp.empty((h, w, 3), dtype=cp.uint8)
        self.warped = cp.empty((h, w, 3), dtype=cp.uint8)
        self.out = cp.empty((h, w, 3), dtype=cp.uint8)

        self.warped_f32 = cp.empty((h, w, 3), dtype=cp.float32)
        self.blend_f32 = cp.empty((h, w, 3), dtype=cp.float32)

        self.fade_alpha = cp.empty((h, w, 1), dtype=cp.float32)
        self.fade_bg = cp.empty((h, w, 3), dtype=cp.float32)
        self.inv_alpha = cp.empty((h, w, 1), dtype=cp.float32)

        if has_zoom_blur:
            self.blur_accum = cp.empty((h, w, 3), dtype=cp.float32)
            self.blur_sample = cp.empty((h, w, 3), dtype=cp.uint8)
            self.blur_sample_f32 = cp.empty((h, w, 3), dtype=cp.float32)
        else:
            self.blur_accum = self.blur_sample = self.blur_sample_f32 = None

        if has_whip:
            self.whip_dst = cp.empty((h, w, 3), dtype=cp.uint8)
        else:
            self.whip_dst = None

        if need_nv12:
            nv12_h = h + h // 2
            self.nv12_a = cp.empty((nv12_h, w), dtype=cp.uint8)
            self.nv12_b = cp.empty((nv12_h, w), dtype=cp.uint8)
        else:
            self.nv12_a = self.nv12_b = None

        self._nv12_ping = True

    def get_nv12_buffers(self):
        """Return (nv12_full, y_plane, uv_plane) and toggle for next frame."""
        buf = self.nv12_a if self._nv12_ping else self.nv12_b
        self._nv12_ping = not self._nv12_ping
        y = buf[:self.h]
        uv = buf[self.h:]
        return buf, y, uv


def _gpu_zoom_blur(pool, g_rgb, w, h, z, sx, sy, strength, n_samples):
    """GPU zoom blur using pre-allocated pool buffers."""
    cx, cy = w / 2.0, h / 2.0
    spread = 0.05 * strength * z

    pool.blur_accum[:] = 0.0
    for i in range(n_samples):
        t = (i / max(n_samples - 1, 1)) * 2.0 - 1.0
        dz = t * spread
        sz = z + dz
        s_sx = sx + cx * (z - sz)
        s_sy = sy + cy * (z - sz)
        _gpu_warp(g_rgb, pool.blur_sample, w, h, sz, s_sx, s_sy)
        cp.copyto(pool.blur_sample_f32, pool.blur_sample, casting='unsafe')
        cp.add(pool.blur_accum, pool.blur_sample_f32, out=pool.blur_accum)

    pool.blur_accum /= n_samples

    cp.copyto(pool.warped_f32, pool.warped, casting='unsafe')
    cp.subtract(pool.blur_accum, pool.warped_f32, out=pool.blend_f32)
    cp.multiply(pool.blend_f32, strength, out=pool.blend_f32)
    cp.add(pool.warped_f32, pool.blend_f32, out=pool.blend_f32)
    cp.clip(pool.blend_f32, 0, 255, out=pool.blend_f32)
    cp.copyto(pool.warped, pool.blend_f32, casting='unsafe')


def _gpu_whip(pool, w, h, strength, direction):
    """GPU directional blur via CUDA kernel."""
    if strength < 0.001:
        return

    kernel_radius = min(int(strength * 80) + 1, 40)
    horizontal = 1 if direction == "h" else 0

    grid = ((w + _BLOCK[0] - 1) // _BLOCK[0],
            (h + _BLOCK[1] - 1) // _BLOCK[1])
    _DIRECTIONAL_BLUR_KERNEL(
        grid, _BLOCK,
        (pool.warped, pool.whip_dst, np.int32(w), np.int32(h),
         np.int32(kernel_radius), np.float32(strength), np.int32(horizontal))
    )
    cp.copyto(pool.warped, pool.whip_dst)


def _gpu_edge_fade(pool, g_base_gradient_3ch, w, h, edge_strip, face_side, p):
    """GPU edge fade using pre-allocated pool buffers."""
    CRUSH_H = 6

    if face_side == "right":
        edge_band = pool.warped[:, :edge_strip].mean(axis=1, dtype=cp.float32)
    else:
        edge_band = pool.warped[:, w - edge_strip:].mean(axis=1, dtype=cp.float32)

    edge_col = edge_band.reshape(h, 1, 3)
    group_size = h // CRUSH_H
    remainder = h - group_size * CRUSH_H
    if remainder == 0:
        crushed = edge_col.reshape(CRUSH_H, group_size, 1, 3).mean(axis=1)
    else:
        trunc = group_size * CRUSH_H
        crushed = edge_col[:trunc].reshape(CRUSH_H, group_size, 1, 3).mean(axis=1)

    expanded = cp.repeat(crushed, (h + CRUSH_H - 1) // CRUSH_H, axis=0)[:h]
    pool.fade_bg[:] = expanded.reshape(h, 1, 3)

    cp.copyto(pool.warped_f32, pool.warped, casting='unsafe')
    cp.multiply(g_base_gradient_3ch, p, out=pool.fade_alpha)
    pool.fade_alpha += (1.0 - p)

    cp.multiply(pool.warped_f32, pool.fade_alpha, out=pool.blend_f32)
    cp.subtract(1.0, pool.fade_alpha, out=pool.inv_alpha)
    g_bg_weighted = pool.fade_bg * pool.inv_alpha
    cp.add(pool.blend_f32, g_bg_weighted, out=pool.blend_f32)
    cp.clip(pool.blend_f32, 0, 255, out=pool.blend_f32)
    cp.copyto(pool.out, pool.blend_f32, casting='unsafe')


def _gpu_overlay_blend(pool, g_ovl_img, g_ovl_mask, opacity, ox, oy, w, h):
    """GPU alpha blend overlay onto pool.out at (ox, oy)."""
    oh, ow_ = g_ovl_img.shape[:2]
    x1, y1 = max(0, ox), max(0, oy)
    x2, y2 = min(w, ox + ow_), min(h, oy + oh)
    if x1 >= x2 or y1 >= y2:
        return
    s1, s2 = x1 - ox, y1 - oy
    sw, sh = x2 - x1, y2 - y1

    roi = pool.out[y1:y2, x1:x2].astype(cp.float32)
    o = g_ovl_img[s2:s2 + sh, s1:s1 + sw]
    a = g_ovl_mask[s2:s2 + sh, s1:s1 + sw] * opacity
    result = o * a + roi * (1.0 - a)
    pool.out[y1:y2, x1:x2] = result.astype(cp.uint8)


class _NV12GPUFrame:
    """Wraps a contiguous CuPy NV12 buffer for NVENC GPU input."""

    def __init__(self, nv12_full, y_plane, uv_plane, width, height):
        self._full = nv12_full
        self._y = y_plane
        self._uv = uv_plane
        self._width = width
        self._height = height

    @property
    def __cuda_array_interface__(self):
        return self._full.__cuda_array_interface__

    def cuda(self):
        return [self._y.reshape(self._height, self._width, 1),
                self._uv.reshape(self._height // 2, self._width // 2, 2)]


def _render_active_segment_nvcodec(
    input_path, output_path, frame_start, frame_end,
    face_data, face_data_stable, p_curve, zooms,
    blur_strength, blur_n_samples, whip_strength, whip_direction,
    times, overlay, overlay_config, face_side, dest_x_full,
    stabilize, debug_labels, fps, w, h,
):
    """Zero-copy GPU render using PyNvVideoCodec decode/encode."""
    seg_p = p_curve[frame_start:frame_end + 1]
    seg_blur = blur_strength[frame_start:frame_end + 1]
    seg_whip = whip_strength[frame_start:frame_end + 1]
    n_seg = frame_end - frame_start + 1

    has_zoom_blur = seg_blur.max() > 0
    has_whip = seg_whip.max() > 0

    ovl_pos = overlay_config.get("position", "left") if overlay_config else "left"
    ovl_mg = overlay_config.get("margin", 1.8) if overlay_config else 1.8

    need_fade = face_side != "center"
    edge_strip = max(int(w * EDGE_STRIP_FRAC), 1)
    fade_width = int(w * FADE_WIDTH_FRAC)

    pool = GPUBufferPool(h, w, has_zoom_blur=has_zoom_blur,
                         has_whip=has_whip, need_nv12=True)

    g_base_gradient_3ch = None
    if need_fade:
        ramp = np.linspace(0, 1, fade_width).astype(np.float32)
        base_gradient = np.ones((h, w), dtype=np.float32)
        if face_side == "right":
            base_gradient[:, :fade_width] = ramp[np.newaxis, :]
        else:
            base_gradient[:, w - fade_width:] = ramp[::-1][np.newaxis, :]
        g_base_gradient_3ch = cp.asarray(base_gradient[:, :, np.newaxis])

    g_ovl_img = g_ovl_mask = None
    if overlay:
        oi, om = overlay.get_frame(0)
        g_ovl_img = cp.asarray(oi)
        g_ovl_mask = cp.asarray(om)

    decoder = nvc.SimpleDecoder(
        input_path, gpu_id=0,
        use_device_memory=True,
        output_color_type=nvc.OutputColorType.RGB,
    )

    raw_h264_path = output_path + ".h264"
    encoder = nvc.CreateEncoder(
        w, h, "NV12",
        usecpuinputbuffer=False, gpu_id=0,
        codec="h264",
        preset="P4",
        tuning_info="high_quality",
        fps=int(round(fps)),
    )

    bitstream_file = open(raw_h264_path, "wb")

    damp_fx = damp_fy = None

    for idx in range(frame_start, frame_end + 1):
        p = float(p_curve[idx])
        z = float(zooms[idx])
        local = idx - frame_start

        surface = decoder[idx]
        g_frame = cp.from_dlpack(surface)

        if g_frame.shape != (h, w, 3):
            if g_frame.ndim == 3 and g_frame.shape[0] == 3:
                g_frame = cp.transpose(g_frame, (1, 2, 0))

        if p < 0.001 and blur_strength[idx] < 0.001 and whip_strength[idx] < 0.001:
            damp_fx = damp_fy = None
            cp.copyto(pool.rgb, g_frame)
            nv12_full, nv12_y, nv12_uv = pool.get_nv12_buffers()
            _gpu_rgb_to_nv12(pool.rgb, nv12_y, nv12_uv, w, h)
            frame_obj = _NV12GPUFrame(nv12_full, nv12_y, nv12_uv, w, h)
            bitstream = encoder.Encode(frame_obj)
            if bitstream:
                bitstream_file.write(bytes(bitstream))
            continue

        cp.copyto(pool.rgb, g_frame)

        t = times[idx]

        fx_raw, fy_raw, fw_raw, fh_raw = face_data[idx]
        if face_data_stable is not None and p > 0.001:
            fx_st = float(face_data_stable[idx][0])
            fy_st = float(face_data_stable[idx][1])
            fx = lerp(float(fx_raw), fx_st, p)
            fy = lerp(float(fy_raw), fy_st, p)
        else:
            fx, fy = float(fx_raw), float(fy_raw)
        fw, fh = float(fw_raw), float(fh_raw)

        # Zoom-proportional dampen: suppress tracking jitter that zoom amplifies
        if z > 1.001:
            dampen_alpha = 0.03
            if damp_fx is None:
                damp_fx, damp_fy = fx, fy
            else:
                damp_fx = dampen_alpha * fx + (1.0 - dampen_alpha) * damp_fx
                damp_fy = dampen_alpha * fy + (1.0 - dampen_alpha) * damp_fy
            fx, fy = damp_fx, damp_fy
        else:
            damp_fx = damp_fy = None

        tx = lerp(w / 2, fx, p)
        ty = lerp(h / 2, fy, p)
        dx = lerp(w / 2, dest_x_full, p)
        sx_val = dx - tx * z
        sy_val = h / 2 - ty * z
        sx_val = max((1.0 - z) * (w - 1), min(0.0, sx_val))
        sy_val = max((1.0 - z) * (h - 1), min(0.0, sy_val))

        sfx = fx * z + sx_val
        sfy = fy * z + sy_val
        sfw = fw * z
        sfh = fh * z

        _gpu_warp(pool.rgb, pool.warped, w, h, z, sx_val, sy_val)

        if has_zoom_blur and blur_strength[idx] > 0.001:
            _gpu_zoom_blur(
                pool, pool.rgb, w, h, z, sx_val, sy_val,
                float(blur_strength[idx]), int(blur_n_samples[idx]),
            )

        if has_whip and whip_strength[idx] > 0.001:
            _gpu_whip(pool, w, h, float(whip_strength[idx]), whip_direction[idx])

        if p < 0.001 or not need_fade:
            cp.copyto(pool.out, pool.warped)
        else:
            _gpu_edge_fade(pool, g_base_gradient_3ch, w, h,
                           edge_strip, face_side, p)

        if overlay and overlay_config and p > 0.01:
            opacity = min(p * 3.0, 1.0)
            if opacity > 0:
                if hasattr(overlay, 'clip'):
                    oi, om = overlay.get_frame(t)
                    g_ovl_img = cp.asarray(oi)
                    g_ovl_mask = cp.asarray(om)

                oh_, ow_ = g_ovl_img.shape[:2]
                if ovl_pos == "left":
                    ox, oy = int(sfx - sfw / 2 * ovl_mg - ow_), int(sfy - oh_ // 2)
                elif ovl_pos == "right":
                    ox, oy = int(sfx + sfw / 2 * ovl_mg), int(sfy - oh_ // 2)
                elif ovl_pos == "top":
                    ox, oy = int(sfx - ow_ // 2), int(sfy - sfh / 2 * ovl_mg - oh_)
                else:
                    ox, oy = int(sfx - ow_ // 2), int(sfy + sfh / 2 * ovl_mg)

                _gpu_overlay_blend(pool, g_ovl_img, g_ovl_mask, opacity,
                                   ox, oy, w, h)

        nv12_full, nv12_y, nv12_uv = pool.get_nv12_buffers()
        _gpu_rgb_to_nv12(pool.out, nv12_y, nv12_uv, w, h)
        frame_obj = _NV12GPUFrame(nv12_full, nv12_y, nv12_uv, w, h)
        bitstream = encoder.Encode(frame_obj)
        if bitstream:
            bitstream_file.write(bytes(bitstream))

        if local % 100 == 0:
            print(f"     frame {local}/{n_seg}", flush=True)

    rendered = local + 1
    expected = frame_end - frame_start + 1
    if rendered < expected:
        print(f"     WARNING: decoded {rendered}/{expected} frames (segment may be short)", flush=True)

    remaining = encoder.EndEncode()
    if remaining:
        bitstream_file.write(bytes(remaining))
    bitstream_file.close()

    subprocess.run(
        ["ffmpeg", "-y", "-i", raw_h264_path,
         "-c:v", "copy", "-an", output_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True,
    )
    os.remove(raw_h264_path)


class _ThreadedDecoder:
    """Decode frames via ffmpeg pipe in a background thread.

    Outputs raw RGB24 frames. The background thread prefetches into a small
    queue so decode of frame N+1 overlaps GPU compute of frame N.
    """

    def __init__(self, input_path, frame_start, frame_end, w, h, fps):
        self._queue = queue.Queue(maxsize=2)
        self._frame_size = w * h * 3
        self._shape = (h, w, 3)
        n_frames = frame_end - frame_start + 1
        t_start = frame_start / fps
        cmd = [
            "ffmpeg",
            "-ss", str(t_start),
            "-i", input_path,
            "-frames:v", str(n_frames),
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "pipe:1",
        ]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._thread = threading.Thread(
            target=self._loop, args=(n_frames,), daemon=True)
        self._thread.start()

    def _loop(self, n_frames):
        for _ in range(n_frames):
            data = self._proc.stdout.read(self._frame_size)
            if len(data) != self._frame_size:
                self._queue.put((False, None))
                return
            frame = np.frombuffer(data, dtype=np.uint8).reshape(
                self._shape).copy()
            self._queue.put((True, frame))
        self._queue.put((False, None))

    def read(self):
        try:
            return self._queue.get(timeout=30)
        except queue.Empty:
            return (False, None)

    def release(self):
        self._proc.stdout.close()
        self._proc.terminate()
        self._proc.wait()


def _render_active_segment_fallback(
    input_path, output_path, frame_start, frame_end,
    face_data, face_data_stable, p_curve, zooms,
    blur_strength, blur_n_samples, whip_strength, whip_direction,
    times, overlay, overlay_config, face_side, dest_x_full,
    stabilize, debug_labels, fps, w, h, enc,
):
    """Fallback GPU render using cv2.VideoCapture + ffmpeg pipe."""
    seg_p = p_curve[frame_start:frame_end + 1]
    seg_blur = blur_strength[frame_start:frame_end + 1]
    seg_whip = whip_strength[frame_start:frame_end + 1]
    seg_z = zooms[frame_start:frame_end + 1]
    n_seg = frame_end - frame_start + 1

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

    ovl_pos = overlay_config.get("position", "left") if overlay_config else "left"
    ovl_mg = overlay_config.get("margin", 1.8) if overlay_config else 1.8

    need_fade = face_side != "center"
    edge_strip = max(int(w * EDGE_STRIP_FRAC), 1)
    fade_width = int(w * FADE_WIDTH_FRAC)

    pool = GPUBufferPool(h, w, has_zoom_blur=has_zoom_blur,
                         has_whip=has_whip, need_nv12=True)

    g_base_gradient_3ch = None
    if need_fade:
        ramp = np.linspace(0, 1, fade_width).astype(np.float32)
        base_gradient = np.ones((h, w), dtype=np.float32)
        if face_side == "right":
            base_gradient[:, :fade_width] = ramp[np.newaxis, :]
        else:
            base_gradient[:, w - fade_width:] = ramp[::-1][np.newaxis, :]
        g_base_gradient_3ch = cp.asarray(base_gradient[:, :, np.newaxis])

    g_ovl_img = g_ovl_mask = None
    if overlay:
        oi, om = overlay.get_frame(0)
        g_ovl_img = cp.asarray(oi)
        g_ovl_mask = cp.asarray(om)

    buf_nv12_cpu = np.empty((h + h // 2, w), dtype=np.uint8)

    decoder = _ThreadedDecoder(input_path, frame_start, frame_end, w, h, fps)
    writer = open_ffmpeg_writer(output_path, w, h, fps, enc, pix_fmt="nv12")

    damp_fx = damp_fy = None
    _rendered_count = 0
    for idx in range(frame_start, frame_end + 1):
        ok, rgb = decoder.read()
        if not ok:
            break

        p = float(p_curve[idx])
        z = float(zooms[idx])
        local = idx - frame_start
        _rendered_count = local + 1

        if p < 0.001 and blur_strength[idx] < 0.001 and whip_strength[idx] < 0.001:
            damp_fx = damp_fy = None
            pool.rgb[:] = cp.asarray(rgb)
            nv12_full, nv12_y, nv12_uv = pool.get_nv12_buffers()
            _gpu_rgb_to_nv12(pool.rgb, nv12_y, nv12_uv, w, h)
            cp.asnumpy(nv12_full, out=buf_nv12_cpu)
            writer.stdin.write(buf_nv12_cpu.data)
            continue

        pool.rgb[:] = cp.asarray(rgb)

        t = times[idx]

        fx_raw, fy_raw, fw_raw, fh_raw = face_data[idx]
        if face_data_stable is not None and p > 0.001:
            fx_st = float(face_data_stable[idx][0])
            fy_st = float(face_data_stable[idx][1])
            fx = lerp(float(fx_raw), fx_st, p)
            fy = lerp(float(fy_raw), fy_st, p)
        else:
            fx, fy = float(fx_raw), float(fy_raw)
        fw, fh = float(fw_raw), float(fh_raw)

        # Zoom-proportional dampen: suppress tracking jitter that zoom amplifies
        if z > 1.001:
            dampen_alpha = 0.03
            if damp_fx is None:
                damp_fx, damp_fy = fx, fy
            else:
                damp_fx = dampen_alpha * fx + (1.0 - dampen_alpha) * damp_fx
                damp_fy = dampen_alpha * fy + (1.0 - dampen_alpha) * damp_fy
            fx, fy = damp_fx, damp_fy
        else:
            damp_fx = damp_fy = None

        tx = lerp(w / 2, fx, p)
        ty = lerp(h / 2, fy, p)
        dx = lerp(w / 2, dest_x_full, p)
        sx_val = dx - tx * z
        sy_val = h / 2 - ty * z
        sx_val = max((1.0 - z) * (w - 1), min(0.0, sx_val))
        sy_val = max((1.0 - z) * (h - 1), min(0.0, sy_val))

        sfx = fx * z + sx_val
        sfy = fy * z + sy_val
        sfw = fw * z
        sfh = fh * z

        _gpu_warp(pool.rgb, pool.warped, w, h, z, sx_val, sy_val)

        if has_zoom_blur and blur_strength[idx] > 0.001:
            _gpu_zoom_blur(
                pool, pool.rgb, w, h, z, sx_val, sy_val,
                float(blur_strength[idx]), int(blur_n_samples[idx]),
            )

        if has_whip and whip_strength[idx] > 0.001:
            _gpu_whip(pool, w, h, float(whip_strength[idx]), whip_direction[idx])

        if p < 0.001 or not need_fade:
            cp.copyto(pool.out, pool.warped)
        else:
            _gpu_edge_fade(pool, g_base_gradient_3ch, w, h,
                           edge_strip, face_side, p)

        if overlay and overlay_config and p > 0.01:
            opacity = min(p * 3.0, 1.0)
            if opacity > 0:
                if hasattr(overlay, 'clip'):
                    oi, om = overlay.get_frame(t)
                    g_ovl_img = cp.asarray(oi)
                    g_ovl_mask = cp.asarray(om)

                oh_, ow_ = g_ovl_img.shape[:2]
                if ovl_pos == "left":
                    ox, oy = int(sfx - sfw / 2 * ovl_mg - ow_), int(sfy - oh_ // 2)
                elif ovl_pos == "right":
                    ox, oy = int(sfx + sfw / 2 * ovl_mg), int(sfy - oh_ // 2)
                elif ovl_pos == "top":
                    ox, oy = int(sfx - ow_ // 2), int(sfy - sfh / 2 * ovl_mg - oh_)
                else:
                    ox, oy = int(sfx - ow_ // 2), int(sfy + sfh / 2 * ovl_mg)

                _gpu_overlay_blend(pool, g_ovl_img, g_ovl_mask, opacity,
                                   ox, oy, w, h)

        nv12_full, nv12_y, nv12_uv = pool.get_nv12_buffers()
        _gpu_rgb_to_nv12(pool.out, nv12_y, nv12_uv, w, h)
        cp.asnumpy(nv12_full, out=buf_nv12_cpu)
        writer.stdin.write(buf_nv12_cpu.data)

        if local % 100 == 0:
            print(f"     frame {local}/{n_seg}", flush=True)

    expected = frame_end - frame_start + 1
    if _rendered_count < expected:
        print(f"     WARNING: decoded {rendered}/{expected} frames (segment may be short)", flush=True)

    decoder.release()
    writer.stdin.close()
    writer.wait()


def _render_active_segment_gpu(
    input_path, output_path, frame_start, frame_end,
    face_data, face_data_stable, p_curve, zooms,
    blur_strength, blur_n_samples, whip_strength, whip_direction,
    times, overlay, overlay_config, face_side, dest_x_full,
    stabilize, debug_labels, fps, w, h, enc,
):
    """GPU render: cv2 decode + GPU effects + ffmpeg h264_nvenc encode."""
    _render_active_segment_fallback(
        input_path, output_path, frame_start, frame_end,
        face_data, face_data_stable, p_curve, zooms,
        blur_strength, blur_n_samples, whip_strength, whip_direction,
        times, overlay, overlay_config, face_side, dest_x_full,
        stabilize, debug_labels, fps, w, h, enc,
    )


def _run_segment_pipeline_gpu(
    input_path, output_path, render_ranges, n_frames, fps,
    face_data, face_data_stable, p_curve, zooms,
    blur_strength, blur_n_samples, whip_strength, whip_direction,
    times, overlay, overlay_config, face_side, dest_x_full,
    stabilize, debug_labels, w, h, enc, src_codec="h264",
):
    """GPU version of _run_segment_pipeline."""
    tmp_dir = tempfile.mkdtemp(prefix="zb_seg_")
    segments = []
    seg_idx = 0
    min_hold_frames = int(fps)

    kf_times = _probe_keyframe_times(input_path)
    kf_frames = sorted(set(int(round(t * fps)) for t in kf_times)) if kf_times else []

    def _snap_forward(frame):
        i = bisect.bisect_left(kf_frames, frame)
        return kf_frames[i] if i < len(kf_frames) else None

    def _snap_backward(frame):
        i = bisect.bisect_right(kf_frames, frame) - 1
        return kf_frames[i] if i >= 0 else None

    prev_end = 0
    for rng_start, rng_end in render_ranges:
        if rng_start > prev_end:
            pass_start = prev_end
            pass_end = rng_start - 1
            if kf_frames:
                snapped_start = _snap_forward(pass_start)
                snapped_end_kf = _snap_backward(pass_end)
                if (snapped_start is not None and snapped_end_kf is not None
                        and snapped_start < snapped_end_kf):
                    if snapped_start > pass_start:
                        seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}_active.mp4")
                        segments.append((seg_path, "active", pass_start, snapped_start - 1))
                        seg_idx += 1
                    seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}_pass.mp4")
                    segments.append((seg_path, "passthrough", snapped_start, snapped_end_kf - 1))
                    seg_idx += 1
                    if snapped_end_kf <= pass_end:
                        seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}_active.mp4")
                        segments.append((seg_path, "active", snapped_end_kf, pass_end))
                        seg_idx += 1
                else:
                    seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}_active.mp4")
                    segments.append((seg_path, "active", pass_start, pass_end))
                    seg_idx += 1
            else:
                seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}_active.mp4")
                segments.append((seg_path, "active", pass_start, pass_end))
                seg_idx += 1

        seg_p = p_curve[rng_start:rng_end + 1]
        seg_blur = blur_strength[rng_start:rng_end + 1]
        seg_whip = whip_strength[rng_start:rng_end + 1]
        is_hold = (seg_p > 0.999) & (seg_blur < 0.001) & (seg_whip < 0.001)

        hold_runs = []
        if not overlay:
            in_run = False
            run_start = 0
            for li in range(len(is_hold)):
                if is_hold[li] and not in_run:
                    run_start = li
                    in_run = True
                elif not is_hold[li] and in_run:
                    run_end = li - 1
                    if run_end - run_start + 1 > min_hold_frames:
                        abs_s = rng_start + run_start
                        abs_e = rng_start + run_end
                        run_z = zooms[abs_s:abs_e + 1]
                        if float(run_z.max() - run_z.min()) < 0.01:
                            hold_runs.append((run_start, run_end))
                    in_run = False
            if in_run:
                run_end = len(is_hold) - 1
                if run_end - run_start + 1 > min_hold_frames:
                    abs_s = rng_start + run_start
                    abs_e = rng_start + run_end
                    run_z = zooms[abs_s:abs_e + 1]
                    if float(run_z.max() - run_z.min()) < 0.01:
                        hold_runs.append((run_start, run_end))

        if hold_runs:
            cursor = 0
            for hold_start, hold_end in hold_runs:
                if hold_start > cursor:
                    abs_s = rng_start + cursor
                    abs_e = rng_start + hold_start - 1
                    seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}_active.mp4")
                    segments.append((seg_path, "active", abs_s, abs_e))
                    seg_idx += 1
                abs_s = rng_start + hold_start
                abs_e = rng_start + hold_end
                seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}_active.mp4")
                segments.append((seg_path, "active", abs_s, abs_e))
                seg_idx += 1
                cursor = hold_end + 1
            if cursor <= rng_end - rng_start:
                abs_s = rng_start + cursor
                seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}_active.mp4")
                segments.append((seg_path, "active", abs_s, rng_end))
                seg_idx += 1
            prev_end = rng_end + 1
            continue

        seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}_active.mp4")
        segments.append((seg_path, "active", rng_start, rng_end))
        seg_idx += 1
        prev_end = rng_end + 1

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
            seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}_active.mp4")
            segments.append((seg_path, "active", pass_start, pass_end))
            seg_idx += 1

    n_pass = sum(1 for _, t, *_ in segments if t == "passthrough")
    n_active = sum(1 for _, t, *_ in segments if t == "active")
    pass_frames = sum(fe - fs + 1 for _, t, fs, fe in segments if t == "passthrough")
    codec_label = "GPU"
    print(f"   Segment pipeline: {len(segments)} segments ({n_active} active [{codec_label}], {n_pass} passthrough [{pass_frames} frames stream-copy])")

    t0 = time.monotonic()

    pass_segs = [(s, fs, fe) for s, typ, fs, fe in segments if typ == "passthrough"]
    enc_codec = enc.split("_")[0] if "_" in enc else enc
    need_reencode = src_codec != enc_codec
    if pass_segs:
        total_pass_frames = sum(fe - fs + 1 for _, fs, fe in pass_segs)
        def _extract(args):
            path, fs, fe = args
            _extract_passthrough(input_path, path, fs / fps, (fe + 1) / fps, enc,
                                 reencode=need_reencode)
        with ThreadPoolExecutor(max_workers=min(len(pass_segs), 4)) as pool_ex:
            list(pool_ex.map(_extract, pass_segs))
        mode = "re-encoded" if need_reencode else "stream-copied"
        print(f"   Passthrough segments: {len(pass_segs)} {mode} ({total_pass_frames} frames) in {time.monotonic() - t0:.1f}s")

    t1 = time.monotonic()
    active_segs = [(s, fs, fe) for s, typ, fs, fe in segments if typ == "active"]
    total_active_frames = sum(fe - fs + 1 for _, fs, fe in active_segs)

    for si, (path, fs, fe) in enumerate(active_segs):
        n_seg = fe - fs + 1
        print(f"   Rendering segment {si+1}/{len(active_segs)}: frames {fs}-{fe} ({n_seg} frames) [{codec_label}]", flush=True)
        _render_active_segment_gpu(
            input_path, path, fs, fe,
            face_data, face_data_stable, p_curve, zooms,
            blur_strength, blur_n_samples, whip_strength, whip_direction,
            times, overlay, overlay_config, face_side, dest_x_full,
            stabilize, debug_labels, fps, w, h, enc,
        )

    elapsed_render = time.monotonic() - t1
    print(f"   Active segments: {len(active_segs)} rendered ({total_active_frames} frames) in {elapsed_render:.1f}s ({total_active_frames / max(elapsed_render, 0.01):.1f} fps) [{codec_label}]")

    segment_paths = [s for s, *_ in segments]
    tmp_concat = os.path.join(tmp_dir, "concat_silent.mp4")
    _concat_segments(segment_paths, tmp_concat)

    print("3. Muxing audio ...")
    mux_audio(input_path, tmp_concat, output_path)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    total = time.monotonic() - t0
    print(f"   Total segment pipeline: {total:.1f}s")
    print(f"Done -> {output_path}")


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
    GPU-accelerated zoom bounce effect.
    Same signature as zoom_bounce.create_zoom_bounce_effect.
    Falls back to CPU path for stabilize != 0.
    """
    if bounces is None:
        bounces = [(1.0, 2.5)]
    if overlay_config is None and text_config is not None:
        overlay_config = text_config
    if bounce_mode not in EASE_FUNCTIONS:
        raise ValueError(
            f"Unknown bounce_mode: {bounce_mode!r}. Use: {list(EASE_FUNCTIONS)}"
        )

    if stabilize != 0:
        from zoom_bounce import create_zoom_bounce_effect as cpu_effect
        return cpu_effect(
            input_path, output_path, zoom_max=zoom_max, bounces=bounces,
            bounce_mode=bounce_mode, face_side=face_side,
            overlay_config=overlay_config, text_config=text_config,
            fade_mode=fade_mode, stabilize=stabilize,
            stabilize_alpha=stabilize_alpha, debug_labels=debug_labels,
            detect_holds=detect_holds,
        )

    print("GPU pipeline: CuPy + ffmpeg pipe decode + h264_nvenc")

    print("1. Analyzing face trajectory ...")
    active_ranges = None
    probe_cap = cv2.VideoCapture(input_path)
    probe_fps = probe_cap.get(cv2.CAP_PROP_FPS)
    probe_n = int(probe_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    probe_cap.release()
    active_ranges = _compute_active_frame_ranges(bounces, probe_fps, probe_n,
                                                   detect_holds=detect_holds)
    if active_ranges is not None:
        detect_frames = sum(e - s + 1 for s, e in active_ranges)
        print(f"   Selective detection: {detect_frames}/{probe_n} frames ({100*detect_frames/max(probe_n,1):.0f}%)")
        raw_data, fps, (w, h) = get_face_data_seek(input_path, active_ranges, probe_n)
    else:
        probe_cap = cv2.VideoCapture(input_path)
        fps = probe_cap.get(cv2.CAP_PROP_FPS)
        w, h = int(probe_cap.get(3)), int(probe_cap.get(4))
        probe_n = int(probe_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        probe_cap.release()
        default = (w // 2, h // 2, 100, 100)
        raw_data = [default] * probe_n
        print("   No face-dependent events — skipping detection")

    face_data = smooth_data(raw_data, alpha=0.05)
    n_frames = len(face_data)
    face_data_stable = smooth_data(raw_data, alpha=stabilize_alpha)

    times, p_curve, zooms = build_bounce_curves(
        n_frames, fps, bounces, bounce_mode, zoom_max
    )
    blur_strength, blur_n_samples, whip_strength, whip_direction = build_effect_curves(
        n_frames, fps, bounces, bounce_mode, zoom_max
    )

    render_ranges = _compute_render_ranges(bounces, fps, n_frames)
    if render_ranges is None:
        print("   No render ranges — nothing to do")
        return

    src_codec = _probe_source_codec(input_path)

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

    enc = detect_best_encoder(src_codec)
    _HW_SUFFIXES = ("_nvenc", "_videotoolbox", "_qsv")
    if not any(enc.endswith(s) for s in _HW_SUFFIXES):
        enc = detect_best_encoder("h264")

    render_frames = sum(e - s + 1 for s, e in render_ranges)
    print(f"   Render ranges: {len(render_ranges)} range(s), {render_frames}/{n_frames} frames ({100*render_frames/max(n_frames,1):.0f}%)")
    print(f"2. Segment pipeline ({bounce_mode} mode, {len(bounces)} bounce(s)) [GPU] ...")
    _run_segment_pipeline_gpu(
        input_path, output_path, render_ranges, n_frames, fps,
        face_data, face_data_stable, p_curve, zooms,
        blur_strength, blur_n_samples, whip_strength, whip_direction,
        times, overlay, overlay_config, face_side, dest_x_full,
        stabilize, debug_labels, w, h, enc, src_codec=src_codec,
    )
