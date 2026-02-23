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
import cupy as cp

from zoom_bounce import (
    # Constants
    EDGE_STRIP_FRAC, FADE_WIDTH_FRAC,
    # Curve builders
    build_bounce_curves, build_effect_curves,
    # Face detection
    get_face_data, get_face_data_seek, smooth_data,
    # Segment management
    _compute_active_frame_ranges, _compute_render_ranges,
    _probe_source_codec, _probe_keyframe_times,
    _extract_passthrough, _render_hold_ffmpeg,
    _concat_segments,
    # FFmpeg
    open_ffmpeg_writer, mux_audio, detect_best_encoder,
    # Overlay
    create_overlay,
    # Easing
    EASE_FUNCTIONS,
    # Misc
    lerp,
)

# ─── Optional PyNvVideoCodec import ─────────────────────────────────────────

_HAS_NVCODEC = False
try:
    import PyNvVideoCodec as nvc
    _HAS_NVCODEC = True
except ImportError:
    pass


# ─── CUDA Kernels ────────────────────────────────────────────────────────────

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

    // Border replicate
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
    // Each thread handles one pixel for Y, and contributes to UV at 2x2 blocks
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int rgb_idx = (y * w + x) * 3;
    float r = (float)rgb[rgb_idx];
    float g = (float)rgb[rgb_idx + 1];
    float b = (float)rgb[rgb_idx + 2];

    // BT.601 coefficients (match ffmpeg default)
    float yf =  0.257f * r + 0.504f * g + 0.098f * b + 16.0f;
    y_plane[y * w + x] = (unsigned char)fminf(fmaxf(yf + 0.5f, 0.0f), 255.0f);

    // UV: only top-left pixel of each 2x2 block writes
    if ((x & 1) == 0 && (y & 1) == 0) {
        // Average the 2x2 block for chroma
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

        int uv_idx = (y / 2) * w + x;  // interleaved UV, w bytes per row
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
    int kernel_radius,  // half-size of box filter
    float strength,     // blend weight [0,1]
    int horizontal      // 1=horizontal, 0=vertical
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
        // Clamp to borders
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

    // Blend: lerp(original, blurred, strength)
    float orig_r = src[idx];
    float orig_g = src[idx + 1];
    float orig_b = src[idx + 2];

    dst[idx]     = (unsigned char)(orig_r + (br - orig_r) * strength + 0.5f);
    dst[idx + 1] = (unsigned char)(orig_g + (bg - orig_g) * strength + 0.5f);
    dst[idx + 2] = (unsigned char)(orig_b + (bb - orig_b) * strength + 0.5f);
}
''', 'directional_blur')

_BLOCK = (16, 16)


# ─── GPU Helper Functions ────────────────────────────────────────────────────

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


# ─── GPUBufferPool ───────────────────────────────────────────────────────────

class GPUBufferPool:
    """Pre-allocate all GPU buffers for active segment rendering.

    Eliminates transient allocations in the hot loop. Double-buffered NV12
    planes allow encode of frame N to overlap with compute of frame N+1.
    """

    def __init__(self, h, w, has_zoom_blur=False, has_whip=False, need_nv12=False):
        self.h = h
        self.w = w

        # Primary uint8 buffers
        self.rgb = cp.empty((h, w, 3), dtype=cp.uint8)
        self.warped = cp.empty((h, w, 3), dtype=cp.uint8)
        self.out = cp.empty((h, w, 3), dtype=cp.uint8)

        # Float32 scratch
        self.warped_f32 = cp.empty((h, w, 3), dtype=cp.float32)
        self.blend_f32 = cp.empty((h, w, 3), dtype=cp.float32)

        # Edge fade
        self.fade_alpha = cp.empty((h, w, 1), dtype=cp.float32)
        self.fade_bg = cp.empty((h, w, 3), dtype=cp.float32)
        self.inv_alpha = cp.empty((h, w, 1), dtype=cp.float32)

        # Zoom blur
        if has_zoom_blur:
            self.blur_accum = cp.empty((h, w, 3), dtype=cp.float32)
            self.blur_sample = cp.empty((h, w, 3), dtype=cp.uint8)
            self.blur_sample_f32 = cp.empty((h, w, 3), dtype=cp.float32)
        else:
            self.blur_accum = self.blur_sample = self.blur_sample_f32 = None

        # Whip
        if has_whip:
            self.whip_dst = cp.empty((h, w, 3), dtype=cp.uint8)
        else:
            self.whip_dst = None

        # Double-buffered NV12 for NVENC overlap
        # NV12 is a single contiguous buffer: h rows of Y + h/2 rows of interleaved UV
        if need_nv12:
            nv12_h = h + h // 2
            self.nv12_a = cp.empty((nv12_h, w), dtype=cp.uint8)
            self.nv12_b = cp.empty((nv12_h, w), dtype=cp.uint8)
        else:
            self.nv12_a = self.nv12_b = None

        self._nv12_ping = True  # toggle for double-buffering

    def get_nv12_buffers(self):
        """Return (nv12_full, y_plane, uv_plane) and toggle for next frame."""
        buf = self.nv12_a if self._nv12_ping else self.nv12_b
        self._nv12_ping = not self._nv12_ping
        y = buf[:self.h]
        uv = buf[self.h:]
        return buf, y, uv


# ─── GPU Effect Functions (zero-alloc) ───────────────────────────────────────

def _gpu_zoom_blur(pool, g_rgb, w, h, z, sx, sy, strength, n_samples):
    """GPU zoom blur using pre-allocated pool buffers. Zero transient allocs."""
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
        # Accumulate as float32 without .astype() allocation
        cp.copyto(pool.blur_sample_f32, pool.blur_sample, casting='unsafe')
        cp.add(pool.blur_accum, pool.blur_sample_f32, out=pool.blur_accum)

    pool.blur_accum /= n_samples

    # Blend: warped + (blur_accum - warped) * strength
    cp.copyto(pool.warped_f32, pool.warped, casting='unsafe')
    cp.subtract(pool.blur_accum, pool.warped_f32, out=pool.blend_f32)
    cp.multiply(pool.blend_f32, strength, out=pool.blend_f32)
    cp.add(pool.warped_f32, pool.blend_f32, out=pool.blend_f32)
    cp.clip(pool.blend_f32, 0, 255, out=pool.blend_f32)
    cp.copyto(pool.warped, pool.blend_f32, casting='unsafe')


def _gpu_whip(pool, w, h, strength, direction):
    """GPU directional blur via CUDA kernel. No CPU fallback."""
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

    # Extract edge band mean per row
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

    # Compute fade alpha
    cp.copyto(pool.warped_f32, pool.warped, casting='unsafe')
    cp.multiply(g_base_gradient_3ch, p, out=pool.fade_alpha)
    pool.fade_alpha += (1.0 - p)

    # Blend: warped * alpha + bg * (1-alpha)
    cp.multiply(pool.warped_f32, pool.fade_alpha, out=pool.blend_f32)
    cp.subtract(1.0, pool.fade_alpha, out=pool.inv_alpha)
    # fade_bg * inv_alpha -> warped_f32 (reuse as scratch since we already consumed it)
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


# ─── NV12 Frame wrapper for NVENC GPU input ─────────────────────────────────

class _NV12GPUFrame:
    """Wraps a contiguous CuPy NV12 buffer for NVENC GPU input.

    The encoder with usecpuinputbuffer=False expects an object that either:
    - Has __cuda_array_interface__ (like a CuPy array), or
    - Has a .cuda() method returning [luma_cai, chroma_cai]

    We try both approaches.
    """

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
        # Encoder expects 3D arrays: Y as (h, w, 1), UV as (h/2, w/2, 2)
        return [self._y.reshape(self._height, self._width, 1),
                self._uv.reshape(self._height // 2, self._width // 2, 2)]


# ─── PyNvVideoCodec render path (zero-copy) ─────────────────────────────────

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
    seg_z = zooms[frame_start:frame_end + 1]
    n_seg = frame_end - frame_start + 1

    has_zoom_blur = seg_blur.max() > 0
    has_whip = seg_whip.max() > 0

    # Overlay config
    ovl_pos = overlay_config.get("position", "left") if overlay_config else "left"
    ovl_mg = overlay_config.get("margin", 1.8) if overlay_config else 1.8

    # Gradient fade setup
    need_fade = face_side != "center"
    edge_strip = max(int(w * EDGE_STRIP_FRAC), 1)
    fade_width = int(w * FADE_WIDTH_FRAC)

    # Pre-allocate all GPU buffers
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

    # Upload overlay once
    g_ovl_img = g_ovl_mask = None
    if overlay:
        oi, om = overlay.get_frame(0)
        g_ovl_img = cp.asarray(oi)
        g_ovl_mask = cp.asarray(om)

    # NVDEC decoder — outputs RGB directly on GPU
    decoder = nvc.SimpleDecoder(
        input_path, gpu_id=0,
        use_device_memory=True,
        output_color_type=nvc.OutputColorType.RGB,
    )

    # NVENC encoder — accepts GPU NV12 frames
    # Write raw H.264 bitstream to a temp file, wrap in MP4 after
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

    for idx in range(frame_start, frame_end + 1):
        p = float(p_curve[idx])
        z = float(zooms[idx])
        local = idx - frame_start

        # Decode frame from NVDEC (zero-copy GPU surface)
        surface = decoder[idx]
        g_frame = cp.from_dlpack(surface)

        # Handle shape: SimpleDecoder RGB output is (H, W, 3)
        if g_frame.shape != (h, w, 3):
            # Reshape if needed (e.g., planar format)
            if g_frame.ndim == 3 and g_frame.shape[0] == 3:
                g_frame = cp.transpose(g_frame, (1, 2, 0))

        # Passthrough: no effects
        if p < 0.001 and blur_strength[idx] < 0.001 and whip_strength[idx] < 0.001:
            cp.copyto(pool.rgb, g_frame)
            nv12_full, nv12_y, nv12_uv = pool.get_nv12_buffers()
            _gpu_rgb_to_nv12(pool.rgb, nv12_y, nv12_uv, w, h)
            frame_obj = _NV12GPUFrame(nv12_full, nv12_y, nv12_uv, w, h)
            bitstream = encoder.Encode(frame_obj)
            if bitstream:
                bitstream_file.write(bytes(bitstream))
            continue

        # Copy decoded frame into pool
        cp.copyto(pool.rgb, g_frame)

        t = times[idx]

        # Warp geometry
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
        sx_val = dx - tx * z
        sy_val = h / 2 - ty * z

        sfx = fx * z + sx_val
        sfy = fy * z + sy_val
        sfw = fw * z
        sfh = fh * z

        # GPU warp
        _gpu_warp(pool.rgb, pool.warped, w, h, z, sx_val, sy_val)

        # GPU zoom blur
        if has_zoom_blur and blur_strength[idx] > 0.001:
            _gpu_zoom_blur(
                pool, pool.rgb, w, h, z, sx_val, sy_val,
                float(blur_strength[idx]), int(blur_n_samples[idx]),
            )

        # GPU whip
        if has_whip and whip_strength[idx] > 0.001:
            _gpu_whip(pool, w, h, float(whip_strength[idx]), whip_direction[idx])

        # GPU edge fade
        if p < 0.001 or not need_fade:
            cp.copyto(pool.out, pool.warped)
        else:
            _gpu_edge_fade(pool, g_base_gradient_3ch, w, h,
                           edge_strip, face_side, p)

        # GPU overlay
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

        # Convert to NV12 and encode (stays on GPU)
        nv12_full, nv12_y, nv12_uv = pool.get_nv12_buffers()
        _gpu_rgb_to_nv12(pool.out, nv12_y, nv12_uv, w, h)
        frame_obj = _NV12GPUFrame(nv12_full, nv12_y, nv12_uv, w, h)
        bitstream = encoder.Encode(frame_obj)
        if bitstream:
            bitstream_file.write(bytes(bitstream))

        if local % 100 == 0:
            print(f"     frame {local}/{n_seg}", flush=True)

    # Flush encoder
    remaining = encoder.EndEncode()
    if remaining:
        bitstream_file.write(bytes(remaining))
    bitstream_file.close()

    # Wrap raw H.264 in MP4 container (stream-copy, very fast)
    subprocess.run(
        ["ffmpeg", "-y", "-i", raw_h264_path,
         "-c:v", "copy", "-an", output_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True,
    )
    os.remove(raw_h264_path)


# ─── Threaded Decode Prefetch (ffmpeg pipe) ──────────────────────────────────

class _ThreadedDecoder:
    """Decode frames via ffmpeg pipe in a background thread.

    ffmpeg handles any input codec (H.264, AV1, VP9, etc.) and outputs raw
    RGB24 frames.  The background thread prefetches into a small queue so
    decode of frame N+1 overlaps GPU compute of frame N.
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
        self._queue.put((False, None))  # sentinel

    def read(self):
        return self._queue.get()

    def release(self):
        self._proc.stdout.close()
        self._proc.terminate()
        self._proc.wait()


# ─── Fallback render path (cv2 + ffmpeg pipe) ───────────────────────────────

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

    # Detect hold sub-region
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

    # Pre-allocate all GPU buffers via pool (NV12 enabled for pipe output)
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

    for idx in range(frame_start, frame_end + 1):
        ok, rgb = decoder.read()
        if not ok:
            break

        p = float(p_curve[idx])
        z = float(zooms[idx])
        local = idx - frame_start

        if p < 0.001 and blur_strength[idx] < 0.001 and whip_strength[idx] < 0.001:
            # Passthrough: still need NV12 conversion for pipe format
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

        tx = lerp(w / 2, fx, p)
        ty = lerp(h / 2, fy, p)
        dx = lerp(w / 2, dest_x_full, p)
        sx_val = dx - tx * z
        sy_val = h / 2 - ty * z

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

        # Convert to NV12 on GPU, download smaller buffer (50% of RGB)
        nv12_full, nv12_y, nv12_uv = pool.get_nv12_buffers()
        _gpu_rgb_to_nv12(pool.out, nv12_y, nv12_uv, w, h)
        cp.asnumpy(nv12_full, out=buf_nv12_cpu)
        writer.stdin.write(buf_nv12_cpu.data)

        if local % 100 == 0:
            print(f"     frame {local}/{n_seg}", flush=True)

    decoder.release()
    writer.stdin.close()
    writer.wait()


# ─── Unified render dispatch ────────────────────────────────────────────────

# NOTE: The PyNvVideoCodec (NVDEC/NVENC) zero-copy path is disabled because
# SimpleDecoder random-access decoding causes frame corruption (bad keyframe
# seeking, timing issues). Instead we use cv2 decode + ffmpeg h264_nvenc pipe
# which the BtbN ffmpeg build supports. This gives us HW encode without the
# decode-side issues.

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


# ─── GPU segment pipeline (mirrors _run_segment_pipeline) ───────────────────


def _run_segment_pipeline_gpu(
    input_path, output_path, render_ranges, n_frames, fps,
    face_data, face_data_stable, p_curve, zooms,
    blur_strength, blur_n_samples, whip_strength, whip_direction,
    times, overlay, overlay_config, face_side, dest_x_full,
    stabilize, debug_labels, w, h, enc, src_codec="h264",
):
    """GPU version of _run_segment_pipeline — uses _render_active_segment_gpu."""
    import bisect
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
                    # Keyframes exist but snapping failed — still passthrough
                    seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}_pass.mp4")
                    segments.append((seg_path, "passthrough", pass_start, pass_end))
                    seg_idx += 1
            else:
                # No keyframe info — passthrough anyway, ffmpeg handles it
                seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}_pass.mp4")
                segments.append((seg_path, "passthrough", pass_start, pass_end))
                seg_idx += 1

        # Split active range into multiple hold sub-regions.
        # Find contiguous runs of hold frames (p≈1, no effects) with constant
        # zoom, and extract each as an ffmpeg crop+scale segment.
        seg_p = p_curve[rng_start:rng_end + 1]
        seg_blur = blur_strength[rng_start:rng_end + 1]
        seg_whip = whip_strength[rng_start:rng_end + 1]
        is_hold = (seg_p > 0.999) & (seg_blur < 0.001) & (seg_whip < 0.001)

        # Find contiguous hold runs and check each for constant zoom
        hold_runs = []  # list of (local_start, local_end) for valid holds
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
                        # Check constant zoom within this run
                        abs_s = rng_start + run_start
                        abs_e = rng_start + run_end
                        run_z = zooms[abs_s:abs_e + 1]
                        if float(run_z.max() - run_z.min()) < 0.01:
                            hold_runs.append((run_start, run_end))
                    in_run = False
            # Close final run
            if in_run:
                run_end = len(is_hold) - 1
                if run_end - run_start + 1 > min_hold_frames:
                    abs_s = rng_start + run_start
                    abs_e = rng_start + run_end
                    run_z = zooms[abs_s:abs_e + 1]
                    if float(run_z.max() - run_z.min()) < 0.01:
                        hold_runs.append((run_start, run_end))

        if hold_runs:
            # Emit active/hold segments in order
            cursor = 0  # local offset within this render range
            for hold_start, hold_end in hold_runs:
                # Active segment before this hold
                if hold_start > cursor:
                    abs_s = rng_start + cursor
                    abs_e = rng_start + hold_start - 1
                    seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}_active.mp4")
                    segments.append((seg_path, "active", abs_s, abs_e))
                    seg_idx += 1
                # Hold segment (rendered via ffmpeg crop+scale)
                abs_s = rng_start + hold_start
                abs_e = rng_start + hold_end
                seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}_active.mp4")
                segments.append((seg_path, "active", abs_s, abs_e))
                seg_idx += 1
                cursor = hold_end + 1
            # Trailing active after last hold
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
                # Snapping failed — still passthrough
                seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}_pass.mp4")
                segments.append((seg_path, "passthrough", pass_start, pass_end))
                seg_idx += 1
        else:
            seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}_pass.mp4")
            segments.append((seg_path, "passthrough", pass_start, pass_end))
            seg_idx += 1

    n_pass = sum(1 for _, t, *_ in segments if t == "passthrough")
    n_active = sum(1 for _, t, *_ in segments if t == "active")
    pass_frames = sum(fe - fs + 1 for _, t, fs, fe in segments if t == "passthrough")
    codec_label = "GPU"
    print(f"   Segment pipeline: {len(segments)} segments ({n_active} active [{codec_label}], {n_pass} passthrough [{pass_frames} frames stream-copy])")

    t0 = time.monotonic()

    # Extract passthrough segments in parallel
    pass_segs = [(s, fs, fe) for s, typ, fs, fe in segments if typ == "passthrough"]
    # Determine encoder codec family (e.g. "h264_nvenc" -> "h264")
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

    # Render active segments with GPU
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

    # Concat all segments
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


# ─── Main entry point ────────────────────────────────────────────────────────


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

    if face_side == "center":
        dest_x_full = w * 0.5
    elif face_side == "left":
        dest_x_full = w * 0.28
    else:
        dest_x_full = w * 0.72

    # Try to match source codec (enables stream-copy passthrough). Fall back
    # to H.264 if no HW encoder is available for the source codec.
    enc = detect_best_encoder(src_codec)
    _HW_SUFFIXES = ("_nvenc", "_videotoolbox", "_qsv")
    if not any(enc.endswith(s) for s in _HW_SUFFIXES):
        # Software encoder for source codec — fall back to H.264 HW
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
