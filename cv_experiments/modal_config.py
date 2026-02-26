"""
Shared Modal configuration for zoom-bounce pipelines.

Single source of truth for image definition, GPU tiers, timeouts, and volume.
Both modal_prod.py and modal_dev.py import from here — same cached layers, no double-build.
"""

import modal

# ---------------------------------------------------------------------------
# GPU tiers
# ---------------------------------------------------------------------------
GPU_PROD = "L40S"
GPU_DEV = "L4"

# ---------------------------------------------------------------------------
# Timeouts & scaling
# ---------------------------------------------------------------------------
TIMEOUT_PROD = 1800          # 30 min
TIMEOUT_DEV = 900            # 15 min
SCALEDOWN_PROD = 300         # 5 min warm pool
SCALEDOWN_DEV = 60           # spin down fast, save money

# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------
volume = modal.Volume.from_name("zoom-bounce-videos", create_if_missing=True)
VOLUME_MOUNT = "/data"
INPUT_DIR = f"{VOLUME_MOUNT}/input"
OUTPUT_DIR = f"{VOLUME_MOUNT}/output"

# ---------------------------------------------------------------------------
# Base image — shared layers (no source code baked in)
# Prod will mount S3; dev adds local dir on top.
# ---------------------------------------------------------------------------
base_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.10"
    )
    .env({"NVIDIA_DRIVER_CAPABILITIES": "compute,video,utility"})
    .apt_install("libgl1", "libglib2.0-0", "wget", "xz-utils")
    .run_commands(
        # BtbN static ffmpeg with NVENC support (GPL build)
        "wget -q https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
        "ffmpeg-master-latest-linux64-gpl.tar.xz -O /tmp/ff.tar.xz"
        " && tar xf /tmp/ff.tar.xz -C /opt"
        " && ln -sf /opt/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg /usr/local/bin/ffmpeg"
        " && ln -sf /opt/ffmpeg-master-latest-linux64-gpl/bin/ffprobe /usr/local/bin/ffprobe"
        " && rm /tmp/ff.tar.xz",
    )
    .pip_install(
        "opencv-python-headless", "numpy", "mediapipe",
        "moviepy==1.0.3", "cupy-cuda12x", "PyNvVideoCodec",
    )
)

# Paths for local source mounting (dev only)
LOCAL_SRC_DIR = "/Users/sidhant/sidhant-experiments/cv_experiments"
REMOTE_SRC_DIR = "/root/cv_experiments"
