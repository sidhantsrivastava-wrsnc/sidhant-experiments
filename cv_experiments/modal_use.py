import modal

app = modal.App("zoom-bounce")

image = (
    modal.Image.from_registry("nvidia/cuda:12.2.0-runtime-ubuntu22.04", add_python="3.10")
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
    .add_local_dir(
        "/Users/sidhant/sidhant-experiments/cv_experiments",
        remote_path="/root/cv_experiments",
    )
)


@app.function(
    image=image,
    gpu="L40S",
    timeout=1800,
)
def run_zoom_bounce(input_filename: str, output_filename: str, **kwargs):
    import sys
    sys.path.insert(0, "/root/cv_experiments")
    from zoom_bounce_gpu import create_zoom_bounce_effect

    input_path = f"/root/cv_experiments/{input_filename}"
    output_path = f"/tmp/{output_filename}"

    create_zoom_bounce_effect(input_path, output_path, **kwargs)

    with open(output_path, "rb") as f:
        return f.read()


@app.local_entrypoint()
def main():
    result_bytes = run_zoom_bounce.remote(
        input_filename="extremelylongvid.mp4",
        output_filename="extremelylongvid_gpu_output.mp4",
        stabilize=0,
        debug_labels=True,
        bounces=[
            {"action": "in", "start": 5.0, "end": 5.5, "ease": "smooth", "zoom": 1.35},
            {"action": "out", "start": 15.0, "end": 15.5, "ease": "snap"},
            {"action": "bounce", "start": 30.0, "end": 31.0, "ease": "overshoot", "zoom": 1.32},
            {"action": "zoom_blur", "start": 50.0, "end": 50.7, "intensity": 0.95, "n_samples": 7},
            {"action": "whip", "start": 75.0, "end": 75.4, "direction": "v", "intensity": 0.9},
            {"action": "in", "start": 100.0, "end": 100.5, "ease": "snap", "zoom": 1.38},
            {"action": "bounce", "start": 150.0, "end": 151.0, "ease": "smooth", "zoom": 1.28},
            {"action": "out", "start": 200.0, "end": 200.5, "ease": "smooth"},
            {"action": "in", "start": 300.0, "end": 300.5, "ease": "snap", "zoom": 1.4},
            {"action": "bounce", "start": 400.0, "end": 401.0, "ease": "overshoot", "zoom": 1.3},
            {"action": "whip", "start": 500.0, "end": 500.4, "direction": "h", "intensity": 0.85},
            {"action": "out", "start": 600.0, "end": 600.5, "ease": "smooth"},
            {"action": "in", "start": 800.0, "end": 800.5, "ease": "smooth", "zoom": 1.35},
            {"action": "bounce", "start": 1000.0, "end": 1001.0, "ease": "snap", "zoom": 1.32},
            {"action": "out", "start": 1200.0, "end": 1200.5, "ease": "smooth"},
        ],
    )

    with open("extremelylongvid_gpu_output.mp4", "wb") as f:
        f.write(result_bytes)
    print("Done! Saved extremelylongvid_gpu_output.mp4")
