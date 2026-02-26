"""
Fast-iteration dev runner for zoom-bounce on Modal.

Usage:
  modal serve modal_dev.py                              # hot-reload
  modal run modal_dev.py --case vid_short_mix            # one case
  modal run modal_dev.py --all                           # all cases
  modal run modal_dev.py --case really_big_vid --gpu l40s # big video on L40S
  modal run modal_dev.py --case vid_short_mix --no-debug  # without debug labels

Differences from prod:
  - Function-based (no class overhead)
  - L4 GPU by default (cheaper), override with --gpu l40s
  - No retries — fail fast, see tracebacks immediately
  - 60s scaledown — spin down quick, save money
  - debug_labels=True forced by default
  - Raw bytes I/O — same as modal_use.py, no volume setup needed
"""

from __future__ import annotations

import modal

from modal_config import (
    GPU_DEV,
    GPU_PROD,
    LOCAL_SRC_DIR,
    REMOTE_SRC_DIR,
    SCALEDOWN_DEV,
    TIMEOUT_DEV,
    base_image,
)

# Dev image: base + local source code baked in + config on sys.path
image = (
    base_image
    .add_local_dir(LOCAL_SRC_DIR, remote_path=REMOTE_SRC_DIR)
    .add_local_file(LOCAL_SRC_DIR + "/modal_config.py", remote_path="/root/modal_config.py")
)

app = modal.App("zoom-bounce-dev")

# ---------------------------------------------------------------------------
# GPU lookup for CLI flag
# ---------------------------------------------------------------------------
_GPU_MAP = {
    "l4": GPU_DEV,
    "l40s": GPU_PROD,
}


def _resolve_gpu(gpu_flag: str | None) -> str:
    if gpu_flag is None:
        return GPU_DEV
    key = gpu_flag.lower()
    if key not in _GPU_MAP:
        raise ValueError(f"Unknown GPU: {gpu_flag!r}. Choose from: {list(_GPU_MAP)}")
    return _GPU_MAP[key]


# ---------------------------------------------------------------------------
# Remote function — A10G default, no retries, fast scaledown
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu=GPU_DEV,
    timeout=TIMEOUT_DEV,
    scaledown_window=SCALEDOWN_DEV,
)
def run_zoom_bounce_dev(input_bytes: bytes, output_filename: str, **kwargs):
    """Process a video on Modal. Same bytes-in/bytes-out contract as modal_use.py."""
    import sys

    sys.path.insert(0, "/root/cv_experiments")
    from zoom_bounce_gpu import create_zoom_bounce_effect

    input_path = f"/tmp/input_{output_filename}"
    output_path = f"/tmp/{output_filename}"

    with open(input_path, "wb") as f:
        f.write(input_bytes)

    create_zoom_bounce_effect(input_path, output_path, **kwargs)

    with open(output_path, "rb") as f:
        return f.read()


# We also define an L40S variant so the CLI can pick at runtime
@app.function(
    image=image,
    gpu=GPU_PROD,
    timeout=TIMEOUT_DEV,
    scaledown_window=SCALEDOWN_DEV,
)
def run_zoom_bounce_dev_l40s(input_bytes: bytes, output_filename: str, **kwargs):
    """Same as run_zoom_bounce_dev but on L40S GPU."""
    import sys

    sys.path.insert(0, "/root/cv_experiments")
    from zoom_bounce_gpu import create_zoom_bounce_effect

    input_path = f"/tmp/input_{output_filename}"
    output_path = f"/tmp/{output_filename}"

    with open(input_path, "wb") as f:
        f.write(input_bytes)

    create_zoom_bounce_effect(input_path, output_path, **kwargs)

    with open(output_path, "rb") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Local entrypoint — test runner
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    case: str = None,
    all: bool = False,
    skip_missing: bool = True,
    gpu: str = None,
    no_debug: bool = False,
):
    import os
    import sys
    import time

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from zoom_bounce_test_runner import TEST_CASES

    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "inputs")
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Select test cases
    if all:
        selected = TEST_CASES
    elif case:
        selected = [c for c in TEST_CASES if c["name"] == case]
        if not selected:
            names = [c["name"] for c in TEST_CASES]
            raise ValueError(f"Unknown case: {case!r}. Available: {names}")
    else:
        raise ValueError("Pass --case <name> or --all")

    # Pick the right remote function based on GPU flag
    resolved_gpu = _resolve_gpu(gpu)
    if resolved_gpu == GPU_PROD:
        remote_fn = run_zoom_bounce_dev_l40s
        gpu_label = "L40S"
    else:
        remote_fn = run_zoom_bounce_dev
        gpu_label = "L4"

    ts = int(time.time())
    debug_labels = not no_debug
    print(f"Running {len(selected)} case(s) on Modal [dev, {gpu_label}, debug={debug_labels}]")

    for tc in selected:
        input_path = os.path.join(input_dir, tc["input_path"])
        if not os.path.exists(input_path):
            msg = f"Input missing for case '{tc['name']}': {input_path}"
            if skip_missing:
                print(f"SKIP: {msg}")
                continue
            raise FileNotFoundError(msg)

        stem = os.path.splitext(tc["input_path"])[0]
        output_name = tc["output_template"].format(stem=stem, ts=ts)
        output_path = os.path.join(output_dir, output_name)

        print(f"\n=== {tc['name']} ===")
        print(f"Input:  {input_path} ({os.path.getsize(input_path) / 1e6:.1f} MB)")
        print(f"Output: {output_path}")

        with open(input_path, "rb") as f:
            video_bytes = f.read()

        # Merge kwargs with debug_labels override
        kwargs = {**tc["kwargs"], "debug_labels": debug_labels}

        result_bytes = remote_fn.remote(
            input_bytes=video_bytes,
            output_filename=output_name,
            **kwargs,
        )

        with open(output_path, "wb") as f:
            f.write(result_bytes)
        print(f"Done! Saved {output_path}")
