import modal

from modal_config import GPU_PROD, LOCAL_SRC_DIR, REMOTE_SRC_DIR, TIMEOUT_PROD, base_image

app = modal.App("zoom-bounce")

image = base_image.add_local_dir(LOCAL_SRC_DIR, remote_path=REMOTE_SRC_DIR)


@app.function(
    image=image,
    gpu=GPU_PROD,
    timeout=TIMEOUT_PROD,
)
def run_zoom_bounce(input_bytes: bytes, output_filename: str, **kwargs):
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


@app.local_entrypoint()
def main(case: str = None, all: bool = False, skip_missing: bool = True):
    import os
    import time
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from zoom_bounce_test_runner import TEST_CASES

    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "inputs")
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    if all:
        selected = TEST_CASES
    elif case:
        selected = [c for c in TEST_CASES if c["name"] == case]
        if not selected:
            names = [c["name"] for c in TEST_CASES]
            raise ValueError(f"Unknown case: {case!r}. Available: {names}")
    else:
        raise ValueError("Pass --case <name> or --all")

    ts = int(time.time())
    print(f"Running {len(selected)} case(s) on Modal [gpu]")

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

        result_bytes = run_zoom_bounce.remote(
            input_bytes=video_bytes,
            output_filename=output_name,
            **tc["kwargs"],
        )

        with open(output_path, "wb") as f:
            f.write(result_bytes)
        print(f"Done! Saved {output_path}")
