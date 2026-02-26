"""
Production deployment for zoom-bounce processing.

Deploy:   modal deploy modal_prod.py
Test:     curl -X POST https://<app-url>/process -d '{"input_filename": "clip.mp4", ...}'

Features:
  - Class-based with lifecycle hooks (pre-import heavy libs once)
  - L40S GPU, 30-min timeout, retries with backoff
  - Three HTTP endpoints: /process (sync), /submit (async), /status (poll)
  - TODO: swap add_local_dir for S3 mount when ready
"""

from __future__ import annotations

import uuid
from typing import Any

import modal
from pydantic import BaseModel, Field

from modal_config import (
    GPU_PROD,
    LOCAL_SRC_DIR,
    REMOTE_SRC_DIR,
    SCALEDOWN_PROD,
    TIMEOUT_PROD,
    base_image,
)

# Prod image: base + local source code
# TODO: replace add_local_dir with S3 mount when ready
image = (
    base_image
    .add_local_dir(LOCAL_SRC_DIR, remote_path=REMOTE_SRC_DIR)
    .add_local_file(LOCAL_SRC_DIR + "/modal_config.py", remote_path="/root/modal_config.py")
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = modal.App("zoom-bounce-prod")

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ProcessRequest(BaseModel):
    input_path: str | None = Field(None, description="Absolute path to the input video inside the container")
    input_url: str | None = Field(None, description="URL to download the input video from (e.g. presigned S3 URL)")
    output_filename: str | None = Field(None, description="Output filename (auto-generated if omitted)")
    kwargs: dict[str, Any] = Field(default_factory=dict, description="Extra kwargs for create_zoom_bounce_effect")


class ProcessResponse(BaseModel):
    output_filename: str
    status: str = "completed"


class JobSubmitResponse(BaseModel):
    call_id: str
    status: str = "submitted"


class JobStatusResponse(BaseModel):
    call_id: str
    status: str
    output_filename: str | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Processor class
# ---------------------------------------------------------------------------
@app.cls(
    image=image,
    gpu=GPU_PROD,
    timeout=TIMEOUT_PROD,
    scaledown_window=SCALEDOWN_PROD,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=5.0),
)
class ZoomBounceProcessor:
    @modal.enter()
    def setup(self):
        """Pre-import heavy libraries once per container start."""
        import sys
        sys.path.insert(0, "/root/cv_experiments")

        import cv2  # noqa: F401
        import cupy  # noqa: F401
        import mediapipe  # noqa: F401
        import numpy  # noqa: F401

        from zoom_bounce_gpu import create_zoom_bounce_effect
        self._effect_fn = create_zoom_bounce_effect

    @modal.method()
    def process(self, input_bytes: bytes, output_filename: str, **kwargs) -> bytes:
        """Process video bytes and return result bytes."""
        input_path = f"/tmp/input_{output_filename}"
        output_path = f"/tmp/{output_filename}"

        with open(input_path, "wb") as f:
            f.write(input_bytes)

        self._effect_fn(input_path, output_path, **kwargs)

        with open(output_path, "rb") as f:
            return f.read()


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------
def _resolve_input_bytes(req: ProcessRequest) -> tuple[bytes | None, str | None]:
    """Resolve input bytes from either input_path or input_url. Returns (bytes, error)."""
    import os

    if req.input_url:
        import urllib.request
        try:
            resp = urllib.request.urlopen(req.input_url, timeout=120)
            return resp.read(), None
        except Exception as e:
            return None, f"Failed to download from URL: {e}"
    elif req.input_path:
        if not os.path.exists(req.input_path):
            return None, f"File not found: {req.input_path}"
        with open(req.input_path, "rb") as f:
            return f.read(), None
    else:
        return None, "Provide either input_path or input_url"


@app.function(image=image, timeout=TIMEOUT_PROD)
@modal.fastapi_endpoint(method="POST", docs=True)
def process_sync(req: ProcessRequest):
    """Synchronous processing — blocks until done. Returns the MP4 file directly."""
    import io

    from fastapi.responses import JSONResponse, StreamingResponse

    output_filename = req.output_filename or f"{uuid.uuid4().hex[:12]}.mp4"

    input_bytes, error = _resolve_input_bytes(req)
    if error:
        return JSONResponse(status_code=400, content={"error": error})

    processor = ZoomBounceProcessor()
    result_bytes = processor.process.remote(
        input_bytes=input_bytes,
        output_filename=output_filename,
        **req.kwargs,
    )

    return StreamingResponse(
        io.BytesIO(result_bytes),
        media_type="video/mp4",
        headers={"Content-Disposition": f'attachment; filename="{output_filename}"'},
    )


@app.function(image=image, timeout=60)
@modal.fastapi_endpoint(method="POST", docs=True)
def submit_async(req: ProcessRequest) -> JobSubmitResponse:
    """Async submission — returns call_id immediately. Poll /status for result."""
    from fastapi.responses import JSONResponse

    output_filename = req.output_filename or f"{uuid.uuid4().hex[:12]}.mp4"

    input_bytes, error = _resolve_input_bytes(req)
    if error:
        return JSONResponse(status_code=400, content={"error": error})

    processor = ZoomBounceProcessor()
    call = processor.process.spawn(
        input_bytes=input_bytes,
        output_filename=output_filename,
        **req.kwargs,
    )
    return JobSubmitResponse(call_id=call.object_id)


@app.function(image=image, timeout=60)
@modal.fastapi_endpoint(method="GET", docs=True)
def job_status(call_id: str) -> JobStatusResponse:
    """Poll job status by call_id returned from /submit."""
    from modal.functions import FunctionCall

    fc = FunctionCall.from_id(call_id)
    try:
        result = fc.get(timeout=0)
        return JobStatusResponse(call_id=call_id, status="completed", output_filename="done")
    except TimeoutError:
        return JobStatusResponse(call_id=call_id, status="pending")
    except Exception as e:
        return JobStatusResponse(call_id=call_id, status="failed", error=str(e))


@app.function(image=image, timeout=TIMEOUT_PROD)
@modal.fastapi_endpoint(method="GET", docs=True)
def job_result(call_id: str, filename: str = "output.mp4"):
    """Download the result video for a completed job. Blocks until done."""
    import io

    from fastapi.responses import JSONResponse, StreamingResponse
    from modal.functions import FunctionCall

    fc = FunctionCall.from_id(call_id)
    try:
        result_bytes = fc.get(timeout=TIMEOUT_PROD)
    except TimeoutError:
        return JSONResponse(status_code=202, content={"status": "pending", "call_id": call_id})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "failed", "error": str(e)})

    return StreamingResponse(
        io.BytesIO(result_bytes),
        media_type="video/mp4",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# Local entrypoint — run from CLI with local files
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(input_file: str, output_file: str = None, debug_labels: bool = True):
    """Run a local video file through the deployed prod processor.

    Usage: modal run modal_prod.py --input-file path/to/vid.mp4
    """
    import os
    import time

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    stem = os.path.splitext(os.path.basename(input_file))[0]
    ts = int(time.time())
    output_filename = output_file or f"{stem}_prod_{ts}.mp4"

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    print(f"Input:  {input_file} ({os.path.getsize(input_file) / 1e6:.1f} MB)")
    print(f"Output: {output_path}")
    print(f"GPU:    {GPU_PROD}")

    with open(input_file, "rb") as f:
        input_bytes = f.read()

    processor = ZoomBounceProcessor()
    result_bytes = processor.process.remote(
        input_bytes=input_bytes,
        output_filename=output_filename,
        debug_labels=debug_labels,
        stabilize=0,
        bounces=[
            {"action": "in", "start": 1.2, "end": 1.6, "ease": "snap", "zoom": 1.35},
            {"action": "out", "start": 3.1, "end": 3.5, "ease": "smooth"},
        ],
    )

    with open(output_path, "wb") as f:
        f.write(result_bytes)
    print(f"Done! Saved {output_path}")


# ---------------------------------------------------------------------------
# External trigger helper
# ---------------------------------------------------------------------------
def get_remote_processor():
    """Get a handle to the deployed processor from any Python env with Modal auth.

    Usage:
        from modal_prod import get_remote_processor
        processor = get_remote_processor()
        result = processor.process.remote(input_bytes=b"...", output_filename="out.mp4")
    """
    return modal.Cls.from_name("zoom-bounce-prod", "ZoomBounceProcessor")()
