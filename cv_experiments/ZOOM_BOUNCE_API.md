# Zoom Bounce API — Microservice Documentation

## Overview

GPU-accelerated video effects processor running on Modal. Supports two GPU tiers — **standard** (NVIDIA L4) and **premium** (NVIDIA L40S) — with automatic routing based on video size and fallback between tiers.

**Base URL:** `https://writesonic--zoom-bounce-prod-{endpoint}.modal.run`

**Deployed via:** `modal deploy modal_prod.py`

---

## Architecture

```
                        ┌──────────────────────────────────────────────────┐
                        │                Modal Cloud                       │
                        │                                                  │
┌────────┐   POST       │  ┌─────────────┐      ┌───────────────────────┐ │
│        │─────────────>│  │ Web Endpoint │      │ GPU Containers        │ │
│ Caller │   JSON +     │  │ (lightweight)│──┬──>│ L4  (standard tier)   │ │
│        │   video URL  │  │              │  │   │                       │ │
│        │              │  │ Downloads    │  │   ├───────────────────────┤ │
│        │<─────────────│  │ video, routes│  └──>│ L40S (premium tier)   │ │
│        │   MP4 stream │  │ to GPU tier  │      │                       │ │
└────────┘              │  └─────────────┘      │ zoom_bounce_gpu.py    │ │
                        │                        │ cupy + opencv +       │ │
                        │                        │ mediapipe + h264_nvenc│ │
                        │                        └───────────────────────┘ │
                        └──────────────────────────────────────────────────┘
```

### GPU Tier Routing

| Tier | GPU | When used |
|------|-----|-----------|
| `standard` | NVIDIA L4 | Videos < 50 MB (default) |
| `premium` | NVIDIA L40S | Videos >= 50 MB, or explicitly requested |

**Routing priority:** explicit `gpu_tier` field > auto-route by video size > default (`standard`).

**Fallback:** If the selected tier fails (capacity/timeout), the system automatically tries the next tier in order: `standard` → `premium`.

---

## Endpoints

### 1. `POST /process` — Synchronous (blocks until done)

Best for short videos (<30s). Blocks until processing finishes, then streams the MP4 back.

```bash
curl -X POST https://writesonic--zoom-bounce-prod-process-sync.modal.run \
  -H "Content-Type: application/json" \
  -o output.mp4 \
  -d '{
    "input_url": "https://your-s3-presigned-url...",
    "gpu_tier": "premium",
    "kwargs": {
      "stabilize": 0,
      "debug_labels": false,
      "bounces": [
        {"action": "in", "start": 1.2, "end": 1.6, "ease": "snap", "zoom": 1.35},
        {"action": "out", "start": 3.1, "end": 3.5, "ease": "smooth"}
      ]
    }
  }'
```

**Response:** Raw MP4 file stream (save with `-o output.mp4`). The `X-GPU-Tier` response header indicates which tier processed the video.

---

### 2. `POST /submit` — Async (returns immediately)

Best for long videos. Returns a `call_id` you can poll or use to download the result later.

```bash
curl -s -X POST https://writesonic--zoom-bounce-prod-submit-async.modal.run \
  -H "Content-Type: application/json" \
  -d '{
    "input_url": "https://your-s3-presigned-url...",
    "kwargs": {
      "stabilize": 0,
      "bounces": [
        {"action": "bounce", "start": 5.0, "end": 6.5, "ease": "smooth", "zoom": 1.3}
      ]
    }
  }'
```

**Response:**
```json
{"call_id": "fc-01KJCB98E0D9B0J7K8JZNK7HVH", "status": "submitted", "gpu_tier": "standard"}
```

---

### 3. `GET /status` — Poll job status

```bash
curl -s "https://writesonic--zoom-bounce-prod-job-status.modal.run?call_id=fc-01KJCB98E0D9B0J7K8JZNK7HVH"
```

**Response:**
```json
{"call_id": "fc-...", "status": "pending", "output_filename": null, "error": null}
{"call_id": "fc-...", "status": "completed", "output_filename": "done", "error": null}
{"call_id": "fc-...", "status": "failed", "output_filename": null, "error": "...message..."}
```

---

### 4. `GET /result` — Download result video

Blocks until the job finishes (up to 30 min), then streams the MP4 back. You can skip polling `/status` and call this directly after `/submit`.

```bash
curl -o output.mp4 "https://writesonic--zoom-bounce-prod-job-result.modal.run?call_id=fc-01KJCB98E0D9B0J7K8JZNK7HVH&filename=my_video.mp4"
```

**Response:** Raw MP4 file stream.

---

## Request Schema

### ProcessRequest

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `input_url` | string | One of `input_url` or `input_path` | URL to download video from (presigned S3, public URL, etc.) |
| `input_path` | string | One of `input_url` or `input_path` | Absolute path inside the container (e.g. `/root/cv_experiments/inputs/vid.mp4`) |
| `output_filename` | string | No | Output filename. Auto-generated UUID if omitted. |
| `gpu_tier` | string | No | `"standard"` (L4), `"premium"` (L40S), or omit for auto-routing by video size. |
| `kwargs` | object | No | Parameters passed to `create_zoom_bounce_effect()`. See below. |

### kwargs — Effect Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `bounces` | array | `[(1.0, 2.5)]` | Timeline of effects. See effect types below. |
| `stabilize` | float | `0.0` | Stabilization strength. `0` = off (GPU path). Non-zero falls back to CPU. |
| `debug_labels` | bool | `false` | Overlay debug text showing effect names/timing on the video. |
| `zoom_max` | float | `1.4` | Default max zoom level. |
| `bounce_mode` | string | `"snap"` | Default easing. Overridden per-bounce by `ease` field. |
| `face_side` | string | `"center"` | Face tracking anchor: `"center"`, `"left"`, `"right"`. |
| `fade_mode` | string | `"band"` | Fade style for transitions. |

### Bounce/Effect Types

Each item in the `bounces` array is an object with an `action` field:

#### `in` — Zoom in
```json
{"action": "in", "start": 1.2, "end": 1.6, "ease": "snap", "zoom": 1.35}
```

#### `out` — Zoom out
```json
{"action": "out", "start": 3.1, "end": 3.5, "ease": "smooth"}
```

#### `bounce` — Zoom in then out
```json
{"action": "bounce", "start": 5.2, "end": 6.0, "ease": "overshoot", "zoom": 1.28}
```

#### `zoom_blur` — Radial blur effect
```json
{"action": "zoom_blur", "start": 7.4, "end": 8.0, "intensity": 0.9, "n_samples": 7}
```

#### `whip` — Whip pan effect
```json
{"action": "whip", "start": 10.4, "end": 10.8, "direction": "h", "intensity": 0.9}
```

**Easing options:** `"snap"`, `"smooth"`, `"overshoot"`

---

## Usage Patterns

### Pattern A: Quick sync call (short videos)

```bash
# One-liner: submit + download
curl -X POST https://writesonic--zoom-bounce-prod-process-sync.modal.run \
  -H "Content-Type: application/json" \
  -o result.mp4 \
  -d '{"input_url": "https://...", "kwargs": {"bounces": [...]}}'
```

### Pattern B: Async submit + poll + download (long videos)

```bash
# 1. Submit
CALL_ID=$(curl -s -X POST https://writesonic--zoom-bounce-prod-submit-async.modal.run \
  -H "Content-Type: application/json" \
  -d '{"input_url": "https://...", "kwargs": {"bounces": [...]}}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['call_id'])")

echo "Submitted: $CALL_ID"

# 2. Poll until done
while true; do
  STATUS=$(curl -s "https://writesonic--zoom-bounce-prod-job-status.modal.run?call_id=$CALL_ID" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
  echo "Status: $STATUS"
  [ "$STATUS" != "pending" ] && break
  sleep 10
done

# 3. Download
curl -o result.mp4 "https://writesonic--zoom-bounce-prod-job-result.modal.run?call_id=$CALL_ID"
```

### Pattern C: Async fire-and-forget + blocking download

```bash
# 1. Submit
CALL_ID=$(curl -s -X POST https://writesonic--zoom-bounce-prod-submit-async.modal.run \
  -H "Content-Type: application/json" \
  -d '{"input_url": "https://...", "kwargs": {"bounces": [...]}}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['call_id'])")

# 2. Download (blocks until done — no polling needed)
curl -o result.mp4 "https://writesonic--zoom-bounce-prod-job-result.modal.run?call_id=$CALL_ID"
```

### Pattern D: Explicit GPU tier selection

```bash
# Force premium tier (L40S) for a large or time-sensitive video
curl -s -X POST https://writesonic--zoom-bounce-prod-submit-async.modal.run \
  -H "Content-Type: application/json" \
  -d '{
    "input_url": "https://...",
    "gpu_tier": "premium",
    "kwargs": {"bounces": [...]}
  }'
# Response: {"call_id": "fc-...", "status": "submitted", "gpu_tier": "premium"}
```

### Pattern E: From Python (with Modal auth)

```python
from modal_prod import get_remote_processor

# Get a specific tier
processor = get_remote_processor("premium")
result_bytes = processor.process.remote(
    input_bytes=open("video.mp4", "rb").read(),
    output_filename="processed.mp4",
    stabilize=0,
    bounces=[
        {"action": "in", "start": 1.0, "end": 1.5, "ease": "snap", "zoom": 1.4},
    ],
)
open("processed.mp4", "wb").write(result_bytes)
```

### Pattern F: Modal CLI

```bash
# Auto-routes based on file size
modal run modal_prod.py --input-file /path/to/video.mp4

# Explicit tier
modal run modal_prod.py --input-file /path/to/video.mp4 --gpu-tier premium
```

---

## Infrastructure

| Setting | Standard Tier | Premium Tier |
|---------|---------------|--------------|
| GPU | NVIDIA L4 | NVIDIA L40S |
| Timeout | 30 minutes | 30 minutes |
| Retries | 2 (exponential backoff: 5s, 10s) | 2 (exponential backoff: 5s, 10s) |
| Scaledown | 2 minutes | 5 minutes warm pool |
| Image | CUDA 12.2 + ffmpeg (NVENC) + cupy + opencv + mediapipe | Same |
| Auth | None (public endpoints) | Same |

### Deploy

```bash
cd cv_experiments
modal deploy modal_prod.py
```

After deploying, you should see both `ZoomBounceL4.*` and `ZoomBounceL40S.*` functions registered.

### Dev mode (hot-reload, cheaper GPU)

```bash
modal serve modal_dev.py          # hot-reload on code changes
modal run modal_dev.py --case vid_short_mix  # run a test case
```

---

## Limits & Gotchas

- **No auth** — endpoints are public. Add bearer token or `modal.web_auth` before exposing to users.
- **Bytes through scheduler** — video bytes pass through Modal's scheduler (~2x video size). Fine for <200MB videos. For larger, switch to S3 direct I/O.
- **Presigned URL expiry** — if using S3 presigned URLs, ensure they don't expire before the endpoint downloads the video (allow 2+ min).
- **Cold start** — first request after scaledown takes ~30-60s (GPU container boot + library imports). Subsequent requests on warm containers are fast. Standard tier (L4) has shorter scaledown (2 min) so cold starts are more frequent but cheaper.
- **stabilize != 0** — falls back to CPU path (much slower). Keep `stabilize: 0` for GPU speed.
- **Fallback behavior** — if the selected GPU tier fails, the system tries the next tier automatically. This means a "standard" request may end up on L40S if L4 has no capacity.

---

## Future: S3 Direct I/O

When S3 mount is set up, the flow changes to:

```
Caller → POST /submit {input_key: "uploads/vid.mp4"} → GPU reads from S3 → GPU writes to S3 → Caller gets output_key
```

No bytes through the scheduler. Only filenames. This removes the ~200MB practical limit.
