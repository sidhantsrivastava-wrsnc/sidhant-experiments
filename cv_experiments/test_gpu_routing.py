"""
Test bed for GPU tier routing.

Usage:
    modal run test_gpu_routing.py                # run all tests
    modal run test_gpu_routing.py --dry-run      # print test matrix only
    modal run test_gpu_routing.py --filter vid    # run only tests matching "vid"
"""

from __future__ import annotations

import os
import time

import modal

from modal_config import GPU_TIERS, FALLBACK_ORDER

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = modal.App("zoom-bounce-test-routing")

# ---------------------------------------------------------------------------
# Deployed app lookup — call the already-deployed prod classes by name
# ---------------------------------------------------------------------------
PROD_APP_NAME = "zoom-bounce-prod"
_TIER_TO_CLS_NAME = {
    "standard": "ZoomBounceL4",
    "premium": "ZoomBounceL40S",
}


def _call_deployed(input_bytes: bytes, output_filename: str, tier: str, **kwargs):
    """Call the deployed prod processor via modal.Cls.from_name. Returns (result_bytes, tier_used)."""
    start_idx = FALLBACK_ORDER.index(tier) if tier in FALLBACK_ORDER else 0
    tiers_to_try = FALLBACK_ORDER[start_idx:]

    last_exc = None
    for t in tiers_to_try:
        try:
            cls_name = _TIER_TO_CLS_NAME[t]
            processor = modal.Cls.from_name(PROD_APP_NAME, cls_name)()
            result = processor.process.remote(
                input_bytes=input_bytes,
                output_filename=output_filename,
                **kwargs,
            )
            return result, t
        except Exception as exc:
            print(f"         Tier {t} failed: {exc} — trying next tier")
            last_exc = exc
            continue

    raise RuntimeError(f"All GPU tiers exhausted. Last error: {last_exc}")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "inputs")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "test_routing")

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
TEST_CASES = [
    # ── vid.mp4 (17MB, 15s) — small, multiple runs ──────────────────────
    # State rules: `in` sets zoomed=true, `out` sets zoomed=false.
    # bounce/zoom_blur/whip don't change zoom state.
    # Every `in` needs a matching `out` before the next `in`.
    {
        "name": "vid_auto",
        "input_file": "vid.mp4",
        "gpu_tier": None,
        "expected_tier": "standard",
        "bounces": [
            # pair 1: in → effects while zoomed → out
            {"action": "in", "start": 0.5, "end": 1.0, "ease": "snap", "zoom": 1.3},
            {"action": "zoom_blur", "start": 1.5, "end": 2.0},
            {"action": "whip", "start": 2.5, "end": 2.8, "direction": "h"},
            {"action": "out", "start": 3.5, "end": 4.0, "ease": "smooth"},
            # pair 2: in → effects while zoomed → out
            {"action": "in", "start": 5.0, "end": 5.5, "ease": "overshoot", "zoom": 1.35},
            {"action": "zoom_blur", "start": 6.0, "end": 6.5, "intensity": 0.8},
            {"action": "out", "start": 7.0, "end": 7.5, "ease": "snap"},
            # pair 3: in → whip → out
            {"action": "in", "start": 8.5, "end": 9.0, "ease": "smooth", "zoom": 1.25},
            {"action": "whip", "start": 9.5, "end": 9.8, "direction": "v"},
            {"action": "out", "start": 10.5, "end": 11.0, "ease": "overshoot"},
            # pair 4: in → out
            {"action": "in", "start": 12.0, "end": 12.5, "ease": "snap", "zoom": 1.4},
            {"action": "out", "start": 13.5, "end": 14.0, "ease": "smooth"},
        ],
        "kwargs": {"stabilize": 0, "debug_labels": True},
    },
    {
        "name": "vid_premium",
        "input_file": "vid.mp4",
        "gpu_tier": "premium",
        "expected_tier": "premium",
        "bounces": [
            # pair 1
            {"action": "in", "start": 0.3, "end": 0.8, "ease": "snap", "zoom": 1.4},
            {"action": "zoom_blur", "start": 1.3, "end": 1.8, "intensity": 0.9},
            {"action": "whip", "start": 2.3, "end": 2.6, "direction": "h"},
            {"action": "out", "start": 3.2, "end": 3.7, "ease": "smooth"},
            # pair 2
            {"action": "in", "start": 4.5, "end": 5.0, "ease": "overshoot", "zoom": 1.35},
            {"action": "whip", "start": 5.5, "end": 5.8, "direction": "v"},
            {"action": "zoom_blur", "start": 6.3, "end": 6.8},
            {"action": "out", "start": 7.5, "end": 8.0, "ease": "snap"},
            # pair 3
            {"action": "in", "start": 9.0, "end": 9.5, "ease": "smooth", "zoom": 1.3},
            {"action": "zoom_blur", "start": 10.0, "end": 10.5, "intensity": 0.7},
            {"action": "out", "start": 11.0, "end": 11.5, "ease": "overshoot"},
            # pair 4
            {"action": "in", "start": 12.5, "end": 13.0, "ease": "snap", "zoom": 1.25},
            {"action": "out", "start": 13.5, "end": 14.0, "ease": "smooth"},
        ],
        "kwargs": {"stabilize": 0, "debug_labels": True},
    },
    {
        "name": "vid_standard",
        "input_file": "vid.mp4",
        "gpu_tier": "standard",
        "expected_tier": "standard",
        "bounces": [
            # pair 1
            {"action": "in", "start": 0.5, "end": 1.0, "ease": "snap", "zoom": 1.3},
            {"action": "whip", "start": 1.5, "end": 1.8, "direction": "h"},
            {"action": "out", "start": 2.5, "end": 3.0, "ease": "smooth"},
            # pair 2
            {"action": "in", "start": 3.5, "end": 4.0, "ease": "overshoot", "zoom": 1.35},
            {"action": "zoom_blur", "start": 4.5, "end": 5.0, "intensity": 0.8},
            {"action": "whip", "start": 5.5, "end": 5.8, "direction": "v"},
            {"action": "out", "start": 6.5, "end": 7.0, "ease": "snap"},
            # pair 3
            {"action": "in", "start": 7.5, "end": 8.0, "ease": "smooth", "zoom": 1.25},
            {"action": "zoom_blur", "start": 8.5, "end": 9.0},
            {"action": "out", "start": 9.5, "end": 10.0, "ease": "overshoot"},
            # pair 4
            {"action": "in", "start": 11.0, "end": 11.5, "ease": "snap", "zoom": 1.4},
            {"action": "whip", "start": 12.0, "end": 12.3, "direction": "h"},
            {"action": "zoom_blur", "start": 12.8, "end": 13.3, "intensity": 0.7},
            {"action": "out", "start": 13.8, "end": 14.3, "ease": "smooth"},
        ],
        "kwargs": {"stabilize": 0, "debug_labels": True},
    },
    # ── verylongvid.mp4 (38MB, 109s) — small, multiple runs ─────────────
    {
        "name": "verylongvid_auto",
        "input_file": "verylongvid.mp4",
        "gpu_tier": None,
        "expected_tier": "standard",
        "bounces": [
            # pair 1
            {"action": "in", "start": 2.0, "end": 3.0, "ease": "snap", "zoom": 1.35},
            {"action": "zoom_blur", "start": 5.0, "end": 6.0, "intensity": 0.8},
            {"action": "whip", "start": 8.0, "end": 8.4, "direction": "h"},
            {"action": "out", "start": 11.0, "end": 12.0, "ease": "smooth"},
            # pair 2
            {"action": "in", "start": 18.0, "end": 19.0, "ease": "overshoot", "zoom": 1.4},
            {"action": "zoom_blur", "start": 23.0, "end": 24.0},
            {"action": "out", "start": 28.0, "end": 29.0, "ease": "snap"},
            # pair 3
            {"action": "in", "start": 35.0, "end": 36.0, "ease": "smooth", "zoom": 1.25},
            {"action": "whip", "start": 40.0, "end": 40.4, "direction": "v"},
            {"action": "zoom_blur", "start": 45.0, "end": 46.0, "intensity": 0.6},
            {"action": "out", "start": 50.0, "end": 51.0, "ease": "overshoot"},
            # pair 4
            {"action": "in", "start": 58.0, "end": 59.0, "ease": "snap", "zoom": 1.3},
            {"action": "whip", "start": 63.0, "end": 63.4, "direction": "h"},
            {"action": "out", "start": 68.0, "end": 69.0, "ease": "smooth"},
            # pair 5
            {"action": "in", "start": 76.0, "end": 77.0, "ease": "overshoot", "zoom": 1.35},
            {"action": "zoom_blur", "start": 82.0, "end": 83.0, "intensity": 0.9},
            {"action": "whip", "start": 88.0, "end": 88.4, "direction": "v"},
            {"action": "out", "start": 93.0, "end": 94.0, "ease": "snap"},
            # pair 6
            {"action": "in", "start": 100.0, "end": 101.0, "ease": "smooth", "zoom": 1.4},
            {"action": "out", "start": 105.0, "end": 106.0, "ease": "overshoot"},
        ],
        "kwargs": {"stabilize": 0, "debug_labels": True},
    },
    {
        "name": "verylongvid_premium",
        "input_file": "verylongvid.mp4",
        "gpu_tier": "premium",
        "expected_tier": "premium",
        "bounces": [
            # pair 1
            {"action": "in", "start": 3.0, "end": 4.0, "ease": "snap", "zoom": 1.3},
            {"action": "zoom_blur", "start": 7.0, "end": 8.0, "intensity": 0.9},
            {"action": "out", "start": 12.0, "end": 13.0, "ease": "smooth"},
            # pair 2
            {"action": "in", "start": 19.0, "end": 20.0, "ease": "overshoot", "zoom": 1.4},
            {"action": "whip", "start": 24.0, "end": 24.4, "direction": "h"},
            {"action": "zoom_blur", "start": 28.0, "end": 29.0},
            {"action": "out", "start": 34.0, "end": 35.0, "ease": "snap"},
            # pair 3
            {"action": "in", "start": 42.0, "end": 43.0, "ease": "smooth", "zoom": 1.35},
            {"action": "whip", "start": 47.0, "end": 47.4, "direction": "v"},
            {"action": "out", "start": 52.0, "end": 53.0, "ease": "overshoot"},
            # pair 4
            {"action": "in", "start": 60.0, "end": 61.0, "ease": "snap", "zoom": 1.25},
            {"action": "zoom_blur", "start": 65.0, "end": 66.0, "intensity": 0.7},
            {"action": "whip", "start": 70.0, "end": 70.4, "direction": "h"},
            {"action": "out", "start": 75.0, "end": 76.0, "ease": "smooth"},
            # pair 5
            {"action": "in", "start": 83.0, "end": 84.0, "ease": "overshoot", "zoom": 1.3},
            {"action": "zoom_blur", "start": 89.0, "end": 90.0, "intensity": 0.8},
            {"action": "out", "start": 95.0, "end": 96.0, "ease": "snap"},
            # pair 6
            {"action": "in", "start": 101.0, "end": 102.0, "ease": "smooth", "zoom": 1.4},
            {"action": "out", "start": 106.0, "end": 107.0, "ease": "overshoot"},
        ],
        "kwargs": {"stabilize": 0, "debug_labels": True},
    },
    # ── longvid.mp4 (72MB, 29s) — large, 1 run, multi-effect ────────────
    {
        "name": "longvid_auto",
        "input_file": "longvid.mp4",
        "gpu_tier": None,
        "expected_tier": "premium",
        "bounces": [
            # pair 1
            {"action": "in", "start": 0.5, "end": 1.0, "ease": "snap", "zoom": 1.35},
            {"action": "zoom_blur", "start": 1.5, "end": 2.0, "intensity": 0.8},
            {"action": "whip", "start": 2.5, "end": 2.8, "direction": "h"},
            {"action": "out", "start": 3.5, "end": 4.0, "ease": "smooth"},
            # pair 2
            {"action": "in", "start": 5.0, "end": 5.5, "ease": "overshoot", "zoom": 1.4},
            {"action": "zoom_blur", "start": 6.0, "end": 6.5},
            {"action": "out", "start": 7.5, "end": 8.0, "ease": "snap"},
            # pair 3
            {"action": "in", "start": 9.0, "end": 9.5, "ease": "smooth", "zoom": 1.25},
            {"action": "whip", "start": 10.0, "end": 10.3, "direction": "v"},
            {"action": "zoom_blur", "start": 11.0, "end": 11.5, "intensity": 0.9},
            {"action": "out", "start": 12.5, "end": 13.0, "ease": "overshoot"},
            # pair 4
            {"action": "in", "start": 14.0, "end": 14.5, "ease": "snap", "zoom": 1.3},
            {"action": "whip", "start": 15.0, "end": 15.3, "direction": "h"},
            {"action": "out", "start": 16.0, "end": 16.5, "ease": "smooth"},
            # pair 5
            {"action": "in", "start": 17.5, "end": 18.0, "ease": "overshoot", "zoom": 1.35},
            {"action": "zoom_blur", "start": 19.0, "end": 19.5, "intensity": 0.7},
            {"action": "whip", "start": 20.5, "end": 20.8, "direction": "v"},
            {"action": "out", "start": 21.5, "end": 22.0, "ease": "snap"},
            # pair 6
            {"action": "in", "start": 23.0, "end": 23.5, "ease": "smooth", "zoom": 1.4},
            {"action": "zoom_blur", "start": 24.5, "end": 25.0},
            {"action": "out", "start": 26.0, "end": 26.5, "ease": "overshoot"},
            # pair 7
            {"action": "in", "start": 27.5, "end": 28.0, "ease": "snap", "zoom": 1.25},
            {"action": "out", "start": 28.5, "end": 29.0, "ease": "smooth"},
        ],
        "kwargs": {"stabilize": 0, "debug_labels": True},
    },
    # ── extremelylongvid.mp4 (248MB, 1344s) — large, 1 run ──────────────
    {
        "name": "extremelylongvid_auto",
        "input_file": "extremelylongvid.mp4",
        "gpu_tier": None,
        "expected_tier": "premium",
        "bounces": [
            # pair 1
            {"action": "in", "start": 5.0, "end": 6.0, "ease": "snap", "zoom": 1.4},
            {"action": "zoom_blur", "start": 15.0, "end": 16.0, "intensity": 0.8},
            {"action": "whip", "start": 25.0, "end": 25.5, "direction": "h"},
            {"action": "out", "start": 40.0, "end": 41.0, "ease": "smooth"},
            # pair 2
            {"action": "in", "start": 70.0, "end": 71.0, "ease": "overshoot", "zoom": 1.3},
            {"action": "zoom_blur", "start": 90.0, "end": 91.0},
            {"action": "whip", "start": 110.0, "end": 110.5, "direction": "v"},
            {"action": "out", "start": 130.0, "end": 131.0, "ease": "snap"},
            # pair 3
            {"action": "in", "start": 170.0, "end": 171.0, "ease": "smooth", "zoom": 1.35},
            {"action": "zoom_blur", "start": 200.0, "end": 201.0, "intensity": 0.9},
            {"action": "out", "start": 230.0, "end": 231.0, "ease": "overshoot"},
            # pair 4
            {"action": "in", "start": 280.0, "end": 281.0, "ease": "snap", "zoom": 1.25},
            {"action": "whip", "start": 310.0, "end": 310.5, "direction": "h"},
            {"action": "zoom_blur", "start": 340.0, "end": 341.0, "intensity": 0.7},
            {"action": "out", "start": 370.0, "end": 371.0, "ease": "smooth"},
            # pair 5
            {"action": "in", "start": 420.0, "end": 421.0, "ease": "overshoot", "zoom": 1.4},
            {"action": "whip", "start": 460.0, "end": 460.5, "direction": "v"},
            {"action": "out", "start": 500.0, "end": 501.0, "ease": "snap"},
            # pair 6
            {"action": "in", "start": 560.0, "end": 561.0, "ease": "smooth", "zoom": 1.3},
            {"action": "zoom_blur", "start": 600.0, "end": 601.0, "intensity": 0.8},
            {"action": "whip", "start": 640.0, "end": 640.5, "direction": "h"},
            {"action": "out", "start": 680.0, "end": 681.0, "ease": "overshoot"},
            # pair 7
            {"action": "in", "start": 740.0, "end": 741.0, "ease": "snap", "zoom": 1.35},
            {"action": "zoom_blur", "start": 790.0, "end": 791.0},
            {"action": "out", "start": 840.0, "end": 841.0, "ease": "smooth"},
            # pair 8
            {"action": "in", "start": 910.0, "end": 911.0, "ease": "overshoot", "zoom": 1.25},
            {"action": "whip", "start": 960.0, "end": 960.5, "direction": "v"},
            {"action": "zoom_blur", "start": 1010.0, "end": 1011.0, "intensity": 0.6},
            {"action": "out", "start": 1060.0, "end": 1061.0, "ease": "snap"},
            # pair 9
            {"action": "in", "start": 1130.0, "end": 1131.0, "ease": "smooth", "zoom": 1.4},
            {"action": "whip", "start": 1180.0, "end": 1180.5, "direction": "h"},
            {"action": "out", "start": 1230.0, "end": 1231.0, "ease": "overshoot"},
            # pair 10
            {"action": "in", "start": 1290.0, "end": 1291.0, "ease": "snap", "zoom": 1.3},
            {"action": "out", "start": 1335.0, "end": 1336.0, "ease": "smooth"},
        ],
        "kwargs": {"stabilize": 0, "debug_labels": True},
    },
    # ── really_big_video.webm (679MB, 395s) — large, 1 run, .webm ───────
    {
        "name": "really_big_video_auto",
        "input_file": "really_big_video.webm",
        "gpu_tier": None,
        "expected_tier": "premium",
        "bounces": [
            # pair 1
            {"action": "in", "start": 3.0, "end": 4.0, "ease": "snap", "zoom": 1.35},
            {"action": "zoom_blur", "start": 8.0, "end": 9.0, "intensity": 0.8},
            {"action": "whip", "start": 14.0, "end": 14.4, "direction": "h"},
            {"action": "out", "start": 20.0, "end": 21.0, "ease": "smooth"},
            # pair 2
            {"action": "in", "start": 35.0, "end": 36.0, "ease": "overshoot", "zoom": 1.4},
            {"action": "zoom_blur", "start": 45.0, "end": 46.0},
            {"action": "whip", "start": 55.0, "end": 55.4, "direction": "v"},
            {"action": "out", "start": 65.0, "end": 66.0, "ease": "snap"},
            # pair 3
            {"action": "in", "start": 80.0, "end": 81.0, "ease": "smooth", "zoom": 1.25},
            {"action": "zoom_blur", "start": 90.0, "end": 91.0, "intensity": 0.9},
            {"action": "out", "start": 100.0, "end": 101.0, "ease": "overshoot"},
            # pair 4
            {"action": "in", "start": 115.0, "end": 116.0, "ease": "snap", "zoom": 1.3},
            {"action": "whip", "start": 125.0, "end": 125.4, "direction": "h"},
            {"action": "zoom_blur", "start": 135.0, "end": 136.0, "intensity": 0.7},
            {"action": "out", "start": 145.0, "end": 146.0, "ease": "smooth"},
            # pair 5
            {"action": "in", "start": 165.0, "end": 166.0, "ease": "overshoot", "zoom": 1.35},
            {"action": "whip", "start": 180.0, "end": 180.4, "direction": "v"},
            {"action": "out", "start": 195.0, "end": 196.0, "ease": "snap"},
            # pair 6
            {"action": "in", "start": 215.0, "end": 216.0, "ease": "smooth", "zoom": 1.4},
            {"action": "zoom_blur", "start": 230.0, "end": 231.0, "intensity": 0.8},
            {"action": "whip", "start": 245.0, "end": 245.4, "direction": "h"},
            {"action": "out", "start": 260.0, "end": 261.0, "ease": "overshoot"},
            # pair 7
            {"action": "in", "start": 280.0, "end": 281.0, "ease": "snap", "zoom": 1.25},
            {"action": "zoom_blur", "start": 300.0, "end": 301.0},
            {"action": "out", "start": 320.0, "end": 321.0, "ease": "smooth"},
            # pair 8
            {"action": "in", "start": 345.0, "end": 346.0, "ease": "overshoot", "zoom": 1.3},
            {"action": "whip", "start": 360.0, "end": 360.4, "direction": "v"},
            {"action": "out", "start": 380.0, "end": 381.0, "ease": "snap"},
        ],
        "kwargs": {"stabilize": 0, "debug_labels": True},
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _probe_video(path: str) -> tuple[str, str]:
    """Probe video duration and resolution via ffprobe. Returns (duration_str, resolution_str)."""
    import subprocess
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,duration",
             "-show_entries", "format=duration",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=10,
        )
        lines = [l.strip() for l in out.stdout.strip().splitlines() if l.strip()]
        width, height, resolution, duration_str = None, None, "?", "?"
        for line in lines:
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    w, h = int(parts[0]), int(parts[1])
                    width, height = w, h
                    resolution = f"{w}x{h}"
                except ValueError:
                    pass
            if len(parts) == 1:
                try:
                    dur = float(parts[0])
                    m, s = divmod(int(dur), 60)
                    duration_str = f"{m}m{s:02d}s" if m else f"{s}s"
                except ValueError:
                    pass
            # stream line: width,height,duration
            if len(parts) >= 3 and width is None:
                try:
                    width, height = int(parts[0]), int(parts[1])
                    resolution = f"{width}x{height}"
                except ValueError:
                    pass
                try:
                    dur = float(parts[2])
                    m, s = divmod(int(dur), 60)
                    duration_str = f"{m}m{s:02d}s" if m else f"{s}s"
                except ValueError:
                    pass
        return duration_str, resolution
    except Exception:
        return "?", "?"


def _format_size(size_bytes: int) -> str:
    mb = size_bytes / 1e6
    if mb >= 1000:
        return f"{mb / 1000:.1f}GB"
    return f"{mb:.0f}MB"


def _effects_summary(bounces: list[dict]) -> str:
    actions = [b["action"] for b in bounces]
    if len(actions) <= 6:
        return ", ".join(actions)
    return f"{', '.join(actions[:4])}... ({len(actions)} total)"


def _print_matrix(cases: list[dict]):
    """Print the test matrix without executing."""
    print("\n" + "=" * 78)
    print("GPU ROUTING TEST MATRIX")
    print("=" * 78)
    for i, tc in enumerate(cases, 1):
        path = os.path.join(INPUT_DIR, tc["input_file"])
        size = os.path.getsize(path) if os.path.exists(path) else 0
        dur, res = _probe_video(path) if os.path.exists(path) else ("?", "?")
        tier_label = tc["gpu_tier"] or "auto"
        effects = _effects_summary(tc["bounces"])
        print(
            f"  [{i}/{len(cases)}] {tc['input_file']} ({_format_size(size)}, {dur}, {res}) "
            f"| tier: {tier_label} -> expected: {tc['expected_tier']} "
            f"| effects: {effects}"
        )
    print("=" * 78 + "\n")


def _print_summary(results: list[dict]):
    """Print final summary table."""
    print("\n" + "=" * 90)
    print(f"{'TEST':<30} {'SIZE':>7} {'REQUESTED':>10} {'USED':>10} {'STATUS':>8} {'TIME':>8}")
    print("-" * 90)

    passed = 0
    failed = 0
    for r in results:
        tier_req = r["tier_requested"] or "auto"
        status = "PASS" if r["passed"] else "FAIL"
        symbol = "+" if r["passed"] else "X"
        duration = f"{r['duration']:.1f}s" if r["duration"] else "—"
        tier_used = r.get("tier_used", "—")

        if r["passed"]:
            passed += 1
        else:
            failed += 1

        print(
            f"  {symbol} {r['name']:<28} {r['size']:>7} {tier_req:>10} "
            f"{tier_used:>10} {status:>8} {duration:>8}"
        )

    print("-" * 90)
    total = passed + failed
    print(f"  Total: {total} | Passed: {passed} | Failed: {failed}")
    print("=" * 90 + "\n")


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def run_tests(dry_run: bool = False, filter: str = ""):
    """Run GPU routing tests.

    Usage:
        modal run test_gpu_routing.py                    # run all tests
        modal run test_gpu_routing.py --dry-run           # print matrix only
        modal run test_gpu_routing.py --filter vid.mp4    # filter by name
    """
    # Filter test cases
    cases = TEST_CASES
    if filter:
        cases = [tc for tc in cases if filter in tc["input_file"] or filter in tc["name"]]
        if not cases:
            print(f"No test cases match filter: '{filter}'")
            return

    # Dry run — just print the matrix
    if dry_run:
        _print_matrix(cases)
        return

    # Ensure output dir exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Run tests sequentially
    results = []
    total = len(cases)

    print("\n" + "=" * 78)
    print(f"RUNNING {total} GPU ROUTING TESTS")
    print("=" * 78)

    for i, tc in enumerate(cases, 1):
        path = os.path.join(INPUT_DIR, tc["input_file"])
        if not os.path.exists(path):
            print(f"\n  [{i}/{total}] SKIP {tc['name']} — file not found: {path}")
            results.append({
                "name": tc["name"],
                "size": "—",
                "vid_dur": "—",
                "vid_res": "—",
                "tier_requested": tc["gpu_tier"],
                "tier_used": "—",
                "passed": False,
                "duration": None,
            })
            continue

        with open(path, "rb") as f:
            input_bytes = f.read()

        size_mb = len(input_bytes) / 1e6
        size_str = _format_size(len(input_bytes))
        vid_dur, vid_res = _probe_video(path)
        tier_label = tc["gpu_tier"] or "auto"
        effects = _effects_summary(tc["bounces"])

        print(
            f"\n  [{i}/{total}] {tc['input_file']} ({size_str}, {vid_dur}, {vid_res}) "
            f"| tier: {tier_label} -> expected: {tc['expected_tier']} "
            f"| effects: {effects}"
        )

        # Determine tier using same routing logic as prod
        if tc["gpu_tier"] and tc["gpu_tier"] in GPU_TIERS:
            tier = tc["gpu_tier"]
        else:
            tier = "premium" if size_mb >= 50 else "standard"

        # Build kwargs
        kwargs = {**tc["kwargs"], "bounces": tc["bounces"]}

        # Extension for output
        ext = os.path.splitext(tc["input_file"])[1] or ".mp4"
        output_filename = f"{tc['name']}{ext}"

        t0 = time.time()
        try:
            result_bytes, tier_used = _call_deployed(
                input_bytes=input_bytes,
                output_filename=output_filename,
                tier=tier,
                **kwargs,
            )
            elapsed = time.time() - t0

            # Save output
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            with open(output_path, "wb") as f:
                f.write(result_bytes)

            # Verify tier
            passed = tier_used == tc["expected_tier"]
            if passed:
                print(f"         + PASS (tier: {tier_used}, {elapsed:.1f}s) -> {output_path}")
            else:
                print(
                    f"         X FAIL (expected {tc['expected_tier']}, "
                    f"got {tier_used}, {elapsed:.1f}s) -> {output_path}"
                )

            results.append({
                "name": tc["name"],
                "size": size_str,
                "tier_requested": tc["gpu_tier"],
                "tier_used": tier_used,
                "passed": passed,
                "duration": elapsed,
            })

        except Exception as exc:
            elapsed = time.time() - t0
            print(f"         X ERROR ({elapsed:.1f}s): {exc}")
            results.append({
                "name": tc["name"],
                "size": size_str,
                "tier_requested": tc["gpu_tier"],
                "tier_used": "error",
                "passed": False,
                "duration": elapsed,
            })

    # Summary
    _print_summary(results)
