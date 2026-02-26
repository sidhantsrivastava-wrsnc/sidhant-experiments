"""
Batch runner for create_zoom_bounce_effect().

Define multiple video test cases (each with its own timeline), then run:
  python cv_experiments/zoom_bounce_test_runner.py --all
or:
  python cv_experiments/zoom_bounce_test_runner.py --case extremelylongvid_mix
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

def _load_renderers():
    """Lazy-load renderers to avoid heavy imports at module level."""
    from zoom_bounce import create_zoom_bounce_effect

    try:
        from zoom_bounce_nuclear import create_zoom_bounce_effect_nuclear
    except ImportError:
        create_zoom_bounce_effect_nuclear = None

    try:
        from zoom_bounce_gpu import create_zoom_bounce_effect as create_zoom_bounce_effect_gpu
    except ImportError:
        create_zoom_bounce_effect_gpu = None

    return create_zoom_bounce_effect, create_zoom_bounce_effect_nuclear, create_zoom_bounce_effect_gpu


TEST_CASES = [
    {
        "name": "extremelylongvid_mix",
        "input_path": "extremelylongvid.mp4",
        "output_template": "{stem}_all_effects_{ts}.mp4",
        "kwargs": {
            "stabilize": 0,
            "debug_labels": True,
            "bounces": [
                {"action": "in", "start": 30.0, "end": 30.5, "ease": "snap", "zoom": 1.4},
                {"action": "out", "start": 38.0, "end": 38.8, "ease": "smooth"},
                {
                    "action": "bounce",
                    "start": 180.0,
                    "end": 181.5,
                    "ease": "smooth",
                    "zoom": 1.3,
                },
                {
                    "action": "zoom_blur",
                    "start": 180.2,
                    "end": 181.3,
                    "intensity": 1.0,
                    "n_samples": 8,
                },
                {
                    "action": "whip",
                    "start": 360.0,
                    "end": 360.5,
                    "direction": "h",
                    "intensity": 1.0,
                },
                {
                    "action": "in",
                    "start": 540.0,
                    "end": 540.3,
                    "ease": "overshoot",
                    "zoom": 1.5,
                },
                {"action": "out", "start": 548.0, "end": 548.5, "ease": "snap"},
                (720.0, 721.5, "smooth", 1.3),
                {
                    "action": "bounce",
                    "start": 900.0,
                    "end": 901.5,
                    "ease": "overshoot",
                    "zoom": 1.5,
                },
                {
                    "action": "whip",
                    "start": 960.0,
                    "end": 960.5,
                    "direction": "v",
                    "intensity": 0.8,
                },
                {"action": "in", "start": 1080.0, "end": 1080.8, "ease": "smooth", "zoom": 1.4},
                {"action": "out", "start": 1260.0, "end": 1260.5, "ease": "overshoot"},
            ],
        },
    },
    {
        # verylongvid.mp4 duration: ~108.96s
        "name": "verylongvid_mix",
        "input_path": "verylongvid.mp4",
        "output_template": "{stem}_mix_{ts}.mp4",
        "kwargs": {
            "stabilize": 0,
            "debug_labels": True,
            "bounces": [
                {"action": "in", "start": 6.0, "end": 6.5, "ease": "smooth", "zoom": 1.35},
                {"action": "out", "start": 12.0, "end": 12.5, "ease": "snap"},
                {
                    "action": "bounce",
                    "start": 24.0,
                    "end": 25.2,
                    "ease": "smooth",
                    "zoom": 1.3,
                },
                {
                    "action": "whip",
                    "start": 36.0,
                    "end": 36.5,
                    "direction": "h",
                    "intensity": 1.0,
                },
                {
                    "action": "zoom_blur",
                    "start": 49.8,
                    "end": 50.8,
                    "intensity": 1.0,
                    "n_samples": 8,
                },
                {
                    "action": "in",
                    "start": 66.0,
                    "end": 66.4,
                    "ease": "overshoot",
                    "zoom": 1.45,
                },
                {"action": "out", "start": 74.0, "end": 74.5, "ease": "smooth"},
                {
                    "action": "whip",
                    "start": 90.0,
                    "end": 90.4,
                    "direction": "v",
                    "intensity": 0.85,
                },
                {
                    "action": "bounce",
                    "start": 103.2,
                    "end": 104.6,
                    "ease": "snap",
                    "zoom": 1.32,
                },
            ],
        },
    },
    {
        # vid.mp4 duration: ~15.05s
        "name": "vid_short_mix",
        "input_path": "vid.mp4",
        "output_template": "{stem}_short_mix_{ts}.mp4",
        "kwargs": {
            "stabilize": 0,
            "debug_labels": True,
            "bounces": [
                {"action": "in", "start": 1.2, "end": 1.6, "ease": "snap", "zoom": 1.35},
                {"action": "out", "start": 3.1, "end": 3.5, "ease": "smooth"},
                {
                    "action": "bounce",
                    "start": 5.2,
                    "end": 6.0,
                    "ease": "overshoot",
                    "zoom": 1.28,
                },
                {
                    "action": "zoom_blur",
                    "start": 7.4,
                    "end": 8.0,
                    "intensity": 0.9,
                    "n_samples": 7,
                },
                {
                    "action": "whip",
                    "start": 10.4,
                    "end": 10.8,
                    "direction": "h",
                    "intensity": 0.9,
                },
                {
                    "action": "in",
                    "start": 12.2,
                    "end": 12.5,
                    "ease": "smooth",
                    "zoom": 1.3,
                },
                {"action": "out", "start": 14.0, "end": 14.4, "ease": "snap"},
            ],
        },
    },
    {
        # longvid.mp4 duration: ~29.47s
        "name": "longvid_mix",
        "input_path": "longvid.mp4",
        "output_template": "{stem}_mix_{ts}.mp4",
        "kwargs": {
            "stabilize": 0,
            "debug_labels": True,
            "bounces": [
                {"action": "in", "start": 2.0, "end": 2.4, "ease": "smooth", "zoom": 1.35},
                {"action": "out", "start": 5.2, "end": 5.6, "ease": "snap"},
                {
                    "action": "bounce",
                    "start": 8.8,
                    "end": 9.8,
                    "ease": "overshoot",
                    "zoom": 1.32,
                },
                {
                    "action": "zoom_blur",
                    "start": 12.0,
                    "end": 12.7,
                    "intensity": 0.95,
                    "n_samples": 7,
                },
                {
                    "action": "whip",
                    "start": 16.0,
                    "end": 16.4,
                    "direction": "v",
                    "intensity": 0.9,
                },
                {
                    "action": "in",
                    "start": 20.4,
                    "end": 20.8,
                    "ease": "snap",
                    "zoom": 1.38,
                },
                {"action": "out", "start": 24.4, "end": 24.8, "ease": "smooth"},
                {
                    "action": "bounce",
                    "start": 27.0,
                    "end": 28.3,
                    "ease": "smooth",
                    "zoom": 1.28,
                },
            ],
        },
    },
    {
        # bansi_SAM_6_minutes.mp4 duration: ~310.68s (5m 11s), 4K 3840x2160, h264
        "name": "really_big_vid",
        "input_path": "really_big_vid.mp4",
        "output_template": "{stem}_big_{ts}.mp4",
        "kwargs": {
            "stabilize": 0,
            "debug_labels": True,
            "bounces": [
                {"action": "in", "start": 8.0, "end": 8.5, "ease": "snap", "zoom": 1.4},
                {"action": "out", "start": 16.0, "end": 16.6, "ease": "smooth"},
                {
                    "action": "bounce",
                    "start": 40.0,
                    "end": 41.5,
                    "ease": "smooth",
                    "zoom": 1.3,
                },
                {
                    "action": "zoom_blur",
                    "start": 40.2,
                    "end": 41.3,
                    "intensity": 1.0,
                    "n_samples": 8,
                },
                {
                    "action": "whip",
                    "start": 75.0,
                    "end": 75.5,
                    "direction": "h",
                    "intensity": 1.0,
                },
                {
                    "action": "in",
                    "start": 120.0,
                    "end": 120.4,
                    "ease": "overshoot",
                    "zoom": 1.45,
                },
                {"action": "out", "start": 140.0, "end": 140.5, "ease": "snap"},
                {
                    "action": "bounce",
                    "start": 200.0,
                    "end": 201.5,
                    "ease": "overshoot",
                    "zoom": 1.35,
                },
                {
                    "action": "whip",
                    "start": 240.0,
                    "end": 240.4,
                    "direction": "v",
                    "intensity": 0.85,
                },
                {"action": "in", "start": 280.0, "end": 280.6, "ease": "smooth", "zoom": 1.38},
                {"action": "out", "start": 300.0, "end": 300.5, "ease": "overshoot"},
            ],
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run zoom-bounce batch test cases")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Case name to run (repeat flag for multiple names)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all test cases",
    )
    parser.add_argument(
        "--nuclear",
        action="store_true",
        help="Use nuclear (zero-copy FFmpeg) renderer instead of original",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU (CuPy + h264_nvenc) renderer",
    )
    parser.add_argument(
        "--skip-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip cases where input file does not exist (default: true)",
    )
    return parser.parse_args()


def resolve_cases(args: argparse.Namespace) -> list[dict]:
    if args.all:
        return TEST_CASES
    if args.case:
        wanted = set(args.case)
        selected = [case for case in TEST_CASES if case["name"] in wanted]
        missing = sorted(wanted - {case["name"] for case in selected})
        if missing:
            raise ValueError(f"Unknown case name(s): {', '.join(missing)}")
        return selected
    raise ValueError("Choose one: --all OR --case <name>")


def main() -> None:
    args = parse_args()
    ts = int(time.time())
    base_dir = Path(__file__).resolve().parent
    input_dir = base_dir / "inputs"
    output_dir = base_dir / "outputs"
    output_dir.mkdir(exist_ok=True)
    selected = resolve_cases(args)

    create_zoom_bounce_effect, create_zoom_bounce_effect_nuclear, create_zoom_bounce_effect_gpu = _load_renderers()

    if args.gpu:
        if create_zoom_bounce_effect_gpu is None:
            raise ImportError("GPU renderer unavailable (cupy not installed — requires NVIDIA GPU)")
        effect_fn = create_zoom_bounce_effect_gpu
        mode_label = "gpu"
    elif args.nuclear:
        if create_zoom_bounce_effect_nuclear is None:
            raise ImportError("Nuclear renderer unavailable (zoom_bounce_nuclear not found)")
        effect_fn = create_zoom_bounce_effect_nuclear
        mode_label = "nuclear"
    else:
        effect_fn = create_zoom_bounce_effect
        mode_label = "original"
    print(f"Running {len(selected)} case(s) [{mode_label}]")
    for case in selected:
        input_path = input_dir / case["input_path"]
        if not input_path.exists():
            msg = f"Input missing for case '{case['name']}': {input_path}"
            if args.skip_missing:
                print(f"SKIP: {msg}")
                continue
            raise FileNotFoundError(msg)

        output_name = case["output_template"].format(stem=input_path.stem, ts=ts)
        output_path = output_dir / output_name

        print(f"\n=== {case['name']} ===")
        print(f"Input:  {input_path}")
        print(f"Output: {output_path}")
        effect_fn(
            input_path=str(input_path),
            output_path=str(output_path),
            **case["kwargs"],
        )


if __name__ == "__main__":
    main()
