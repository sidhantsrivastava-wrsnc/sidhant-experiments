"""
Standalone runner that uses Replicate API face detection with zoom_bounce_gpu.

Monkey-patches zoom_bounce.get_face_data_seek so that zoom_bounce_gpu picks up
the Replicate backend without any modifications to existing files.

Usage:
    python replicate_runner.py input.mp4 output.mp4 [--bounces bounces.json]

Requires:
    REPLICATE_API_TOKEN env var set
    pip install replicate
"""

import sys
import json
import argparse

# Monkey-patch before importing zoom_bounce_gpu
import zoom_bounce
from replicate_detect import get_face_data_replicate

_original_get_face_data_seek = zoom_bounce.get_face_data_seek


def _patched_get_face_data_seek(video_path, active_ranges, n_frames, stride=3):
    print("[replicate_runner] Using Replicate API for face detection")
    return get_face_data_replicate(video_path, active_ranges, n_frames, stride=stride)


zoom_bounce.get_face_data_seek = _patched_get_face_data_seek

from zoom_bounce_gpu import create_zoom_bounce_effect


def main():
    parser = argparse.ArgumentParser(description="Zoom bounce with Replicate face detection")
    parser.add_argument("input", help="Input video path")
    parser.add_argument("output", help="Output video path")
    parser.add_argument("--bounces", help="JSON file or inline JSON array of bounce configs")
    parser.add_argument("--zoom-max", type=float, default=1.4)
    parser.add_argument("--face-side", default="center", choices=["left", "right", "center"])
    parser.add_argument("--bounce-mode", default="snap")
    parser.add_argument("--stabilize", type=float, default=0.0)
    parser.add_argument("--debug-labels", action="store_true")
    args = parser.parse_args()

    bounces = None
    if args.bounces:
        try:
            bounces = json.loads(args.bounces)
        except json.JSONDecodeError:
            with open(args.bounces) as f:
                bounces = json.load(f)

    create_zoom_bounce_effect(
        args.input,
        args.output,
        zoom_max=args.zoom_max,
        bounces=bounces,
        bounce_mode=args.bounce_mode,
        face_side=args.face_side,
        stabilize=args.stabilize,
        debug_labels=args.debug_labels,
    )


if __name__ == "__main__":
    main()
