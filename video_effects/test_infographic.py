"""Standalone test CLI for the infographic code-generation pipeline.

Calls the A0-A4 activity functions directly without Temporal,
allowing fast iteration on prompts and codegen quality.

Usage:
    python -m video_effects.test_infographic --text "transcript here..."
    python -m video_effects.test_infographic --file transcript.txt
    python -m video_effects.test_infographic --file transcript.txt --type pie_chart
    python -m video_effects.test_infographic --spec spec.json  # skip planning
"""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

from video_effects.activities.infographic import (
    build_generated_registry,
    cleanup_generated,
    generate_infographic_code,
    plan_infographics,
    validate_infographic,
)
from video_effects.config import settings
from video_effects.helpers.remotion import _get_remotion_dir


def _heartbeat_mock(msg: str) -> None:
    print(f"  [{msg}]")


def _build_spatial_context(
    transcript: str, video_duration: float, video_fps: int
) -> dict:
    return {
        "video": {
            "width": 1920,
            "height": 1080,
            "fps": video_fps,
            "duration": video_duration,
        },
        "transcript": {"full_text": transcript, "segments": []},
        "face_windows": [],
        "opencv_effects": [],
        "face_data_path": "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test infographic code-generation pipeline without Temporal"
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", help="Inline transcript string")
    input_group.add_argument("--file", help="Path to transcript text file")
    input_group.add_argument(
        "--spec",
        help="Path to JSON file with InfographicSpec list (skips planning)",
    )

    parser.add_argument(
        "--type", help="Filter to a single infographic type (e.g. pie_chart)"
    )
    parser.add_argument(
        "--retries", type=int, default=3, help="Max codegen retries per infographic"
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip A3 validation (tsc + render)",
    )
    parser.add_argument("--model", help="Override LLM model")
    parser.add_argument(
        "--no-show-code",
        action="store_true",
        help="Don't print generated TSX to stdout",
    )
    parser.add_argument("--style", help="Style preset name")
    parser.add_argument(
        "--video-duration", type=float, default=60, help="Fake video duration (s)"
    )
    parser.add_argument("--video-fps", type=int, default=30, help="Fake video FPS")

    args = parser.parse_args()
    show_code = not args.no_show_code

    # Apply model override
    if args.model:
        settings.INFOGRAPHIC_LLM_MODEL = args.model

    # Load style config
    style_config = None
    if args.style:
        try:
            from video_effects.helpers.style import get_style

            style_config = get_style(args.style).config.model_dump()
        except Exception as e:
            print(f"Warning: Could not load style '{args.style}': {e}")

    # Resolve transcript
    transcript = ""
    if args.text:
        transcript = args.text
    elif args.file:
        transcript = Path(args.file).read_text()

    print("═══ INFOGRAPHIC TEST CLI ═══\n")

    # ------------------------------------------------------------------
    # A0: Cleanup
    # ------------------------------------------------------------------
    with patch("temporalio.activity.heartbeat", side_effect=_heartbeat_mock):
        result = cleanup_generated({})
    print(f"── A0: Cleanup ── removed {result['cleaned']} files\n")

    # ------------------------------------------------------------------
    # A1: Plan (or load from --spec)
    # ------------------------------------------------------------------
    if args.spec:
        print("── A1: Planning (skipped — loading from spec file) ──")
        raw_specs = json.loads(Path(args.spec).read_text())
        if isinstance(raw_specs, dict):
            raw_specs = raw_specs.get("infographics", [raw_specs])
        specs = raw_specs
        reasoning = "(loaded from file)"
    else:
        print("── A1: Planning ──")
        print(f"  Transcript: {len(transcript)} chars")

        spatial_context = _build_spatial_context(
            transcript, args.video_duration, args.video_fps
        )
        plan_input = {
            "spatial_context": spatial_context,
            "transcript": transcript,
            "segments": [],
            "style_config": style_config,
            "video_fps": args.video_fps,
        }

        with patch("temporalio.activity.heartbeat", side_effect=_heartbeat_mock):
            plan_result = plan_infographics(plan_input)

        specs = plan_result["infographics"]
        reasoning = plan_result.get("reasoning", "")

    if reasoning:
        print(f"  LLM reasoning: \"{reasoning[:200]}{'...' if len(reasoning) > 200 else ''}\"")

    # Apply --type filter
    if args.type:
        specs = [s for s in specs if s.get("type") == args.type]
        if not specs:
            print(f"\n  No infographics of type '{args.type}' found in plan.")
            return 1

    # Print plan table
    print(f"\n  {'#':<3} {'Type':<18} {'Title':<25} {'Time':<12} {'Data keys'}")
    for i, spec in enumerate(specs, 1):
        data = spec.get("data", {})
        data_summary = ", ".join(
            f"{k}({len(v) if isinstance(v, list) else v})"
            for k, v in list(data.items())[:4]
        )
        print(
            f"  {i:<3} {spec.get('type', '?'):<18} "
            f"{spec.get('title', '?')[:24]:<25} "
            f"{spec.get('start_time', 0):.1f}-{spec.get('end_time', 0):.1f}s  "
            f"{data_summary}"
        )
    print()

    # ------------------------------------------------------------------
    # A2+A3: Generate + validate loop
    # ------------------------------------------------------------------
    video_info = {
        "width": 1920,
        "height": 1080,
        "fps": args.video_fps,
        "duration": args.video_duration,
    }

    successful_components = []
    results_summary = []

    for idx, spec in enumerate(specs, 1):
        spec_type = spec.get("type", "unknown")
        spec_title = spec.get("title", "Untitled")
        print(f"── A2+A3: Generating {spec_type} ({idx}/{len(specs)}) ──")

        generated = None
        previous_errors: list[str] = []
        previous_code = ""

        for attempt in range(1, args.retries + 1):
            print(f"  Attempt {attempt}/{args.retries}... generating code...")

            codegen_input = {
                "spec": spec,
                "style_config": style_config,
                "video_info": video_info,
                "attempt": attempt,
                "previous_errors": previous_errors,
                "previous_code": previous_code,
            }

            with patch(
                "temporalio.activity.heartbeat", side_effect=_heartbeat_mock
            ):
                gen_result = generate_infographic_code(codegen_input)

            tsx_code = gen_result["tsx_code"]
            export_name = gen_result["export_name"]
            component_id = gen_result["component_id"]
            line_count = tsx_code.count("\n") + 1

            print(f"  Export: {export_name} ({component_id}.tsx)")
            print(f"  Code: {line_count} lines")

            if show_code:
                print(f"  ┌── TSX ──")
                for line in tsx_code.split("\n"):
                    print(f"  │ {line}")
                print(f"  └──────────")

            # Write TSX to disk regardless of validation
            generated_dir = _get_remotion_dir() / "src" / "components" / "generated"
            generated_dir.mkdir(parents=True, exist_ok=True)
            component_path = generated_dir / f"{component_id}.tsx"
            component_path.write_text(tsx_code)
            print(f"  Wrote: {component_path}")

            # Validate
            if args.skip_validate:
                print("  Validating... SKIPPED (--skip-validate)")
                generated = gen_result
                break

            print("  Validating...", end=" ", flush=True)

            val_input = {
                "component_id": component_id,
                "tsx_code": tsx_code,
                "export_name": export_name,
            }
            with patch(
                "temporalio.activity.heartbeat", side_effect=_heartbeat_mock
            ):
                val_result = validate_infographic(val_input)

            if val_result["valid"]:
                print("tsc ✓  render ✓")
                print("  PASS")
                generated = gen_result
                break
            else:
                errors = val_result["errors"]
                tsc_errors = [e for e in errors if "error TS" in e]
                render_errors = [e for e in errors if "Render" in e]

                if tsc_errors:
                    print(f"tsc ✗ ({len(tsc_errors)} errors)")
                elif render_errors:
                    print("tsc ✓  render ✗")
                else:
                    print(f"FAIL ({len(errors)} errors)")

                for err in errors[:5]:
                    print(f"    - {err}")
                if len(errors) > 5:
                    print(f"    ... and {len(errors) - 5} more")

                previous_errors = errors
                previous_code = tsx_code

        if generated:
            successful_components.append(
                {
                    "component_id": generated["component_id"],
                    "export_name": generated["export_name"],
                    "spec": spec,
                    "props": generated["props"],
                }
            )
            results_summary.append((spec_type, spec_title, "✓", attempt))
        else:
            results_summary.append((spec_type, spec_title, "✗", args.retries))

        print()

    # ------------------------------------------------------------------
    # A4: Build registry
    # ------------------------------------------------------------------
    print("── A4: Registry ──")
    if successful_components:
        registry_input = {
            "generated_components": successful_components,
            "video_fps": args.video_fps,
            "style_config": style_config,
        }
        with patch("temporalio.activity.heartbeat", side_effect=_heartbeat_mock):
            reg_result = build_generated_registry(registry_input)

        print(
            f"  Wrote _registry.ts with {len(successful_components)} components"
        )
        print(f"  Path: {reg_result['registry_path']}")
    else:
        print("  No components to register.")
    print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("═══ SUMMARY ═══")
    passed = 0
    for spec_type, title, status, attempts in results_summary:
        label = f"{spec_type}: {title}"
        if status == "✓":
            print(f"  {label:<40} ✓ (attempt {attempts})")
            passed += 1
        else:
            print(f"  {label:<40} ✗ (failed after {attempts} attempts)")

    total = len(results_summary)
    print(f"  Total: {passed}/{total} generated, {total - passed} failed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
