"""Standalone CLI for the Video Effects workflow.

Usage:
    python -m video_effects.cli run input.mp4 --output output.mp4
    python -m video_effects.cli run input.mp4 --auto-approve
"""

import argparse
import asyncio
import json
import os
import uuid

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from video_effects.config import settings
from video_effects.schemas.styles import STYLE_PRESETS
from video_effects.schemas.workflow import VideoEffectsInput


NUMBER_OF_APPROVAL_ATTEMPTS = 5

async def get_client() -> Client:
    return await Client.connect(
        settings.TEMPORAL_ENDPOINT,
        namespace=settings.TEMPORAL_NAMESPACE,
        data_converter=pydantic_data_converter,
    )


def _print_timeline(timeline: dict) -> None:
    """Pretty-print the effect timeline table."""
    effects = timeline.get("effects", [])
    conflicts = timeline.get("conflicts_resolved", 0)

    print("\n" + "=" * 100)
    print("  VIDEO EFFECTS TIMELINE")
    print("=" * 100)

    if not effects:
        print("  No effects detected.")
        print("=" * 100)
        return

    print(f"  {'#':<4} {'Type':<30} {'Start':>7} {'End':>7} {'Conf':>5}  {'Cue / Text'}")
    print("-" * 100)

    for i, e in enumerate(effects, 1):
        etype = e.get("effect_type", "?")
        start = e.get("start_time", 0)
        end = e.get("end_time", 0)
        conf = e.get("confidence", 0)
        cue = e.get("verbal_cue", "")

        # Show type-specific details
        detail = ""
        if etype == "zoom" and e.get("zoom_params"):
            zp = e["zoom_params"]
            detail = f" [{zp.get('tracking', '?')} {zp.get('zoom_level', '?')}x]"
        elif etype == "blur" and e.get("blur_params"):
            bp = e["blur_params"]
            detail = f" [{bp.get('blur_type', '?')}]"
        elif etype == "color_change" and e.get("color_params"):
            cp = e["color_params"]
            detail = f" [{cp.get('preset', '?')}]"
        elif etype == "subtitle" and e.get("subtitle_params"):
            sp = e["subtitle_params"]
            text = sp.get("text", "")
            detail = f' ["{text}"]'
        elif etype == "whip" and e.get("whip_params"):
            wp = e["whip_params"]
            detail = f" [{wp.get('direction', '?')} {wp.get('intensity', '?')}x]"
        elif etype == "speed_ramp" and e.get("speed_ramp_params"):
            srp = e["speed_ramp_params"]
            detail = f" [{srp.get('speed', '?')}x {srp.get('easing', '?')}]"
        elif etype == "vignette" and e.get("vignette_params"):
            vp = e["vignette_params"]
            detail = f" [str={vp.get('strength', '?')} r={vp.get('radius', '?')}]"

        print(f"  {i:<4} {etype + detail:<30} {start:>6.1f}s {end:>6.1f}s {conf:>4.0%}  {cue}")

    print("-" * 100)
    print(f"  Total effects: {len(effects)}  |  Conflicts resolved: {conflicts}")
    print("=" * 100)


def _print_mg_plan(mg_plan: dict) -> None:
    """Pretty-print the motion graphics plan."""
    components = mg_plan.get("components", [])
    palette = mg_plan.get("color_palette", [])
    reasoning = mg_plan.get("reasoning", "")
    issues = mg_plan.get("validation_issues", [])

    print("\n" + "=" * 100)
    print("  MOTION GRAPHICS PLAN")
    print("=" * 100)

    if not components:
        print("  No motion graphics components planned.")
        print("=" * 100)
        return

    if reasoning:
        print(f"  Reasoning: {reasoning}")
        print()

    print(f"  {'#':<4} {'Template':<20} {'Start':>7} {'End':>7}  {'Text / Props'}")
    print("-" * 100)

    for i, c in enumerate(components, 1):
        template = c.get("template", "?")
        props = c.get("props", {})
        bounds = c.get("bounds", {})

        # Try to find the best display text from props
        display_text = props.get("text", "")
        if not display_text and props.get("name"):
            display_text = props["name"]

        # Show key props compactly
        prop_parts = []
        if display_text:
            prop_parts.append(f'"{display_text}"')
        style = props.get("style")
        if style:
            prop_parts.append(f"style={style}")
        font_size = props.get("fontSize")
        if font_size:
            prop_parts.append(f"size={font_size}")
        prop_str = ", ".join(prop_parts) if prop_parts else str(props)

        # Times: Remotion format uses startFrame/durationInFrames
        start_frame = c.get("startFrame", 0)
        dur_frames = c.get("durationInFrames", 0)
        # Approximate seconds (assume 30fps if not available)
        fps = 30
        start_s = start_frame / fps
        end_s = (start_frame + dur_frames) / fps

        pos = ""
        if bounds:
            bx = bounds.get("x", 0)
            by = bounds.get("y", 0)
            pos = f" @({bx:.0%},{by:.0%})"

        print(f"  {i:<4} {template:<20} {start_s:>6.1f}s {end_s:>6.1f}s  {prop_str}{pos}")

    print("-" * 100)
    print(f"  Total: {len(components)} component(s)  |  Palette: {', '.join(palette) or 'none'}")
    if issues:
        print(f"  Validation fixes: {len(issues)}")
        for issue in issues:
            print(f"    - {issue}")
    print("=" * 100)


async def run_workflow(args) -> None:
    """Start the workflow and handle interactive approval."""
    client = await get_client()

    workflow_id = f"vfx-{uuid.uuid4().hex[:8]}"
    # --mg and --infographics both route through the code-gen pipeline
    enable_infographics = getattr(args, "infographics", False) or args.motion_graphics
    enable_programmer = getattr(args, "programmer", False)
    enable_mg = args.motion_graphics or enable_infographics or enable_programmer

    input_data = VideoEffectsInput(
        input_video=os.path.abspath(args.input),
        output_video=os.path.abspath(args.output),
        auto_approve=args.auto_approve,
        enable_motion_graphics=enable_mg,
        style=args.style,
        dev_mode=args.dev,
        smooth_jump_cuts=args.smooth_cuts,
        enable_infographics=enable_infographics,
        enable_programmer=enable_programmer,
    )

    print(f"Starting Video Effects workflow: {workflow_id}")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")

    handle = await client.start_workflow(
        "VideoEffectsWorkflow",
        input_data,
        id=workflow_id,
        task_queue=settings.TASK_QUEUE,
    )

    if not args.auto_approve:
        approved = False
        for attempt in range(NUMBER_OF_APPROVAL_ATTEMPTS):
            # Poll for timeline to be ready
            msg = "Waiting for effect analysis..." if attempt == 0 else "Waiting for re-analysis..."
            print(f"\n{msg}")
            prev_timeline = await handle.query("get_timeline")
            _print_timeline(prev_timeline)


            # Interactive approval
            while True:
                choice = input("\nApprove? [y/n/json]: ").strip().lower()
                if choice in ("y", "yes"):
                    await handle.signal("approve_timeline", [True, ""])
                    print("Timeline approved. Processing effects...")
                    approved = True
                    break
                elif choice in ("n", "no"):
                    feedback = input("Feedback (what to change): ").strip()
                    await handle.signal("approve_timeline", [False, feedback])
                    print(f"Rejected with feedback. Retrying... (attempt {attempt + 2}/{NUMBER_OF_APPROVAL_ATTEMPTS})")
                    break
                elif choice == "json":
                    print(json.dumps(prev_timeline, indent=2))
                else:
                    print("Please enter y, n, or json")

            if approved:
                break
        else:
            print(f"Max retries ({NUMBER_OF_APPROVAL_ATTEMPTS}) reached.")
            result = await handle.result()
            error = result.get("error") if isinstance(result, dict) else result.error
            if error:
                print(f"  Error: {error}")
            return

    # Wait for completion
    print("\nWaiting for workflow completion...")
    result = await handle.result()

    print("\nWorkflow completed successfully!")
    if isinstance(result, dict):
        print(f"  Output: {result.get('output_video', 'N/A')}")
        print(f"  Effects applied: {result.get('effects_applied', 'N/A')}")
        print(f"  Phases executed: {result.get('phases_executed', 'N/A')}")
        mg = result.get("motion_graphics_applied", 0)
        if mg:
            print(f"  Motion graphics: {mg} components")
        if result.get("error"):
            print(f"  Error: {result['error']}")
    else:
        print(f"  Output: {result.output_video}")
        print(f"  Effects applied: {result.effects_applied}")
        print(f"  Phases executed: {result.phases_executed}")
        if result.motion_graphics_applied:
            print(f"  Motion graphics: {result.motion_graphics_applied} components")
        if result.error:
            print(f"  Error: {result.error}")


def main():
    parser = argparse.ArgumentParser(
        description="Video Effects Workflow CLI",
        prog="python -m video_effects.cli",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the video effects workflow")
    run_parser.add_argument("input", help="Input video file path")
    run_parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output video file path (default: input_effects.mp4)",
    )
    run_parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Skip interactive approval",
    )
    run_parser.add_argument(
        "--motion-graphics", "--mg",
        action="store_true",
        help="Enable code-gen motion graphics overlays (routes through infographic pipeline)",
    )
    run_parser.add_argument(
        "--style", "-s",
        default="",
        choices=list(STYLE_PRESETS.keys()),
        help="Style preset (default: auto-detect from transcript)",
    )
    run_parser.add_argument(
        "--dev",
        action="store_true",
        help="Dev mode: effects triggered by explicit verbal commands instead of inferred",
    )
    run_parser.add_argument(
        "--smooth-cuts",
        action="store_true",
        help="Detect jump cuts and insert synthetic zoom transitions to smooth them",
    )
    run_parser.add_argument(
        "--infographics",
        action="store_true",
        help="Enable LLM-generated infographic overlays (same as --mg)",
    )
    run_parser.add_argument(
        "--programmer",
        action="store_true",
        help="Enable free-hand creative programmer workflow (replaces --infographics)",
    )

    args = parser.parse_args()

    if args.command == "run":
        if args.output is None:
            base = args.input.rsplit(".", 1)[0]
            args.output = f"{base}_effects.mp4"

        asyncio.run(run_workflow(args))


if __name__ == "__main__":
    main()
