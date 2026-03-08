"""Activities for the infographic code-generation workflow (A1-A4)."""

import json
import logging
import os
import re
import subprocess
from pathlib import Path

from temporalio import activity

from video_effects.config import settings
from video_effects.helpers.llm import call_structured, call_text, load_prompt
from video_effects.helpers.remotion import render_still, _get_remotion_dir
from video_effects.schemas.infographic import (
    InfographicPlanResponse,
    InfographicSpec,
)

logger = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _build_style_guide(style_config: dict | None) -> str:
    """Build a minimal style guide section for the planner."""
    if not style_config:
        return ""
    palette = style_config.get("palette", [])
    lines = ["## Style Guide\n"]

    if isinstance(palette, list) and len(palette) >= 3:
        lines.append(f"**Color palette**: {palette[0]} (text), {palette[1]} (secondary), {palette[2]} (accent)")
    elif isinstance(palette, dict):
        # Fallback: LLM returned a dict palette
        parts = [f"{v} ({k})" for k, v in palette.items() if isinstance(v, str) and v.startswith("#")]
        if parts:
            lines.append(f"**Color palette**: {', '.join(parts)}")

    font = style_config.get("font_family", "")
    if font:
        lines.append(f"**Font**: {font}")

    return "\n".join(lines) + "\n" if len(lines) > 1 else ""


# ---------------------------------------------------------------------------
# A1: Plan infographics
# ---------------------------------------------------------------------------


@activity.defn(name="vfx_cleanup_generated")
def cleanup_generated(input_data: dict) -> dict:
    """Remove all previously generated infographic components.

    Input: {} (no params needed)
    Output: {"cleaned": int}
    """
    remotion_dir = _get_remotion_dir()
    generated_dir = remotion_dir / "src" / "components" / "generated"

    cleaned = 0
    if generated_dir.exists():
        for f in generated_dir.iterdir():
            if f.name != ".gitignore":
                f.unlink()
                cleaned += 1

    logger.info("Cleaned %d generated files", cleaned)
    return {"cleaned": cleaned}


@activity.defn(name="vfx_plan_infographics")
def plan_infographics(input_data: dict) -> dict:
    """LLM analyzes transcript and decides WHAT infographics to create.

    Input: {
        "spatial_context": dict,
        "transcript": str,
        "segments": list[dict],
        "style_config": dict | None,
        "video_fps": int,
    }
    Output: {
        "infographics": list[dict],  # list of InfographicSpec dicts
        "reasoning": str,
    }
    """
    context = input_data["spatial_context"]
    transcript = input_data.get("transcript", "")
    segments = input_data.get("segments", [])
    style_config = input_data.get("style_config")
    fps = input_data.get("video_fps", 30)

    base_prompt = load_prompt("plan_infographics.md")
    style_guide = _build_style_guide(style_config)
    system_prompt = base_prompt.replace("{STYLE_GUIDE}", style_guide)

    # Build user message
    lines = []
    video = context.get("video", {})
    lines.append(f"## Video Info")
    lines.append(f"- Duration: {video.get('duration', 0):.1f}s")
    lines.append(f"- Resolution: {video.get('width', '?')}x{video.get('height', '?')}")
    lines.append(f"- FPS: {video.get('fps', 30)}\n")

    # Face windows
    face_windows = context.get("face_windows", [])
    if face_windows:
        lines.append("## Face Position (time windows)")
        for fw in face_windows[:15]:
            fr = fw["face_region"]
            safe_labels = [s["label"] for s in fw.get("safe_regions", [])]
            lines.append(
                f"- [{fw['start_time']:.1f}s - {fw['end_time']:.1f}s] "
                f"face at ({fr['x']:.2f}, {fr['y']:.2f}) safe: {', '.join(safe_labels) or 'none'}"
            )
        lines.append("")

    # Timestamped transcript
    if segments:
        lines.append("## Timestamped Transcript")
        for seg in segments:
            if seg.get("type") == "word":
                lines.append(f"[{seg.get('start', 0):.2f}s] {seg.get('text', '')}")
        lines.append("")

    if transcript:
        lines.append(f"## Full Transcript\n\n{transcript[:3000]}\n")

    user_message = "\n".join(lines)

    activity.heartbeat("Planning infographics via LLM")

    raw = call_structured(
        system_prompt=system_prompt,
        user_message=user_message,
        response_model=InfographicPlanResponse,
        model=settings.INFOGRAPHIC_LLM_MODEL,
    )

    infographics = raw.get("infographics", [])
    logger.info("LLM planned %d infographics", len(infographics))

    return {
        "infographics": infographics,
        "reasoning": raw.get("reasoning", ""),
    }


# ---------------------------------------------------------------------------
# A2: Generate infographic code
# ---------------------------------------------------------------------------


def _build_codegen_prompt(api_reference: str) -> str:
    """Assemble the code generation system prompt with API ref and examples."""
    base = load_prompt("generate_infographic_code.md")

    # Load real component examples
    components_dir = _get_remotion_dir() / "src" / "components"
    examples = []
    for name in ("DataAnimation.tsx", "AnimatedTitle.tsx"):
        path = components_dir / name
        if path.exists():
            examples.append(f"### Example: {name}\n\n```tsx\n{path.read_text()}\n```")

    examples_section = "## Real Component Examples\n\n" + "\n\n".join(examples) if examples else ""

    return (
        base
        .replace("{API_REFERENCE}", f"## Allowed Imports (API Reference)\n\n{api_reference}")
        .replace("{EXAMPLES}", examples_section)
    )


@activity.defn(name="vfx_generate_infographic_code")
def generate_infographic_code(input_data: dict) -> dict:
    """LLM generates TSX source code for ONE infographic.

    Input: {
        "spec": dict,                # InfographicSpec
        "style_config": dict | None,
        "video_info": dict,          # {width, height, fps, duration}
        "attempt": int,              # 1-based attempt number
        "previous_errors": list[str],
        "previous_code": str,
    }
    Output: {
        "component_id": str,
        "tsx_code": str,
        "export_name": str,
        "props": dict,
    }
    """
    spec = input_data["spec"]
    style_config = input_data.get("style_config")
    video_info = input_data.get("video_info", {})
    attempt = input_data.get("attempt", 1)
    previous_errors = input_data.get("previous_errors", [])
    previous_code = input_data.get("previous_code", "")

    api_reference = load_prompt("infographic_api_reference.md")
    system_prompt = _build_codegen_prompt(api_reference)

    # Build user message
    component_id = spec["id"]
    # PascalCase export name from id — must be a valid JS identifier
    export_name = "".join(word.capitalize() for word in component_id.replace("-", "_").split("_"))
    if export_name and export_name[0].isdigit():
        export_name = "Ig" + export_name

    lines = [
        f"## Infographic Spec",
        f"- Component ID: `{component_id}`",
        f"- Export name: `{export_name}`",
        f"- Type: {spec['type']}",
        f"- Title: {spec['title']}",
        f"- Description: {spec['description']}",
        f"- Duration: {spec['end_time'] - spec['start_time']:.1f}s",
        f"- Video: {video_info.get('width', 1920)}x{video_info.get('height', 1080)} @ {video_info.get('fps', 30)}fps",
        f"",
        f"## Data to Visualize",
        f"```json",
        json.dumps(spec.get("data", {}), indent=2),
        f"```",
    ]

    if style_config:
        palette = style_config.get("palette", [])
        if palette:
            lines.append(f"\n## Style")
            lines.append(f"Use useStyle() for colors, but for reference the palette is: {palette}")

    if attempt > 1 and previous_errors:
        lines.append(f"\n## RETRY (attempt {attempt})")
        lines.append(f"The previous code had these errors:")
        for err in previous_errors:
            lines.append(f"- {err}")
        if previous_code:
            lines.append(f"\n## Previous Code (fix these issues)\n\n```tsx\n{previous_code}\n```")

    user_message = "\n".join(lines)

    activity.heartbeat(f"Generating code for {component_id} (attempt {attempt})")

    raw_code = call_text(
        system_prompt=system_prompt,
        user_message=user_message,
        model=settings.INFOGRAPHIC_LLM_MODEL,
    )

    # Strip markdown fencing if present
    tsx_code = raw_code.strip()
    if tsx_code.startswith("```"):
        # Remove opening fence
        tsx_code = re.sub(r"^```\w*\n?", "", tsx_code)
        # Remove closing fence
        tsx_code = re.sub(r"\n?```$", "", tsx_code)

    # Build props dict from spec data (what gets passed at render time)
    props = dict(spec.get("data", {}))

    return {
        "component_id": component_id,
        "tsx_code": tsx_code,
        "export_name": export_name,
        "props": props,
    }


# ---------------------------------------------------------------------------
# A3: Validate infographic
# ---------------------------------------------------------------------------


@activity.defn(name="vfx_validate_infographic")
def validate_infographic(input_data: dict) -> dict:
    """Write TSX to disk, type-check, and test render.

    Input: {
        "component_id": str,
        "tsx_code": str,
        "export_name": str,
    }
    Output: {
        "valid": bool,
        "errors": list[str],
        "preview_path": str,
    }
    """
    component_id = input_data["component_id"]
    tsx_code = input_data["tsx_code"]
    export_name = input_data["export_name"]

    remotion_dir = _get_remotion_dir()
    generated_dir = remotion_dir / "src" / "components" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    # Write component file
    component_path = generated_dir / f"{component_id}.tsx"
    component_path.write_text(tsx_code)

    errors: list[str] = []

    # Step 1: TypeScript type-check
    activity.heartbeat(f"Type-checking {component_id}")
    try:
        tsc_result = subprocess.run(
            ["npx", "tsc", "--noEmit", "--pretty", "false"],
            cwd=str(remotion_dir),
            capture_output=True,
            text=True,
            timeout=60,
        )
        if tsc_result.returncode != 0:
            # Filter errors to only those from our generated file
            stderr = tsc_result.stdout + tsc_result.stderr
            for line in stderr.split("\n"):
                if component_id in line or f"generated/{component_id}" in line:
                    errors.append(line.strip())
            if not errors and tsc_result.returncode != 0:
                # Include all errors if none matched our file
                all_errors = [l.strip() for l in stderr.split("\n") if l.strip() and "error TS" in l]
                errors = all_errors[:10]
    except subprocess.TimeoutExpired:
        errors.append("TypeScript type-check timed out after 60 seconds")
    except FileNotFoundError:
        errors.append("npx/tsc not found — cannot type-check")

    if errors:
        logger.warning("Type-check failed for %s: %d errors", component_id, len(errors))
        # Clean up the broken file
        component_path.unlink(missing_ok=True)
        return {"valid": False, "errors": errors, "preview_path": ""}

    # Step 2: Test render a still frame
    activity.heartbeat(f"Test-rendering {component_id}")
    preview_path = str(generated_dir / f"{component_id}_preview.png")

    # Build a minimal registry so we can render just this component
    _write_temp_registry(generated_dir, component_id, export_name)

    try:
        test_plan = {
            "components": [{
                "template": component_id,
                "startFrame": 0,
                "durationInFrames": 90,
                "props": {"position": {"x": 0.1, "y": 0.1, "w": 0.4, "h": 0.3}},
                "bounds": {"x": 0.1, "y": 0.1, "w": 0.4, "h": 0.3},
                "zIndex": 1,
            }],
            "colorPalette": [],
            "includeBaseVideo": False,
        }

        render_still(
            composition_id="MotionOverlay",
            frame=30,
            props=test_plan,
            output_path=preview_path,
        )
    except Exception as e:
        err_msg = str(e)
        # Truncate long error messages
        if len(err_msg) > 500:
            err_msg = err_msg[:500] + "..."
        errors.append(f"Render test failed: {err_msg}")
        component_path.unlink(missing_ok=True)
        _cleanup_registry(generated_dir)
        return {"valid": False, "errors": errors, "preview_path": ""}

    logger.info("Validation passed for %s", component_id)
    return {"valid": True, "errors": [], "preview_path": preview_path}


def _write_temp_registry(generated_dir: Path, component_id: str, export_name: str) -> None:
    """Write a temporary _registry.ts with just the component being tested."""
    # Check if registry already exists and has other components
    registry_path = generated_dir / "_registry.ts"
    registry_path.write_text(
        f'import React from "react";\n'
        f'import {{ {export_name} }} from "./{component_id}";\n'
        f"\n"
        f"type ComponentMap = {{ [key: string]: React.FC<any> }};\n"
        f"\n"
        f"export const GeneratedRegistry: ComponentMap = {{\n"
        f'  "{component_id}": {export_name} as React.FC<any>,\n'
        f"}};\n"
    )


def _cleanup_registry(generated_dir: Path) -> None:
    """Remove the temporary registry file."""
    registry_path = generated_dir / "_registry.ts"
    registry_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# A4: Build generated registry
# ---------------------------------------------------------------------------


@activity.defn(name="vfx_build_generated_registry")
def build_generated_registry(input_data: dict) -> dict:
    """Write the final auto-generated registry and convert specs to frame-based ComponentSpec list.

    Input: {
        "generated_components": list[dict],  # [{component_id, export_name, spec, props}]
        "video_fps": int,
        "style_config": dict | None,
    }
    Output: {
        "components": list[dict],  # Remotion-ready ComponentSpec dicts
        "registry_path": str,
    }
    """
    components_data = input_data["generated_components"]
    fps = input_data.get("video_fps", 30)
    style_config = input_data.get("style_config")

    if not components_data:
        return {"components": [], "registry_path": ""}

    remotion_dir = _get_remotion_dir()
    generated_dir = remotion_dir / "src" / "components" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    # Build registry file
    imports = []
    entries = []
    for comp in components_data:
        cid = comp["component_id"]
        ename = comp["export_name"]
        imports.append(f'import {{ {ename} }} from "./{cid}";')
        entries.append(f'  "{cid}": {ename} as React.FC<any>,')

    registry_code = (
        'import React from "react";\n'
        + "\n".join(imports) + "\n"
        "\n"
        "type ComponentMap = { [key: string]: React.FC<any> };\n"
        "\n"
        "export const GeneratedRegistry: ComponentMap = {\n"
        + "\n".join(entries) + "\n"
        "};\n"
    )

    registry_path = generated_dir / "_registry.ts"
    registry_path.write_text(registry_code)
    logger.info("Wrote generated registry with %d components", len(components_data))

    # Convert time-based specs to frame-based Remotion ComponentSpec dicts
    remotion_components = []
    for comp in components_data:
        spec = comp["spec"]
        start_frame = round(spec["start_time"] * fps)
        end_frame = round(spec["end_time"] * fps)
        duration_frames = max(1, end_frame - start_frame)

        remotion_components.append({
            "template": comp["component_id"],
            "startFrame": start_frame,
            "durationInFrames": duration_frames,
            "props": comp["props"],
            "bounds": spec.get("bounds", {"x": 0.1, "y": 0.1, "w": 0.35, "h": 0.3}),
            "zIndex": 10,  # generated components render above templates
            "anchor": spec.get("anchor", "static"),
        })

    return {
        "components": remotion_components,
        "registry_path": str(registry_path),
    }
