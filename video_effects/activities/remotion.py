"""Activities for Remotion motion graphics overlay rendering."""

import json
import logging
import math
import os
from pathlib import Path

from temporalio import activity

from video_effects.helpers.llm import call_structured, load_prompt
from video_effects.helpers.remotion import composite_overlay, render_media, render_still
from video_effects.prompts.motion_graphics_schema import MotionGraphicsPlanResponse
from video_effects.schemas.mg_templates import (
    MGTemplateSpec,
    MG_TEMPLATE_REGISTRY,
    get_available_templates,
    load_guidance,
)
from video_effects.schemas.styles import StyleConfig, StylePreset, get_style

logger = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


# ---------------------------------------------------------------------------
# Dynamic prompt assembly
# ---------------------------------------------------------------------------


def _render_template_section(spec: MGTemplateSpec) -> str:
    """Format a single template's metadata + guidance into a markdown section."""
    lines = [f"### {spec.name}", spec.description, ""]

    # Props table
    lines.append("| Prop | Type | Required | Default | Constraints |")
    lines.append("|------|------|----------|---------|-------------|")
    for p in spec.props:
        constraints = ""
        if p.choices:
            constraints = " | ".join(p.choices)
        elif p.min_value is not None or p.max_value is not None:
            lo = p.min_value if p.min_value is not None else ""
            hi = p.max_value if p.max_value is not None else ""
            constraints = f"{lo}-{hi}"
        lines.append(
            f"| `{p.name}` | {p.type} | {'yes' if p.required else 'no'} "
            f"| {p.default if p.default is not None else '-'} | {constraints} |"
        )
    lines.append("")

    # Duration + spatial hints
    lines.append(f"- **Duration**: {spec.duration_range[0]}-{spec.duration_range[1]} seconds")
    sy = spec.spatial
    lines.append(
        f"- **Typical placement**: y {sy.typical_y_range[0]:.0%}-{sy.typical_y_range[1]:.0%}, "
        f"x {sy.typical_x_range[0]:.0%}-{sy.typical_x_range[1]:.0%}"
        + (" (edge-aligned)" if sy.edge_aligned else "")
    )
    lines.append("")

    # Creative guidance
    guidance = load_guidance(spec)
    if guidance:
        lines.append(guidance)
        lines.append("")

    return "\n".join(lines)


def _build_style_guide(style_config: dict | None, style_preset_name: str = "") -> str:
    """Build the style guide section for the MG planner prompt."""
    if not style_config and not style_preset_name:
        return ""

    preset = get_style(style_preset_name) if style_preset_name else None

    palette = (style_config or {}).get("palette", [])
    palette_desc = ""
    if len(palette) >= 3:
        palette_desc = f"{palette[0]} (text), {palette[1]} (secondary), {palette[2]} (accent)"
    elif palette:
        palette_desc = ", ".join(palette)

    lines = ["## Style Guide\n"]
    if preset and preset.name != "default":
        lines.append(f"**Style: {preset.display_name}**\n")

    if palette_desc:
        lines.append(f"**Color palette**: Use ONLY these colors: {palette_desc}")
    if preset and preset.name != "default":
        if preset.preferred_animations:
            lines.append(f"**Animations**: Prefer {', '.join(preset.preferred_animations)}.")
        if preset.avoided_animations:
            lines.append(f"NEVER use {', '.join(preset.avoided_animations)}.")
        lo, hi = preset.density_range
        lines.append(f"**Density**: {preset.density_label} — target {lo}-{hi} overlays per 60 seconds.")
        if preset.template_preferences:
            lines.append(f"**Preferred templates**: {', '.join(preset.template_preferences)}")
        if preset.preferred_effects:
            lines.append(f"**Preferred effects**: Use {', '.join(preset.preferred_effects)} when the content warrants it.")
        if preset.avoided_effects:
            lines.append(f"**Avoid**: Do NOT use {', '.join(preset.avoided_effects)} unless explicitly requested.")

    lines.append("")
    return "\n".join(lines)


def build_mg_system_prompt(style_config: dict | None = None, style_preset_name: str = "") -> str:
    """Assemble the motion-graphics planner prompt from base + style + templates."""
    base = (_PROMPT_DIR / "plan_motion_graphics_base.md").read_text()

    style_guide = _build_style_guide(style_config, style_preset_name)

    templates = get_available_templates()
    if not templates:
        section = "## Available Templates\n\nNo templates are currently implemented.\n"
    else:
        parts = ["## Available Templates\n"]
        parts.append(f"You may ONLY use the following {len(templates)} template(s).\n")
        for spec in templates:
            parts.append(_render_template_section(spec))
        section = "\n".join(parts)

    return base.replace("{STYLE_GUIDE}", style_guide).replace("{TEMPLATES}", section)


def _validate_props(
    components: list[dict],
) -> tuple[list[dict], list[str]]:
    """Validate and fix component props against the template registry.

    - Drops components whose template is not implemented
    - Checks required props are present
    - Applies defaults for missing optional props
    - Clamps numeric values to declared ranges
    - Validates literal choices
    """
    issues: list[str] = []
    validated: list[dict] = []

    for comp in components:
        tpl_name = comp.get("template", "")
        spec = MG_TEMPLATE_REGISTRY.get(tpl_name)

        if spec is None:
            issues.append(f"Dropped unknown template '{tpl_name}' — not in registry")
            continue

        raw_props = comp.get("props", {})
        if not isinstance(raw_props, dict):
            issues.append(
                f"[{tpl_name}] Expected props object, got {type(raw_props).__name__} — using defaults"
            )
            raw_props = {}
        props = dict(raw_props)

        for p in spec.props:
            if p.name not in props:
                if p.required:
                    issues.append(
                        f"[{tpl_name}] Missing required prop '{p.name}' — skipping component"
                    )
                    break
                if p.default is not None:
                    props[p.name] = p.default
            else:
                val = props[p.name]

                # Clamp numeric ranges
                if p.type in ("int", "float") and isinstance(val, (int, float)):
                    if p.min_value is not None and val < p.min_value:
                        issues.append(
                            f"[{tpl_name}] Clamped {p.name} from {val} to {p.min_value}"
                        )
                        val = p.min_value
                    if p.max_value is not None and val > p.max_value:
                        issues.append(
                            f"[{tpl_name}] Clamped {p.name} from {val} to {p.max_value}"
                        )
                        val = p.max_value
                    if p.type == "int":
                        val = int(val)
                    props[p.name] = val

                # Validate list type (list of strings, max_value = max length)
                if p.type == "list":
                    if isinstance(val, list):
                        max_len = int(p.max_value) if p.max_value is not None else len(val)
                        if len(val) > max_len:
                            issues.append(
                                f"[{tpl_name}] Truncated {p.name} from {len(val)} to {max_len} items"
                            )
                            val = val[:max_len]
                        props[p.name] = val
                    else:
                        issues.append(f"[{tpl_name}] Expected list for {p.name}, got {type(val).__name__}")

                # Validate json type (list of objects, skip deeper validation)
                if p.type == "json":
                    if not isinstance(val, list):
                        issues.append(f"[{tpl_name}] Expected list for {p.name}, got {type(val).__name__}")

                # Validate literal choices
                if p.choices and val not in p.choices:
                    issues.append(
                        f"[{tpl_name}] Invalid {p.name}='{val}', "
                        f"expected one of {p.choices} — using default '{p.default}'"
                    )
                    props[p.name] = p.default if p.default is not None else p.choices[0]
        else:
            # Only reached if inner loop didn't break (all required props present)
            comp["props"] = props
            validated.append(comp)

    return validated, issues


# ---------------------------------------------------------------------------
# G8a: Build Remotion context (face tracking + transcript + effects)
# ---------------------------------------------------------------------------

@activity.defn(name="vfx_build_remotion_context")
def build_remotion_context(input_data: dict) -> dict:
    """Build spatial context for Remotion motion graphics planning.

    Loads face tracking cache, computes time-windowed safe zones,
    packages everything for the LLM planner.

    Input: {
        "video_info": dict,
        "transcript": str,
        "segments": list[dict],
        "effects": list[dict],       # validated EffectCue dicts from G4
        "cache_dir": str,             # where face_tracking_zoom.json lives
    }
    Output: spatial_context dict (see SpatialContext schema)
    """
    video_info = input_data["video_info"]
    transcript = input_data["transcript"]
    segments = input_data["segments"]
    effects = input_data["effects"]
    cache_dir = input_data["cache_dir"]

    width = video_info.get("width", 1920)
    height = video_info.get("height", 1080)
    fps = video_info.get("fps", 30)
    duration = video_info.get("duration", 0)
    total_frames = video_info.get("total_frames", 0) or int(duration * fps)

    # Load face tracking data if available
    face_data_path = os.path.join(cache_dir, "face_tracking_zoom.json")
    face_data = None
    if os.path.exists(face_data_path):
        with open(face_data_path) as f:
            raw = json.load(f)
        # Handle both new dict format and old list format
        if isinstance(raw, dict):
            face_data = raw.get("face_data", [])
        else:
            face_data = raw
        logger.info("Loaded face tracking data: %d frames", len(face_data))

    # Compute face windows (time-windowed averages)
    face_windows = []
    if face_data and total_frames > 0:
        window_seconds = 3.0
        window_frames = int(window_seconds * fps)
        for start_frame in range(0, total_frames, window_frames):
            end_frame = min(start_frame + window_frames - 1, total_frames - 1)
            window_data = face_data[start_frame:end_frame + 1]
            if not window_data:
                continue

            # Average face position in this window (normalized 0-1)
            avg_cx = sum(d[0] for d in window_data) / len(window_data) / width
            avg_cy = sum(d[1] for d in window_data) / len(window_data) / height
            avg_fw = sum(d[2] for d in window_data) / len(window_data) / width
            avg_fh = sum(d[3] for d in window_data) / len(window_data) / height

            face_region = {
                "x": max(0, avg_cx - avg_fw / 2),
                "y": max(0, avg_cy - avg_fh / 2),
                "w": avg_fw,
                "h": avg_fh,
            }

            # Compute safe regions (where face is NOT)
            safe_regions = _compute_safe_regions(face_region)

            face_windows.append({
                "start_time": round(start_frame / fps, 2),
                "end_time": round(end_frame / fps, 2),
                "face_region": face_region,
                "safe_regions": safe_regions,
            })

    # Export per-frame zoom state for Remotion zoom compensation
    zoom_state_path = ""
    zoom_effects = [e for e in effects if e.get("effect_type") == "zoom"]
    if zoom_effects:
        from video_effects.effects.zoom import export_zoom_state
        zoom_state_output = os.path.join(cache_dir, "zoom_state.json")
        zoom_state_path = export_zoom_state(
            effects=effects,
            face_data=face_data,
            total_frames=total_frames,
            fps=fps,
            width=width,
            height=height,
            output_path=zoom_state_output,
        )
        if zoom_state_path:
            logger.info("Exported zoom state to %s", zoom_state_path)

    context = {
        "video": {
            "width": width,
            "height": height,
            "fps": fps,
            "duration": duration,
            "total_frames": total_frames,
        },
        "transcript": {
            "full_text": transcript,
            "segments": segments,
        },
        "face_windows": face_windows,
        "opencv_effects": effects,
        "face_data_path": face_data_path if face_data else "",
        "zoom_state_path": zoom_state_path,
    }

    logger.info(
        "Built remotion context: %d face windows, %d effects, %.1fs duration",
        len(face_windows), len(effects), duration,
    )
    return context


def _compute_safe_regions(face: dict) -> list[dict]:
    """Compute screen regions that don't overlap with the face."""
    regions = []
    fx, fy, fw, fh = face["x"], face["y"], face["w"], face["h"]
    face_right = fx + fw
    face_bottom = fy + fh

    # Left of face
    if fx > 0.15:
        regions.append({"x": 0.0, "y": 0.0, "w": round(fx - 0.05, 2), "h": 1.0, "label": "left"})

    # Right of face
    if face_right < 0.85:
        regions.append({"x": round(face_right + 0.05, 2), "y": 0.0,
                        "w": round(1.0 - face_right - 0.05, 2), "h": 1.0, "label": "right"})

    # Below face
    if face_bottom < 0.75:
        regions.append({"x": 0.0, "y": round(face_bottom + 0.05, 2),
                        "w": 1.0, "h": round(1.0 - face_bottom - 0.05, 2), "label": "bottom"})

    # Above face
    if fy > 0.25:
        regions.append({"x": 0.0, "y": 0.0, "w": 1.0, "h": round(fy - 0.05, 2), "label": "top"})

    return regions


# ---------------------------------------------------------------------------
# G8b: LLM plan motion graphics
# ---------------------------------------------------------------------------

@activity.defn(name="vfx_plan_motion_graphics")
def plan_motion_graphics(input_data: dict) -> dict:
    """Use LLM to plan motion graphics overlays.

    Input: {
        "spatial_context": dict,          # output of G8a
        "style_hint": str,                # optional style preference
        "feedback": str,                  # optional rejection feedback for retry
        "video_fps": float,
    }
    Output: {
        "composition_plan": dict,         # Remotion-ready plan
        "raw_plan": dict,                 # LLM response as-is
        "validation_issues": list[str],
    }
    """
    context = input_data["spatial_context"]
    style_hint = input_data.get("style_hint", "")
    style_config = input_data.get("style_config")
    style_preset_name = input_data.get("style_preset_name", "")
    feedback = input_data.get("feedback", "")
    fps = input_data.get("video_fps", 30)

    system_prompt = build_mg_system_prompt(style_config=style_config, style_preset_name=style_preset_name)

    # Build user message
    lines = []

    if feedback:
        lines.append("## IMPORTANT: Previous plan was rejected by the user")
        lines.append(f"Feedback: {feedback}")
        lines.append("Please adjust your plan based on this feedback.\n")

    video = context.get("video", {})
    vid_duration = video.get('duration', 0)
    lines.append(f"## Video Info")
    lines.append(f"- Resolution: {video.get('width', '?')}x{video.get('height', '?')}")
    lines.append(f"- Duration: {vid_duration:.1f}s")
    lines.append(f"- FPS: {video.get('fps', 30)}")
    lines.append(f"- **All component times must be between 0.0 and {vid_duration:.1f}s**\n")

    if style_hint:
        lines.append(f"## Style Preference\n{style_hint}\n")

    # Face windows summary
    face_windows = context.get("face_windows", [])
    if face_windows:
        lines.append("## Face Position (time windows)")
        for fw in face_windows[:20]:  # cap to avoid huge prompts
            fr = fw["face_region"]
            safe_labels = [s["label"] for s in fw.get("safe_regions", [])]
            lines.append(
                f"- [{fw['start_time']:.1f}s - {fw['end_time']:.1f}s] "
                f"face at ({fr['x']:.2f}, {fr['y']:.2f}, {fr['w']:.2f}x{fr['h']:.2f}) "
                f"safe: {', '.join(safe_labels) or 'none'}"
            )
        lines.append("")

    # OpenCV effects summary
    effects = context.get("opencv_effects", [])
    if effects:
        lines.append("## Existing Video Effects (already applied)")
        for e in effects:
            etype = e.get("effect_type", "?")
            lines.append(f"- {etype}: {e.get('start_time', 0):.1f}s - {e.get('end_time', 0):.1f}s")
        lines.append("")

    # Transcript
    transcript = context.get("transcript", {})
    segments = transcript.get("segments", [])
    if segments:
        lines.append("## Timestamped Transcript")
        for seg in segments:
            if seg.get("type") == "word":
                lines.append(f"[{seg.get('start', 0):.2f}s] {seg.get('text', '')}")
        lines.append("")

    full_text = transcript.get("full_text", "")
    if full_text:
        lines.append(f"## Full Transcript\n\n{full_text}\n")

    user_message = "\n".join(lines)

    activity.heartbeat("Calling LLM for motion graphics plan")

    raw_plan = call_structured(
        system_prompt=system_prompt,
        user_message=user_message,
        response_model=MotionGraphicsPlanResponse,
    )

    components = raw_plan.get("components", [])
    logger.info("LLM planned %d motion graphics components", len(components))

    # Validate props against template registry
    components, prop_issues = _validate_props(components)
    logger.info("Prop validation: %d kept, %d issues", len(components), len(prop_issues))

    # Validate spatial/temporal rules
    validated, issues = _validate_plan(components, context, fps)
    issues = prop_issues + issues

    # Convert to Remotion format (time -> frames)
    remotion_components = []
    for comp in validated:
        start_frame = round(comp["start_time"] * fps)
        end_frame = round(comp["end_time"] * fps)
        duration_frames = max(1, end_frame - start_frame)

        remotion_components.append({
            "template": comp["template"],
            "startFrame": start_frame,
            "durationInFrames": duration_frames,
            "props": comp.get("props", {}),
            "bounds": comp.get("bounds", {"x": 0.1, "y": 0.1, "w": 0.3, "h": 0.1}),
            "zIndex": comp.get("z_index", 0),
            "anchor": comp.get("anchor", "static"),
        })

    composition_plan = {
        "components": remotion_components,
        "colorPalette": raw_plan.get("color_palette", []),
        "includeBaseVideo": False,
        "faceDataPath": context.get("face_data_path", ""),
        "zoomStatePath": context.get("zoom_state_path", ""),
    }

    if style_config:
        composition_plan["styleConfig"] = style_config

    return {
        "composition_plan": composition_plan,
        "raw_plan": raw_plan,
        "validation_issues": issues,
    }


def _rect_overlap_fraction(a: dict, b: dict) -> float:
    """Compute overlap area between two normalized rects as fraction of a's area."""
    ax, ay, aw, ah = a.get("x", 0), a.get("y", 0), a.get("w", 0), a.get("h", 0)
    bx, by, bw, bh = b.get("x", 0), b.get("y", 0), b.get("w", 0), b.get("h", 0)
    a_area = aw * ah
    if a_area <= 0:
        return 0.0
    ox = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    oy = max(0, min(ay + ah, by + bh) - max(ay, by))
    return (ox * oy) / a_area


def _find_best_safe_region(
    comp_w: float, comp_h: float, safe_regions: list[dict]
) -> tuple[float, float, float, float] | None:
    """Pick the safe region with most area that fits the component.

    Returns (x, y, w, h) centered within the chosen region, or None.
    """
    # First try regions that fit the component as-is
    fitting = [r for r in safe_regions if r.get("w", 0) >= comp_w and r.get("h", 0) >= comp_h]
    if fitting:
        best = max(fitting, key=lambda r: r.get("w", 0) * r.get("h", 0))
        cx = best["x"] + (best["w"] - comp_w) / 2
        cy = best["y"] + (best["h"] - comp_h) / 2
        return cx, cy, comp_w, comp_h

    # No region fits — pick the largest and shrink component to fit
    if safe_regions:
        best = max(safe_regions, key=lambda r: r.get("w", 0) * r.get("h", 0))
        new_w = min(comp_w, best.get("w", comp_w))
        new_h = min(comp_h, best.get("h", comp_h))
        cx = best["x"] + (best["w"] - new_w) / 2
        cy = best["y"] + (best["h"] - new_h) / 2
        return cx, cy, new_w, new_h

    return None


def _resolve_spatial_conflicts(
    components: list[dict],
    edge_aligned_templates: set[str],
    issues: list[str],
) -> bool:
    """Single pass of spatial overlap resolution. Returns True if any component was moved."""
    changed = False
    non_edge = [c for c in components if c.get("template", "") not in edge_aligned_templates]
    for i, a in enumerate(non_edge):
        for j, b in enumerate(non_edge):
            if j <= i:
                continue
            # Check time overlap
            if a["start_time"] >= b["end_time"] or b["start_time"] >= a["end_time"]:
                continue
            ab = a.get("bounds", {})
            bb = b.get("bounds", {})
            overlap = _rect_overlap_fraction(ab, bb)
            if overlap <= 0.10:
                continue
            # Shift the lower z_index component
            if a.get("z_index", 0) <= b.get("z_index", 0):
                target, other_bounds = a, bb
            else:
                target, other_bounds = b, ab
            tb = target.get("bounds", {})
            ob_bottom = other_bounds.get("y", 0) + other_bounds.get("h", 0)
            ob_top = other_bounds.get("y", 0)
            # Try below the other component
            if ob_bottom + tb.get("h", 0.1) <= 0.98:
                tb["y"] = round(ob_bottom + 0.02, 3)
            # Try above
            elif ob_top - tb.get("h", 0.1) >= 0.02:
                tb["y"] = round(ob_top - tb.get("h", 0.1) - 0.02, 3)
            # Try opposite horizontal side
            else:
                ob_right = other_bounds.get("x", 0) + other_bounds.get("w", 0)
                if ob_right + tb.get("w", 0.2) <= 0.98:
                    tb["x"] = round(ob_right + 0.02, 3)
                elif other_bounds.get("x", 0) - tb.get("w", 0.2) >= 0.02:
                    tb["x"] = round(other_bounds.get("x", 0) - tb.get("w", 0.2) - 0.02, 3)
            changed = True
            issues.append(
                f"Shifted {target.get('template', '?')} to avoid overlap with "
                f"{(a if target is b else b).get('template', '?')}"
            )
    return changed


def _validate_plan(
    components: list[dict],
    context: dict,
    fps: float,
) -> tuple[list[dict], list[str]]:
    """Validate and fix motion graphics plan.

    Rules:
    1. Hard bounds clamping
    2. Time clamping
    3. Template duration limits
    4. Face overlap relocation
    5. Concurrent count enforcement
    6. Zoom viewport clamping
    7. Zoom transition buffer
    8. Multi-pass spatial conflict resolution (runs last, catches all accumulated conflicts)
    """
    issues: list[str] = []
    if not components:
        return components, issues

    duration = context.get("video", {}).get("duration", 999)

    # 1. Hard bounds clamping — keep components within safe frame
    for comp in components:
        bounds = comp.get("bounds", {})
        bx = bounds.get("x", 0.1)
        by = bounds.get("y", 0.1)
        bw = bounds.get("w", 0.2)
        bh = bounds.get("h", 0.1)

        # Ensure minimum dimensions
        bw = max(bw, 0.05)
        bh = max(bh, 0.03)

        # Clamp origin to safe range
        bx = max(0.02, min(bx, 0.98))
        by = max(0.02, min(by, 0.98))

        # Clamp right/bottom edges
        if bx + bw > 0.98:
            bw = 0.98 - bx
        if by + bh > 0.98:
            bh = 0.98 - by

        if (bx, by, bw, bh) != (bounds.get("x"), bounds.get("y"), bounds.get("w"), bounds.get("h")):
            issues.append(f"Clamped {comp.get('template', '?')} bounds to safe frame")

        bounds["x"] = bx
        bounds["y"] = by
        bounds["w"] = bw
        bounds["h"] = bh

    # 2. Clamp times to video duration
    for comp in components:
        if comp["end_time"] > duration:
            issues.append(f"Clamped {comp['template']} end_time from {comp['end_time']:.1f} to {duration:.1f}")
            comp["end_time"] = duration
        if comp["start_time"] < 0:
            comp["start_time"] = 0
        if comp["end_time"] <= comp["start_time"]:
            comp["end_time"] = comp["start_time"] + 0.5

    # 3. Enforce template duration limits
    for comp in components:
        template_spec = MG_TEMPLATE_REGISTRY.get(comp.get("template", ""))
        if template_spec:
            max_dur = template_spec.duration_range[1]
            actual_dur = comp["end_time"] - comp["start_time"]
            if actual_dur > max_dur:
                issues.append(
                    f"Clamped {comp['template']} duration from {actual_dur:.1f}s to {max_dur:.1f}s"
                )
                comp["end_time"] = comp["start_time"] + max_dur

    # 4. Face overlap relocation — move components away from speaker's face
    face_windows = context.get("face_windows", [])
    for comp in components:
        bounds = comp.get("bounds", {})
        bx, by, bw, bh = bounds.get("x", 0), bounds.get("y", 0), bounds.get("w", 0.2), bounds.get("h", 0.1)
        for fw in face_windows:
            # Check time overlap
            if comp["start_time"] >= fw.get("end_time", 0) or fw.get("start_time", 0) >= comp["end_time"]:
                continue
            face_region = fw.get("face_region", {})
            overlap = _rect_overlap_fraction(bounds, face_region)
            if overlap > 0.05:
                safe_regions = fw.get("safe_regions", [])
                result = _find_best_safe_region(bw, bh, safe_regions)
                if result:
                    new_x, new_y, new_w, new_h = result
                    bounds["x"] = round(new_x, 3)
                    bounds["y"] = round(new_y, 3)
                    bounds["w"] = round(new_w, 3)
                    bounds["h"] = round(new_h, 3)
                    issues.append(
                        f"Relocated {comp.get('template', '?')} away from face at {comp['start_time']:.1f}s"
                    )
                break  # Only need to relocate once per component

    # 5. Check max 2 concurrent (excluding edge-aligned templates like progress_bar)
    edge_aligned_templates = {
        name for name, spec in MG_TEMPLATE_REGISTRY.items()
        if spec.spatial.edge_aligned
    }
    non_bar = [c for c in components if c["template"] not in edge_aligned_templates]
    to_remove = set()
    for i, a in enumerate(non_bar):
        concurrent = []
        for j, b in enumerate(non_bar):
            if i == j:
                continue
            if a["start_time"] < b["end_time"] and b["start_time"] < a["end_time"]:
                concurrent.append(j)
        if len(concurrent) >= 2:
            # Too many concurrent — drop lowest z_index among concurrent
            candidates = sorted(concurrent, key=lambda j: non_bar[j].get("z_index", 0))
            drop_idx = candidates[0]
            if drop_idx not in to_remove:
                to_remove.add(drop_idx)
                issues.append(
                    f"Dropped {non_bar[drop_idx]['template']} at {non_bar[drop_idx]['start_time']:.1f}s "
                    f"(>2 concurrent)"
                )

    if to_remove:
        dropped = {id(non_bar[i]) for i in to_remove}
        components = [c for c in components if id(c) not in dropped]

    # 6. Zoom viewport clamping (before spatial resolution so shifts get caught)
    zoom_effects = [
        e for e in context.get("opencv_effects", [])
        if e.get("effect_type") == "zoom"
    ]
    for comp in components:
        for ze in zoom_effects:
            if comp["start_time"] < ze.get("end_time", 0) and ze.get("start_time", 0) < comp["end_time"]:
                zoom_level = 1.5
                zp = ze.get("zoom_params")
                if zp:
                    zoom_level = zp.get("zoom_level", 1.5)
                inner = 1.0 / zoom_level  # fraction of frame visible
                margin = (1.0 - inner) / 2
                bounds = comp.get("bounds", {})
                bx = bounds.get("x", 0)
                by = bounds.get("y", 0)
                bw = bounds.get("w", 0.2)
                bh = bounds.get("h", 0.1)
                if bx < margin or by < margin or (bx + bw) > (1 - margin) or (by + bh) > (1 - margin):
                    issues.append(
                        f"Adjusted {comp['template']} bounds during {zoom_level}x zoom "
                        f"(viewport inner {inner:.0%})"
                    )
                    bounds["x"] = max(bounds.get("x", 0), margin + 0.02)
                    bounds["y"] = max(bounds.get("y", 0), margin + 0.02)
                    if bounds["x"] + bw > 1 - margin:
                        bounds["w"] = max(0.1, 1 - margin - bounds["x"] - 0.02)
                    if bounds["y"] + bh > 1 - margin:
                        bounds["h"] = max(0.05, 1 - margin - bounds["y"] - 0.02)

    # 7. Zoom transition buffer — shift overlays away from zoom ease-in/ease-out
    for comp in components:
        for ze in zoom_effects:
            if comp["start_time"] >= ze.get("end_time", 0) or ze.get("start_time", 0) >= comp["end_time"]:
                continue
            stable_start, stable_end = _compute_zoom_stable_window(ze)
            stable_dur = stable_end - stable_start
            comp_dur = comp["end_time"] - comp["start_time"]

            # If overlay starts during zoom transition, delay it to stable window
            if comp["start_time"] < stable_start and comp["end_time"] > stable_start:
                comp["start_time"] = stable_start
                issues.append(f"Delayed {comp['template']} to avoid zoom transition")

            # If overlay ends during zoom transition, end it before transition
            if comp["end_time"] > stable_end and comp["start_time"] < stable_end:
                comp["end_time"] = stable_end
                issues.append(f"Trimmed {comp['template']} to avoid zoom transition")

            # If component is entirely during transition (stable window too short), shift outside zoom
            if stable_dur < 1.0 or comp_dur > stable_dur:
                comp["start_time"] = ze.get("end_time", 0) + 0.1
                comp["end_time"] = comp["start_time"] + comp_dur
                issues.append(f"Shifted {comp['template']} past zoom (transition window too short)")

    # 8. Multi-pass spatial conflict resolution (max 3 iterations to converge)
    for pass_num in range(3):
        moved = _resolve_spatial_conflicts(components, edge_aligned_templates, issues)
        if not moved:
            break
        if pass_num > 0:
            issues.append(f"Re-resolved spatial conflicts (pass {pass_num + 1})")

    logger.info("Validation: %d components kept, %d issues", len(components), len(issues))
    return components, issues


def _compute_zoom_stable_window(zoom_cue: dict) -> tuple[float, float]:
    """Return the time range where zoom is stable (not actively transitioning)."""
    start = zoom_cue.get("start_time", 0)
    end = zoom_cue.get("end_time", 0)
    dur = end - start
    if dur <= 0:
        return start, end
    params = zoom_cue.get("zoom_params", {})
    action = params.get("action", "bounce")
    if action == "bounce":
        # sin(pi*t): peaks ~0.3-0.7 range, transitions on edges
        return start + dur * 0.25, start + dur * 0.75
    elif action == "in":
        # Ramps up, stable at end
        return start + dur * 0.6, end
    elif action == "out":
        # Starts zoomed, ramps down
        return start, start + dur * 0.4
    return start, end


@activity.defn(name="vfx_validate_merged_plan")
def validate_merged_plan(input_data: dict) -> dict:
    """Re-validate after merging infographic + MG components.

    Input: {
        "components": list[dict],      # merged components (time-domain, not frame-domain)
        "spatial_context": dict,        # output of G8a
        "video_fps": int,
    }
    Output: {
        "components": list[dict],
        "validation_issues": list[str],
    }
    """
    components = input_data["components"]
    context = input_data["spatial_context"]
    fps = input_data.get("video_fps", 30)
    validated, issues = _validate_plan(components, context, fps)
    if issues:
        logger.info("Post-merge validation: %s", issues)
    return {"components": validated, "validation_issues": issues}


@activity.defn(name="vfx_load_composition_plan")
def load_composition_plan(input_data: dict) -> dict:
    """Load a composition plan JSON file from disk.

    Input: {"plan_path": str}
    Output: {"plan": dict | None}  — None if file doesn't exist or has no components.
    """
    plan_path = input_data["plan_path"]
    if not os.path.exists(plan_path):
        logger.info("No composition_plan.json found at %s", plan_path)
        return {"plan": None}

    with open(plan_path) as f:
        plan = json.load(f)

    if not plan.get("components"):
        logger.info("Empty motion graphics plan at %s", plan_path)
        return {"plan": None}

    return {"plan": plan}


@activity.defn(name="vfx_render_motion_overlay")
def render_motion_overlay(input_data: dict) -> dict:
    """Render transparent motion graphics overlay via Remotion.

    Input: {
        "composition_plan": dict,  # CompositionPlan (components, colorPalette, etc.)
        "output_dir": str,         # directory for overlay .mov
        "video_width": int,
        "video_height": int,
        "video_fps": int,
        "total_frames": int,
    }
    Output: {
        "overlay_path": str,
        "components_rendered": int,
    }
    """
    plan = input_data["composition_plan"]
    output_dir = input_data["output_dir"]
    width = input_data["video_width"]
    height = input_data["video_height"]
    fps = input_data["video_fps"]
    total_frames = input_data["total_frames"]

    os.makedirs(output_dir, exist_ok=True)
    overlay_path = os.path.join(output_dir, "overlay.mov")

    # Ensure overlay-only (no base video baked in)
    plan["includeBaseVideo"] = False
    # Pass video metadata so Remotion's calculateMetadata can size the composition
    plan["durationInFrames"] = total_frames
    plan["fps"] = fps
    plan["width"] = width
    plan["height"] = height

    num_components = len(plan.get("components", []))
    if num_components == 0:
        logger.info("No motion graphics components in plan, skipping render")
        return {"overlay_path": "", "components_rendered": 0}

    activity.heartbeat(f"Rendering {num_components} motion graphics components")

    render_media(
        composition_id="MotionOverlay",
        props=plan,
        output_path=overlay_path,
        width=width,
        height=height,
        fps=fps,
        duration_in_frames=total_frames,
    )

    return {
        "overlay_path": overlay_path,
        "components_rendered": num_components,
    }


@activity.defn(name="vfx_composite_motion_graphics")
def composite_motion_graphics(input_data: dict) -> dict:
    """Composite transparent overlay onto base video.

    If base_video == output_path (FFmpeg can't overwrite its input),
    writes to a temp file first then renames.

    Input: {
        "base_video": str,      # path to base_with_audio.mp4 (G7 output)
        "overlay_video": str,   # path to overlay.mov (G8e output)
        "output_path": str,     # final output path
        "temp_dir": str,        # temp directory for intermediate files
    }
    Output: {"output_video": str}
    """
    import shutil

    base_video = input_data["base_video"]
    overlay_video = input_data["overlay_video"]
    output_path = input_data["output_path"]
    temp_dir = input_data.get("temp_dir", os.path.dirname(output_path))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # FFmpeg can't write to the same file it reads — use temp path if needed
    same_file = os.path.abspath(base_video) == os.path.abspath(output_path)
    actual_output = os.path.join(temp_dir, "mg_composited.mp4") if same_file else output_path

    activity.heartbeat("Compositing motion graphics overlay")

    composite_overlay(
        base_video=base_video,
        overlay_video=overlay_video,
        output_path=actual_output,
    )

    if same_file:
        shutil.move(actual_output, output_path)

    return {"output_video": output_path}


@activity.defn(name="vfx_preview_motion_graphics")
def preview_motion_graphics(input_data: dict) -> dict:
    """Render preview snapshots at key frames.

    Input: {
        "composition_plan": dict,
        "preview_frames": list[int],   # frame numbers to capture
        "output_dir": str,
        "base_video_path": str | None, # if set, include base video in preview
    }
    Output: {
        "snapshots": list[{"frame": int, "path": str}],
    }
    """
    plan = input_data["composition_plan"]
    frames = input_data["preview_frames"]
    output_dir = input_data["output_dir"]
    base_video = input_data.get("base_video_path")

    os.makedirs(output_dir, exist_ok=True)

    # For previews, include base video so user sees full composite
    preview_plan = {**plan, "includeBaseVideo": bool(base_video)}
    if base_video:
        preview_plan["baseVideoPath"] = base_video

    snapshots = []
    for i, frame_num in enumerate(frames):
        output_path = os.path.join(output_dir, f"preview_frame_{frame_num}.png")
        activity.heartbeat(f"Rendering preview {i + 1}/{len(frames)}")

        render_still(
            composition_id="MotionOverlay",
            frame=frame_num,
            props=preview_plan,
            output_path=output_path,
        )
        snapshots.append({"frame": frame_num, "path": output_path})

    return {"snapshots": snapshots}
