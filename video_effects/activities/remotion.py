"""Activities for Remotion motion graphics overlay rendering."""

import json
import logging
import math
import os

from temporalio import activity

from video_effects.helpers.llm import call_structured, load_prompt
from video_effects.helpers.remotion import composite_overlay, render_media, render_still
from video_effects.prompts.motion_graphics_schema import MotionGraphicsPlanResponse

logger = logging.getLogger(__name__)


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
            face_data = json.load(f)
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
    feedback = input_data.get("feedback", "")
    fps = input_data.get("video_fps", 30)

    system_prompt = load_prompt("plan_motion_graphics.md")

    # Build user message
    lines = []

    if feedback:
        lines.append("## IMPORTANT: Previous plan was rejected by the user")
        lines.append(f"Feedback: {feedback}")
        lines.append("Please adjust your plan based on this feedback.\n")

    video = context.get("video", {})
    lines.append(f"## Video Info")
    lines.append(f"- Resolution: {video.get('width', '?')}x{video.get('height', '?')}")
    lines.append(f"- Duration: {video.get('duration', 0):.1f}s")
    lines.append(f"- FPS: {video.get('fps', 30)}\n")

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

    # Validate and fix
    validated, issues = _validate_plan(components, context, fps)

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
        })

    composition_plan = {
        "components": remotion_components,
        "colorPalette": raw_plan.get("color_palette", []),
        "includeBaseVideo": False,
        "faceDataPath": context.get("face_data_path", ""),
    }

    return {
        "composition_plan": composition_plan,
        "raw_plan": raw_plan,
        "validation_issues": issues,
    }


def _validate_plan(
    components: list[dict],
    context: dict,
    fps: float,
) -> tuple[list[dict], list[str]]:
    """Validate and fix motion graphics plan.

    Rules:
    1. Max 2 concurrent components (drop lowest z_index if exceeded)
    2. No spatial overlap between simultaneous components
    3. Zoom viewport: components during zoom must be within inner frame
    """
    issues: list[str] = []
    if not components:
        return components, issues

    duration = context.get("video", {}).get("duration", 999)

    # Clamp times to video duration
    for comp in components:
        if comp["end_time"] > duration:
            issues.append(f"Clamped {comp['template']} end_time from {comp['end_time']:.1f} to {duration:.1f}")
            comp["end_time"] = duration
        if comp["start_time"] < 0:
            comp["start_time"] = 0
        if comp["end_time"] <= comp["start_time"]:
            comp["end_time"] = comp["start_time"] + 0.5

    # Check max 2 concurrent (excluding progress_bar)
    non_bar = [c for c in components if c["template"] != "progress_bar"]
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

    # Check zoom viewport coordination
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

    logger.info("Validation: %d components kept, %d issues", len(components), len(issues))
    return components, issues


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
