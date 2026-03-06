import json
import logging
import os
import subprocess
from pathlib import Path

from video_effects.config import settings

logger = logging.getLogger(__name__)

REMOTION_DIR = Path(__file__).resolve().parent.parent / "remotion"


def _get_remotion_dir() -> Path:
    custom = getattr(settings, "REMOTION_DIR", None)
    if custom:
        return Path(custom)
    return REMOTION_DIR


def render_still(
    composition_id: str,
    frame: int,
    props: dict,
    output_path: str,
) -> str:
    """Render a single frame as PNG via npx remotion still."""
    remotion_dir = _get_remotion_dir()
    props_json = json.dumps(props, separators=(",", ":"))

    cmd = [
        "npx", "remotion", "still",
        composition_id, output_path,
        "--frame", str(frame),
        "--props", props_json,
    ]

    logger.info("Rendering still frame %d -> %s", frame, output_path)
    result = subprocess.run(
        cmd,
        cwd=str(remotion_dir),
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"remotion still failed (exit {result.returncode}): {result.stderr}"
        )

    logger.info("Still rendered: %s", output_path)
    return output_path


def render_media(
    composition_id: str,
    props: dict,
    output_path: str,
    *,
    width: int | None = None,
    height: int | None = None,
    fps: int | None = None,
    duration_in_frames: int | None = None,
    codec: str = "prores",
    prores_profile: str = "4444",
    concurrency: int | None = None,
) -> str:
    """Render video via npx remotion render with transparent ProRes 4444."""
    remotion_dir = _get_remotion_dir()
    props_json = json.dumps(props, separators=(",", ":"))

    cmd = [
        "npx", "remotion", "render",
        composition_id, output_path,
        "--codec", codec,
        "--prores-profile", prores_profile,
        "--image-format", "png",
        "--pixel-format", "yuva444p10le",
        "--props", props_json,
    ]

    if width and height:
        cmd.extend(["--width", str(width), "--height", str(height)])
    if fps:
        cmd.extend(["--fps", str(fps)])

    conc = concurrency or getattr(settings, "REMOTION_CONCURRENCY", None)
    if conc:
        cmd.extend(["--concurrency", str(conc)])

    logger.info("Rendering media -> %s", output_path)
    result = subprocess.run(
        cmd,
        cwd=str(remotion_dir),
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"remotion render failed (exit {result.returncode}): {result.stderr}"
        )

    logger.info("Media rendered: %s (%s bytes)", output_path, os.path.getsize(output_path))
    return output_path


def composite_overlay(
    base_video: str,
    overlay_video: str,
    output_path: str,
    *,
    copy_audio: bool = True,
) -> str:
    """Alpha-composite a transparent overlay onto a base video with FFmpeg."""
    filter_complex = (
        "[1:v]premultiply=inplace=1[ovr];"
        "[0:v][ovr]overlay=0:0:shortest=1"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", base_video,
        "-i", overlay_video,
        "-filter_complex", filter_complex,
        "-c:v", "libx264",
        "-crf", "16",
    ]

    if copy_audio:
        cmd.extend(["-c:a", "copy"])
    else:
        cmd.extend(["-an"])

    cmd.append(output_path)

    logger.info("Compositing %s + %s -> %s", base_video, overlay_video, output_path)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode != 0:
        # Log last 20 lines of stderr for debugging
        stderr_lines = result.stderr.strip().split("\n")
        err_tail = "\n".join(stderr_lines[-20:])
        raise RuntimeError(
            f"FFmpeg composite failed (exit {result.returncode}):\n{err_tail}"
        )

    logger.info("Composite done: %s", output_path)
    return output_path
