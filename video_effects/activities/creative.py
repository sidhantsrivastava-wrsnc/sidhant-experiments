"""Activity for the creative designer — auto-selects and customizes style."""

import logging
from pathlib import Path

from temporalio import activity

from video_effects.helpers.llm import call_structured
from video_effects.schemas.styles import (
    StyleConfig,
    StyleDesignResponse,
    get_style,
)

logger = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


@activity.defn(name="vfx_design_style")
def design_style(input_data: dict) -> dict:
    """Pick and customize a style preset based on transcript + video metadata.

    Input: {
        "transcript": str,
        "video_duration": float,
        "video_fps": float,
        "style_override": str,  # if set, skip LLM and return preset directly
    }
    Output: StyleConfig as dict
    """
    style_override = input_data.get("style_override", "")

    if style_override:
        logger.info("Style override: %s", style_override)
        return {"config": get_style(style_override).config.model_dump(), "preset_name": style_override}

    transcript = input_data.get("transcript", "")
    video_duration = input_data.get("video_duration", 30.0)

    system_prompt = (_PROMPT_DIR / "design_style.md").read_text()

    # Truncate transcript to ~2000 chars for the LLM
    excerpt = transcript[:2000]
    if len(transcript) > 2000:
        excerpt += "\n... [truncated]"

    user_message = (
        f"## Video Info\n"
        f"- Duration: {video_duration:.1f}s\n"
        f"- FPS: {input_data.get('video_fps', 30)}\n\n"
        f"## Transcript\n\n{excerpt}\n"
    )

    activity.heartbeat("Calling LLM for style design")

    raw = call_structured(
        system_prompt=system_prompt,
        user_message=user_message,
        response_model=StyleDesignResponse,
    )

    preset_name = raw.get("preset", "default")
    adjustments = raw.get("adjustments", {})
    reasoning = raw.get("reasoning", "")

    logger.info("Creative designer picked preset=%s reason=%s", preset_name, reasoning)

    # Start from the preset config and merge adjustments
    base_config = get_style(preset_name).config.model_dump()

    # Apply adjustments (flat merge on top-level keys)
    for key, value in adjustments.items():
        if key in base_config:
            if key == "font_weights" and isinstance(value, dict):
                base_config["font_weights"].update(value)
            elif key == "palette" and isinstance(value, dict):
                # LLM may return palette as {"accent": "#FF4040", ...}
                # Map named keys onto the base list [text, secondary, accent]
                palette = list(base_config["palette"])  # copy base
                key_map = {"text": 0, "secondary": 1, "accent": 2}
                for pkey, pval in value.items():
                    idx = key_map.get(pkey)
                    if idx is not None and isinstance(pval, str) and pval.startswith("#"):
                        palette[idx] = pval
                base_config["palette"] = palette
            else:
                base_config[key] = value

    return {"config": base_config, "preset_name": preset_name}
