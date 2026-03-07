from .video import get_video_info, extract_audio
from .transcribe import transcribe_audio
from .parse_cues import parse_effect_cues
from .validate import validate_timeline
from .apply_effects import apply_effects, prepare_render, setup_processors, render_video
from .compose import compose_final
from .remotion import (
    build_remotion_context,
    plan_motion_graphics,
    load_composition_plan,
    render_motion_overlay,
    composite_motion_graphics,
    preview_motion_graphics,
)
from .detect_jump_cuts import detect_jump_cuts

ALL_VIDEO_EFFECTS_ACTIVITIES = [
    get_video_info,
    extract_audio,
    transcribe_audio,
    parse_effect_cues,
    validate_timeline,
    apply_effects,  # legacy, remove later
    prepare_render,
    setup_processors,
    render_video,
    compose_final,
    build_remotion_context,
    plan_motion_graphics,
    load_composition_plan,
    render_motion_overlay,
    composite_motion_graphics,
    preview_motion_graphics,
    detect_jump_cuts,
]
