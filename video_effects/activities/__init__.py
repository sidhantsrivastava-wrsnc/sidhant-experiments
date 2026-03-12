from .video import get_video_info, extract_audio
from .transcribe import transcribe_audio
from .parse_cues import parse_effect_cues
from .validate import validate_timeline
from .apply_effects import apply_effects, prepare_render, setup_processors, render_video
from .compose import compose_final
from .remotion import (
    detect_faces_activity,
    build_remotion_context,
    plan_motion_graphics,
    validate_merged_plan,
    load_composition_plan,
    render_motion_overlay,
    composite_motion_graphics,
    preview_motion_graphics,
)
from .infographic import (
    cleanup_generated,
    plan_infographics,
    plan_diagrams,
    plan_timelines,
    plan_quotes,
    plan_code_blocks,
    plan_comparisons,
    generate_infographic_code,
    validate_infographic,
    build_generated_registry,
)
from .programmer import (
    programmer_brainstorm,
    programmer_critique,
    programmer_generate_code,
)

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
    detect_faces_activity,
    build_remotion_context,
    plan_motion_graphics,
    validate_merged_plan,
    load_composition_plan,
    render_motion_overlay,
    composite_motion_graphics,
    preview_motion_graphics,
    cleanup_generated,
    plan_infographics,
    plan_diagrams,
    plan_timelines,
    plan_quotes,
    plan_code_blocks,
    plan_comparisons,
    generate_infographic_code,
    validate_infographic,
    build_generated_registry,
    programmer_brainstorm,
    programmer_critique,
    programmer_generate_code,
]
