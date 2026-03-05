"""Activity: validate and resolve conflicts in effect timeline."""

import logging

from temporalio import activity

from video_effects.effect_registry import EFFECT_PHASES
from video_effects.schemas.effects import EffectCue, EffectType, ValidatedTimeline

logger = logging.getLogger(__name__)


@activity.defn(name="vfx_validate_timeline")
def validate_timeline(input_data: dict) -> dict:
    """Validate effect timeline and resolve conflicts.

    Conflicts: same region, same time, same phase → keep higher confidence.
    Cross-phase conflicts don't exist (phases are sequential).

    Input: {"effects": list[dict], "duration": float}
    Output: ValidatedTimeline as dict
    """
    raw_effects = [EffectCue(**e) for e in input_data["effects"]]
    duration = input_data.get("duration", 0)

    # 1. Clamp times to video duration
    for effect in raw_effects:
        effect.start_time = max(0, effect.start_time)
        if duration > 0:
            effect.end_time = min(duration, effect.end_time)
        if effect.end_time <= effect.start_time:
            effect.end_time = effect.start_time + 1.0  # minimum 1s duration

    # 2. Remove low-confidence effects
    effects = [e for e in raw_effects if e.confidence >= 0.3]
    removed_low_conf = len(raw_effects) - len(effects)
    if removed_low_conf:
        logger.info(f"Removed {removed_low_conf} low-confidence effects")

    # 3. Resolve same-phase conflicts (same type + overlapping time)
    conflicts_resolved = 0
    resolved = []

    # Group by effect type
    by_type: dict[EffectType, list[EffectCue]] = {}
    for e in effects:
        by_type.setdefault(e.effect_type, []).append(e)

    for effect_type, type_effects in by_type.items():
        # Sort by start time
        type_effects.sort(key=lambda e: e.start_time)

        # Check for overlapping effects of the same type
        kept: list[EffectCue] = []
        for effect in type_effects:
            overlapping = [
                k for k in kept
                if k.start_time < effect.end_time and effect.start_time < k.end_time
            ]
            if overlapping:
                # Keep the one with higher confidence
                for overlap in overlapping:
                    if effect.confidence > overlap.confidence:
                        kept.remove(overlap)
                        kept.append(effect)
                        conflicts_resolved += 1
                    else:
                        conflicts_resolved += 1
                        # Skip this effect, keep existing
            else:
                kept.append(effect)

        resolved.extend(kept)

    # 4. Validate zoom in/out pairs
    resolved = _validate_zoom_pairs(resolved)

    # 5. Sort by phase then start time for ordered execution
    resolved.sort(key=lambda e: (EFFECT_PHASES.get(e.effect_type, 99), e.start_time))

    if conflicts_resolved:
        logger.info(f"Resolved {conflicts_resolved} timeline conflicts")

    timeline = ValidatedTimeline(
        effects=resolved,
        conflicts_resolved=conflicts_resolved,
        total_duration=duration,
    )
    return timeline.model_dump()


def _validate_zoom_pairs(effects: list[EffectCue]) -> list[EffectCue]:
    """Validate and fix zoom in/out pairing.

    - Drop orphaned "out" cues (no prior "in")
    - Drop duplicate "in" cues (already zoomed in)
    - Propagate zoom_level from paired "in" to "out" cue
    """
    zoom_cues = [
        e for e in effects
        if e.effect_type == EffectType.ZOOM and e.zoom_params is not None
    ]
    non_zoom = [e for e in effects if e.effect_type != EffectType.ZOOM or e.zoom_params is None]

    zoom_cues.sort(key=lambda e: e.start_time)

    kept = []
    zoomed_in = False
    last_zoom = 1.5
    dropped = 0

    for cue in zoom_cues:
        action = cue.zoom_params.action
        if action == "out":
            if not zoomed_in:
                logger.warning(
                    f"Dropping orphaned zoom-out at t={cue.start_time:.1f}s (no prior zoom-in)"
                )
                dropped += 1
                continue
            cue.zoom_params.zoom_level = last_zoom
            zoomed_in = False
        elif action == "in":
            if zoomed_in:
                logger.warning(
                    f"Dropping duplicate zoom-in at t={cue.start_time:.1f}s (already zoomed in)"
                )
                dropped += 1
                continue
            zoomed_in = True
            last_zoom = cue.zoom_params.zoom_level
        kept.append(cue)

    if dropped:
        logger.info(f"Zoom pair validation: dropped {dropped} invalid cues")

    return non_zoom + kept
