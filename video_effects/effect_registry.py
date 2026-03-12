from video_effects.schemas.effects import EffectType

# Phase ordering — lower numbers execute first.
# Gaps allow inserting new phases without renumbering.
EFFECT_PHASES: dict[EffectType, int] = {
    EffectType.VIGNETTE: 5,       # Phase 0.5: vignette (under color grading)
    EffectType.COLOR_CHANGE: 10,  # Phase 1: color grading (pixel-level, no geometry)
    EffectType.BLUR: 20,          # Phase 2: blur regions (before overlays)
    EffectType.WHIP: 25,          # Phase 2.5: whip transitions
    EffectType.ZOOM: 30,          # Phase 3: zoom/crop (geometric transform)
    EffectType.SPEED_RAMP: 50,    # Phase 5: speed ramp (visual, runs last)
}


def get_phase(effect_type: EffectType) -> int:
    """Get the execution phase for an effect type."""
    return EFFECT_PHASES[effect_type]


def get_sorted_phases(effect_types: list[EffectType]) -> list[int]:
    """Get unique phases in execution order."""
    return sorted({EFFECT_PHASES[t] for t in effect_types})


def group_by_phase(effects: list) -> dict[int, list]:
    """Group EffectCue objects by their execution phase."""
    groups: dict[int, list] = {}
    for effect in effects:
        phase = EFFECT_PHASES[effect.effect_type]
        groups.setdefault(phase, []).append(effect)
    return dict(sorted(groups.items()))
