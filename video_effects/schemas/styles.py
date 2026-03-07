"""Style system: presets and configuration for visual consistency."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class FontWeights(BaseModel):
    heading: str = "700"
    body: str = "400"
    emphasis: str = "800"
    marker: str = "700"


class StyleConfig(BaseModel):
    """Serialized and passed to Remotion as styleConfig prop."""

    font_family: str = "sans-serif"
    font_import: str = ""  # @remotion/google-fonts name (e.g. "Inter")
    font_weights_to_load: list[str] = Field(default_factory=lambda: ["400", "700"])
    palette: list[str] = Field(
        default_factory=lambda: ["#FFFFFF", "#000000", "#FFD700"]
    )  # [text, secondary, accent]
    text_shadow: str = "0 2px 8px rgba(0,0,0,0.6)"
    font_weights: FontWeights = Field(default_factory=FontWeights)


class StylePreset(BaseModel):
    """Full preset: rendering config + LLM guidance."""

    name: str
    display_name: str
    description: str
    config: StyleConfig
    # LLM guidance (not sent to Remotion):
    preferred_animations: list[str]
    avoided_animations: list[str]
    density_range: tuple[int, int]
    density_label: str
    template_preferences: list[str]  # preferred templates (empty = no preference)
    color_grading_preset: str  # "warm", "cool", "dramatic", ""
    color_grading_intensity: float  # 0.0-1.0
    preferred_effects: list[str] = []  # e.g. ["whip", "vignette"]
    avoided_effects: list[str] = []    # e.g. ["speed_ramp"]


class StyleDesignResponse(BaseModel):
    """Structured output from the creative designer LLM."""

    preset: str
    adjustments: dict[str, Any] = Field(default_factory=dict)
    reasoning: str


# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------

STYLE_PRESETS: dict[str, StylePreset] = {
    "default": StylePreset(
        name="default",
        display_name="Default",
        description="Balanced defaults — works for any content",
        config=StyleConfig(),
        preferred_animations=[],
        avoided_animations=[],
        density_range=(3, 6),
        density_label="moderate",
        template_preferences=[],
        color_grading_preset="",
        color_grading_intensity=0.0,
    ),
    "clean-minimal": StylePreset(
        name="clean-minimal",
        display_name="Clean Minimal",
        description="Elegant, restrained — subtle fades, warm tones, sparse overlays",
        config=StyleConfig(
            font_family="'Inter', sans-serif",
            font_import="Inter",
            font_weights_to_load=["400", "600", "700"],
            palette=["#F5F0EB", "#2D2D2D", "#C49A6C"],
            text_shadow="0 1px 4px rgba(0,0,0,0.3)",
            font_weights=FontWeights(heading="600", body="400", emphasis="700", marker="600"),
        ),
        preferred_animations=["fade", "slide-in"],
        avoided_animations=["bounce", "pop"],
        density_range=(3, 4),
        density_label="sparse",
        template_preferences=[],
        color_grading_preset="warm",
        color_grading_intensity=0.3,
        preferred_effects=["vignette"],
        avoided_effects=["whip", "speed_ramp"],
    ),
    "bold-energy": StylePreset(
        name="bold-energy",
        display_name="Bold Energy",
        description="High-impact — heavy type, neon accents, bouncy animations",
        config=StyleConfig(
            font_family="'Bebas Neue', sans-serif",
            font_import="BebasNeue",
            font_weights_to_load=["400"],
            palette=["#FFFFFF", "#000000", "#39FF14"],
            text_shadow="0 4px 12px rgba(0,0,0,0.8)",
            font_weights=FontWeights(heading="400", body="400", emphasis="400", marker="400"),
        ),
        preferred_animations=["bounce", "pop"],
        avoided_animations=["fade"],
        density_range=(6, 10),
        density_label="high",
        template_preferences=[],
        color_grading_preset="dramatic",
        color_grading_intensity=0.6,
        preferred_effects=["whip", "speed_ramp"],
        avoided_effects=[],
    ),
    "tech-sleek": StylePreset(
        name="tech-sleek",
        display_name="Tech Sleek",
        description="Modern tech aesthetic — clean sans-serif, red accent, snappy slides",
        config=StyleConfig(
            font_family="'DM Sans', sans-serif",
            font_import="DMSans",
            font_weights_to_load=["400", "500", "700"],
            palette=["#FFFFFF", "#1A1A1A", "#FF3333"],
            text_shadow="0 2px 6px rgba(0,0,0,0.5)",
            font_weights=FontWeights(heading="700", body="400", emphasis="700", marker="500"),
        ),
        preferred_animations=["slide-in"],
        avoided_animations=["bounce"],
        density_range=(4, 6),
        density_label="moderate",
        template_preferences=[],
        color_grading_preset="cool",
        color_grading_intensity=0.3,
        preferred_effects=["whip", "vignette"],
        avoided_effects=[],
    ),
    "casual-vlog": StylePreset(
        name="casual-vlog",
        display_name="Casual Vlog",
        description="Friendly vlog style — warm orange accent, relaxed pacing",
        config=StyleConfig(
            font_family="'Oswald', sans-serif",
            font_import="Oswald",
            font_weights_to_load=["400", "500", "700"],
            palette=["#FFFFFF", "#1C1C1C", "#FFB347"],
            text_shadow="0 2px 8px rgba(0,0,0,0.6)",
            font_weights=FontWeights(heading="700", body="400", emphasis="700", marker="500"),
        ),
        preferred_animations=["typewriter", "slide-in"],
        avoided_animations=[],
        density_range=(3, 5),
        density_label="sparse",
        template_preferences=[],
        color_grading_preset="warm",
        color_grading_intensity=0.5,
        preferred_effects=["vignette"],
        avoided_effects=[],
    ),
    "podcast-pro": StylePreset(
        name="podcast-pro",
        display_name="Podcast Pro",
        description="Clean interview/podcast style — minimal overlays, blue accent",
        config=StyleConfig(
            font_family="'Source Sans 3', sans-serif",
            font_import="SourceSans3",
            font_weights_to_load=["400", "600", "700"],
            palette=["#FFFFFF", "#222222", "#3B82F6"],
            text_shadow="0 1px 4px rgba(0,0,0,0.4)",
            font_weights=FontWeights(heading="600", body="400", emphasis="700", marker="600"),
        ),
        preferred_animations=["fade", "slide-in"],
        avoided_animations=["bounce", "pop"],
        density_range=(2, 3),
        density_label="very sparse",
        template_preferences=["lower_third"],
        color_grading_preset="",
        color_grading_intensity=0.0,
        preferred_effects=[],
        avoided_effects=["whip", "speed_ramp"],
    ),
    "tiktok-native": StylePreset(
        name="tiktok-native",
        display_name="TikTok Native",
        description="Short-form social — punchy pops, pink accent, high density",
        config=StyleConfig(
            font_family="'Poppins', sans-serif",
            font_import="Poppins",
            font_weights_to_load=["400", "600", "700", "800"],
            palette=["#FFFFFF", "#000000", "#FF2D55"],
            text_shadow="0 3px 10px rgba(0,0,0,0.7)",
            font_weights=FontWeights(heading="700", body="400", emphasis="800", marker="700"),
        ),
        preferred_animations=["pop", "bounce"],
        avoided_animations=[],
        density_range=(6, 10),
        density_label="high",
        template_preferences=[],
        color_grading_preset="dramatic",
        color_grading_intensity=0.4,
        preferred_effects=["whip", "speed_ramp"],
        avoided_effects=[],
    ),
}


def get_style(name: str) -> StylePreset:
    """Look up a style preset by name, falling back to default."""
    return STYLE_PRESETS.get(name, STYLE_PRESETS["default"])
