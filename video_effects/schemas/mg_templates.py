"""Template registry for motion graphics components.

Each template declares its metadata (props, spatial hints, duration range)
and points to a creative-guidance markdown file. The planner assembles its
system prompt dynamically from only the templates listed in
IMPLEMENTED_TEMPLATES.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Spec models
# ---------------------------------------------------------------------------

class PropSpec(BaseModel):
    """Describes a single prop that a template accepts."""
    name: str
    type: Literal["str", "int", "float", "bool", "literal"]
    required: bool = True
    default: Any = None
    description: str = ""
    choices: list[str] | None = None
    min_value: float | None = None
    max_value: float | None = None


class SpatialHint(BaseModel):
    """Rough guidance on where this template typically lives on screen."""
    typical_y_range: tuple[float, float] = (0.0, 1.0)
    typical_x_range: tuple[float, float] = (0.0, 1.0)
    edge_aligned: bool = False


class MGTemplateSpec(BaseModel):
    """Full metadata for a single motion-graphics template."""
    name: str
    display_name: str
    description: str
    props: list[PropSpec]
    duration_range: tuple[float, float] = (1.0, 10.0)
    spatial: SpatialHint = Field(default_factory=SpatialHint)
    guidance_file: str = ""


# ---------------------------------------------------------------------------
# Template specs
# ---------------------------------------------------------------------------

ANIMATED_TITLE = MGTemplateSpec(
    name="animated_title",
    display_name="Animated Title",
    description="Animated text overlay. Use for section headers, key statements, or emphasis.",
    props=[
        PropSpec(
            name="text",
            type="str",
            required=True,
            description="The text to display",
        ),
        PropSpec(
            name="style",
            type="literal",
            required=False,
            default="fade",
            description="Animation style",
            choices=["fade", "slide-in", "typewriter", "bounce"],
        ),
        PropSpec(
            name="fontSize",
            type="int",
            required=False,
            default=64,
            description="Font size in pixels",
            min_value=24,
            max_value=96,
        ),
        PropSpec(
            name="color",
            type="str",
            required=False,
            default="#FFFFFF",
            description="CSS hex color",
        ),
        PropSpec(
            name="fontWeight",
            type="str",
            required=False,
            default="700",
            description="Font weight (400-900)",
        ),
    ],
    duration_range=(2.0, 5.0),
    spatial=SpatialHint(
        typical_y_range=(0.05, 0.25),
        typical_x_range=(0.1, 0.9),
    ),
    guidance_file="animated_title.md",
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MG_TEMPLATE_REGISTRY: dict[str, MGTemplateSpec] = {
    "animated_title": ANIMATED_TITLE,
}

IMPLEMENTED_TEMPLATES: set[str] = {"animated_title"}

_GUIDANCE_DIR = Path(__file__).resolve().parent.parent / "prompts" / "mg_guidance"


def get_available_templates() -> list[MGTemplateSpec]:
    """Return specs for all implemented templates."""
    return [
        MG_TEMPLATE_REGISTRY[name]
        for name in sorted(IMPLEMENTED_TEMPLATES)
        if name in MG_TEMPLATE_REGISTRY
    ]


def load_guidance(spec: MGTemplateSpec) -> str:
    """Load the creative-guidance markdown for a template.

    Returns empty string if the file doesn't exist.
    """
    if not spec.guidance_file:
        return ""
    path = _GUIDANCE_DIR / spec.guidance_file
    if path.exists():
        return path.read_text().strip()
    return ""
