"""Pydantic schema for LLM structured output (motion graphics planning)."""

from typing import Literal

from pydantic import BaseModel, Field


class MGComponentBounds(BaseModel):
    """Normalized screen region for a component."""
    x: float = Field(0.0, ge=0.0, le=1.0, description="Left edge (0-1)")
    y: float = Field(0.0, ge=0.0, le=1.0, description="Top edge (0-1)")
    w: float = Field(0.2, ge=0.0, le=1.0, description="Width (0-1)")
    h: float = Field(0.1, ge=0.0, le=1.0, description="Height (0-1)")


class MGComponentSpec(BaseModel):
    """A single motion graphics component in the LLM plan."""
    template: Literal[
        "animated_title", "lower_third", "zoom_callout",
        "transition_wipe", "keyword_highlight", "progress_bar",
    ] = Field(description="Which template to use")
    start_time: float = Field(description="Start time in seconds")
    end_time: float = Field(description="End time in seconds")
    props: dict = Field(default_factory=dict, description="Template-specific properties")
    bounds: MGComponentBounds = Field(
        default_factory=MGComponentBounds,
        description="Normalized screen region this component occupies",
    )
    z_index: int = Field(0, description="Layering order (higher = on top)")
    reasoning: str = Field("", description="Why this component is placed here")


class MotionGraphicsPlanResponse(BaseModel):
    """LLM output: full motion graphics plan for a video."""
    components: list[MGComponentSpec] = Field(
        default_factory=list,
        description="List of motion graphics overlay elements",
    )
    color_palette: list[str] = Field(
        default_factory=list,
        description="2-3 CSS hex colors used consistently across components",
    )
    reasoning: str = Field(
        "",
        description="Brief explanation of overall creative choices",
    )
