"""Schemas for the infographic code-generation workflow."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class InfographicType(str, Enum):
    PIE_CHART = "pie_chart"
    BAR_CHART = "bar_chart"
    LINE_CHART = "line_chart"
    FLOWCHART = "flowchart"
    TIMELINE = "timeline"
    COMPARISON = "comparison"
    PROCESS = "process"
    STAT_DASHBOARD = "stat_dashboard"
    CUSTOM = "custom"


class InfographicSpec(BaseModel):
    """Specification for a single infographic to generate."""

    id: str = Field(description="Unique identifier for this infographic")
    type: InfographicType = Field(description="Type of infographic")
    title: str = Field(description="Display title for the infographic")
    description: str = Field(description="What this infographic should show")
    data: dict = Field(default_factory=dict, description="Data to visualize (labels, values, etc.)")
    start_time: float = Field(description="When to show (seconds)")
    end_time: float = Field(description="When to hide (seconds)")
    bounds: dict = Field(
        default_factory=lambda: {"x": 0.1, "y": 0.1, "w": 0.35, "h": 0.3},
        description="Normalized screen region {x, y, w, h}",
    )
    anchor: str = Field("static", description="Face-aware anchor mode")


class InfographicPlanResponse(BaseModel):
    """LLM response from the planning activity."""

    reasoning: str = Field(description="Why these infographics were chosen")
    infographics: list[InfographicSpec] = Field(default_factory=list)


class CodeGenResponse(BaseModel):
    """Result of generating code for a single infographic."""

    component_id: str = Field(description="Unique component ID (used as filename and registry key)")
    tsx_code: str = Field(description="Generated TSX source code")
    export_name: str = Field(description="Named export of the React component")
    props: dict = Field(default_factory=dict, description="Props to pass at render time")


class ValidationResult(BaseModel):
    """Result of validating a generated component."""

    valid: bool = Field(description="Whether the component passed validation")
    errors: list[str] = Field(default_factory=list, description="Type-check or render errors")
    preview_path: str = Field("", description="Path to test-render preview image")


# Fallback mapping: infographic type -> existing template + default style
FALLBACK_MAP: dict[InfographicType, tuple[str, str]] = {
    InfographicType.PIE_CHART: ("data_animation", "bar"),
    InfographicType.BAR_CHART: ("data_animation", "bar"),
    InfographicType.LINE_CHART: ("data_animation", "bar"),
    InfographicType.COMPARISON: ("listicle", "pop"),
    InfographicType.TIMELINE: ("listicle", "slide"),
    InfographicType.PROCESS: ("listicle", "slide"),
    InfographicType.FLOWCHART: ("listicle", "slide"),
    InfographicType.STAT_DASHBOARD: ("data_animation", "stat-callout"),
    InfographicType.CUSTOM: ("animated_title", "fade"),
}
