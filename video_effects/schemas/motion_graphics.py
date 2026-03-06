from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class MGTemplate(str, Enum):
    ANIMATED_TITLE = "animated_title"
    LOWER_THIRD = "lower_third"
    ZOOM_CALLOUT = "zoom_callout"
    TRANSITION_WIPE = "transition_wipe"
    KEYWORD_HIGHLIGHT = "keyword_highlight"
    PROGRESS_BAR = "progress_bar"


class NormalizedRect(BaseModel):
    """A rectangle in normalized [0-1] coordinates."""
    x: float = Field(0.0, ge=0.0, le=1.0)
    y: float = Field(0.0, ge=0.0, le=1.0)
    w: float = Field(1.0, ge=0.0, le=1.0)
    h: float = Field(1.0, ge=0.0, le=1.0)
    label: str = ""


class FaceWindow(BaseModel):
    """Face position averaged over a time window."""
    start_time: float
    end_time: float
    face_region: NormalizedRect
    safe_regions: list[NormalizedRect] = Field(default_factory=list)


class SpatialContext(BaseModel):
    """Full spatial context for Remotion composition planning."""
    video: dict = Field(default_factory=dict)
    transcript: dict = Field(default_factory=dict)
    face_windows: list[FaceWindow] = Field(default_factory=list)
    opencv_effects: list[dict] = Field(default_factory=list)
    face_data_path: str = ""


class MotionGraphicsComponent(BaseModel):
    """A single motion graphics element in the plan."""
    template: Literal[
        "animated_title", "lower_third", "zoom_callout",
        "transition_wipe", "keyword_highlight", "progress_bar",
    ]
    start_time: float
    end_time: float
    props: dict = Field(default_factory=dict)
    bounds: NormalizedRect = Field(default_factory=NormalizedRect)
    z_index: int = 0
    reasoning: str = ""


class MotionGraphicsPlan(BaseModel):
    """LLM-generated plan for all motion graphics overlays."""
    components: list[MotionGraphicsComponent] = Field(default_factory=list)
    color_palette: list[str] = Field(default_factory=list)
    reasoning: str = ""


class MotionGraphicsPreview(BaseModel):
    """Preview snapshot metadata."""
    frame: int
    timestamp: float
    snapshot_path: str
    components_visible: list[str] = Field(default_factory=list)
