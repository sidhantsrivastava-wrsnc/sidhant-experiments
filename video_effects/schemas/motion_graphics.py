from typing import Optional

from pydantic import BaseModel, Field

# Z-index tier constants for consistent layer ordering
Z_TIER_FULLSCREEN = 2       # Full-screen atmospheric effects (color flash, vignette pulse)
Z_TIER_BORDER = 5           # Border/frame effects (animated border, corner brackets, edge glow)
Z_TIER_INFOGRAPHIC = 10     # Standard infographic overlays (data viz, listicles, titles)
Z_TIER_SUBTITLE = 100       # Subtitles (always on top)


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
    template: str
    start_time: float
    end_time: float
    props: dict = Field(default_factory=dict)
    bounds: NormalizedRect = Field(default_factory=NormalizedRect)
    z_index: int = 0
    shadow: str = ""
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
