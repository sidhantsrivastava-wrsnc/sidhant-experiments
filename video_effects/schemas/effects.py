from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class EffectType(str, Enum):
    ZOOM = "zoom"
    BLUR = "blur"
    COLOR_CHANGE = "color_change"
    SUBTITLE = "subtitle"
    WHIP = "whip"
    SPEED_RAMP = "speed_ramp"
    VIGNETTE = "vignette"


class TargetRegion(BaseModel):
    """Region of the frame to apply an effect to."""
    x: float = Field(0.0, description="Normalized x (0-1), left edge of region")
    y: float = Field(0.0, description="Normalized y (0-1), top edge of region")
    width: float = Field(1.0, description="Normalized width (0-1)")
    height: float = Field(1.0, description="Normalized height (0-1)")


class ZoomParams(BaseModel):
    tracking: Literal["face", "center", "point"] = "center"
    zoom_level: float = Field(1.5, ge=1.0, le=3.0, description="Zoom magnification level")
    easing: Literal["smooth", "snap", "overshoot"] = "smooth"
    action: Literal["bounce", "in", "out"] = "bounce"
    motion_blur: float = Field(0.0, ge=0.0, le=1.0, description="Radial motion blur during zoom transition (0=off)")


class WhipParams(BaseModel):
    direction: Literal["left", "right", "up", "down"] = "right"
    intensity: float = Field(1.0, ge=0.3, le=2.0, description="Motion blur strength multiplier")


class SpeedRampParams(BaseModel):
    speed: float = Field(2.0, ge=1.5, le=8.0, description="Playback speed multiplier")
    easing: Literal["smooth", "snap"] = "smooth"


class VignetteParams(BaseModel):
    strength: float = Field(0.5, ge=0.1, le=1.0, description="Darkness of vignette edges")
    radius: float = Field(0.8, ge=0.3, le=1.0, description="How far from center vignette starts (1.0 = no vignette)")


class BlurParams(BaseModel):
    blur_type: Literal["gaussian", "face_pixelate", "background", "radial"] = "gaussian"
    radius: float = Field(15.0, ge=1.0, description="Blur strength / kernel size")
    target_region: TargetRegion = Field(default_factory=TargetRegion)


class ColorParams(BaseModel):
    preset: Literal["warm", "cool", "bw", "sepia", "dramatic", "custom"] = "warm"
    intensity: float = Field(0.5, ge=0.0, le=1.0)
    # For custom: BGR adjustments
    r_adjust: float = 0.0
    g_adjust: float = 0.0
    b_adjust: float = 0.0


class SubtitleParams(BaseModel):
    text: str = ""
    font_size: int = 48
    color: str = "#FFFFFF"
    background_color: Optional[str] = "#000000CC"
    position: Literal["bottom", "top", "center"] = "bottom"
    bold: bool = True


class EffectCue(BaseModel):
    """A single effect cue parsed from transcript."""
    effect_type: EffectType
    start_time: float = Field(description="Start time in seconds")
    end_time: float = Field(description="End time in seconds")
    verbal_cue: str = Field("", description="The transcript text that triggered this effect")
    confidence: float = Field(1.0, ge=0.0, le=1.0)

    # Type-specific params (only one should be set based on effect_type)
    zoom_params: Optional[ZoomParams] = None
    blur_params: Optional[BlurParams] = None
    color_params: Optional[ColorParams] = None
    subtitle_params: Optional[SubtitleParams] = None
    whip_params: Optional[WhipParams] = None
    speed_ramp_params: Optional[SpeedRampParams] = None
    vignette_params: Optional[VignetteParams] = None


class ValidatedTimeline(BaseModel):
    """Ordered, conflict-resolved timeline of effects."""
    effects: list[EffectCue] = Field(default_factory=list)
    conflicts_resolved: int = 0
    total_duration: float = 0.0


class VideoInfo(BaseModel):
    width: int
    height: int
    fps: float
    duration: float
    codec: str = ""
    total_frames: int = 0
    audio_codec: str = ""
    has_audio: bool = True
    color_space: str = ""
    color_transfer: str = ""
    color_primaries: str = ""
    pix_fmt: str = ""
