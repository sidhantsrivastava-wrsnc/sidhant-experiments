from typing import Optional

from pydantic import BaseModel, Field


class VideoEffectsInput(BaseModel):
    input_video: str = Field(description="Path to input video file")
    output_video: str = Field(description="Path for output video file")
    auto_approve: bool = Field(False, description="Skip CLI approval step")
    enable_motion_graphics: bool = Field(False, description="Enable Remotion motion graphics overlay")
    style: str = Field("", description="Style preset name (empty = auto-detect via creative designer)")
    dev_mode: bool = Field(False, description="Dev mode: effects triggered by explicit verbal commands")
    smooth_jump_cuts: bool = Field(False, description="Detect jump cuts and insert synthetic zoom transitions")


class VideoEffectsOutput(BaseModel):
    output_video: str = Field(description="Path to the final output video")
    effects_applied: int = Field(0, description="Number of effects applied")
    transcript_length: int = Field(0, description="Length of transcript in chars")
    phases_executed: int = Field(0, description="Number of phases executed")
    motion_graphics_applied: int = Field(0, description="Number of motion graphics components rendered")
    error: Optional[str] = None
