from typing import Optional

from pydantic import BaseModel, Field


class VideoEffectsInput(BaseModel):
    input_video: str = Field(description="Path to input video file")
    output_video: str = Field(description="Path for output video file")
    auto_approve: bool = Field(False, description="Skip CLI approval step")
    enable_motion_graphics: bool = Field(False, description="Enable Remotion motion graphics overlay")
    motion_graphics_style: str = Field("", description="Style hint for motion graphics planning")


class VideoEffectsOutput(BaseModel):
    output_video: str = Field(description="Path to the final output video")
    effects_applied: int = Field(0, description="Number of effects applied")
    transcript_length: int = Field(0, description="Length of transcript in chars")
    phases_executed: int = Field(0, description="Number of phases executed")
    motion_graphics_applied: int = Field(0, description="Number of motion graphics components rendered")
    error: Optional[str] = None
