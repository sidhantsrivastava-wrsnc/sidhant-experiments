from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class VideoEffectsSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="VFX_",
    )

    TASK_QUEUE: str = "video_effects_queue"
    TEMPORAL_NAMESPACE: str = "default"
    TEMPORAL_ENDPOINT: str = "localhost:7233"
    TEMPORAL_API_KEY: Optional[str] = None

    # Anthropic API for LLM cue parsing
    ANTHROPIC_API_KEY: Optional[str] = None
    LLM_MODEL: str = "claude-sonnet-4-20250514"

    # ElevenLabs for transcription (falls back to local whisper)
    ELEVENLABS_API_KEY: Optional[str] = None

    # Paths
    TEMP_DIR: str = "/tmp/video_effects"
    FACE_LANDMARKER_PATH: str = "cv_experiments/face_landmarker.task"  # Path to face_landmarker.task model

    # Processing
    FACE_DETECTION_STRIDE: int = 3
    SMOOTHING_ALPHA: float = 0.1

    # Remotion motion graphics
    REMOTION_DIR: Optional[str] = None  # Override path to remotion/ project (default: auto-detected)
    REMOTION_CONCURRENCY: Optional[int] = None  # Remotion render concurrency (default: Remotion auto)


settings = VideoEffectsSettings()
