from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    OPENAI_API_KEY: str = ""
    SAMPLE_FPS: int = 1
    MAX_VIDEO_SIZE_MB: int = 500
    FACE_DETECT_INTERVAL: int = 10   # Phase 1: run InsightFace every N sampled frames
    SCENE_ANALYSIS_FRAMES: int = 5   # number of frames to sample for GPT-4o scene analysis
    FACE_SIMILARITY_THRESHOLD: float = 0.55
    SAM3_CONFIDENCE_THRESHOLD: float = 0.3
    UPLOAD_DIR: str = "uploads"
    OUTPUT_DIR: str = "outputs"
    SAM3_CHECKPOINT: str = "checkpoints/sam3.pt"

    @property
    def upload_path(self) -> Path:
        return Path(self.UPLOAD_DIR)

    @property
    def output_path(self) -> Path:
        return Path(self.OUTPUT_DIR)

    model_config = {"env_file": ".env"}


settings = Settings()
