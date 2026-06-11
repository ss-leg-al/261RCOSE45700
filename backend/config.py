from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    OPENAI_API_KEY: str = ""
    SAMPLE_FPS: int = 1
    MAX_VIDEO_SIZE_MB: int = 500
    FACE_DETECT_INTERVAL: int = 1    # Phase 1: run InsightFace every N sampled frames
    SCENE_ANALYSIS_FRAMES: int = 5   # number of frames to sample for GPT-4o scene analysis
    FACE_SIMILARITY_THRESHOLD: float = 0.55
    SAM3_CONFIDENCE_THRESHOLD: float = 0.3
    SAM3_MODE: str = "normal"            # normal=keyframe interpolation, precision=per-frame
    SAM3_MASK_INTERPOLATION_ENABLED: bool = True
    SAM3_MASK_LOWRES_MAX_SIDE: int = 960
    SAM3_MASK_KEYFRAME_INTERVAL: int = 3
    SAM3_MASK_DILATE_PX: int = 24
    BRAND_LOGO_DETECTION_ENABLED: bool = True
    MASK_PREVIEW_MAX_SIDE: int = 960
    DEBUG_MASK_OVERLAY: bool = False
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
