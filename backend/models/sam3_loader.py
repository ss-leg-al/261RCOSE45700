"""SAM3 (facebookresearch/sam3) singleton loader.

API:
    processor = get_sam3_processor()
    state  = processor.set_image(pil_image)
    output = processor.set_text_prompt(state=state, prompt="human face")
    masks  = output["masks"]   # torch.Tensor [N, 1, H, W]
    boxes  = output["boxes"]   # torch.Tensor [N, 4] xyxy
    scores = output["scores"]  # torch.Tensor [N]
"""
from __future__ import annotations

import logging
from pathlib import Path

from ..config import settings

logger = logging.getLogger(__name__)

_model = None
_processor = None
_load_error: str | None = None


def is_available() -> bool:
    """Return whether the SAM3 image model is loaded or loadable."""
    return _processor is not None or (
        _load_error is None and Path(settings.SAM3_CHECKPOINT).exists()
    )


def get_load_error() -> str | None:
    return _load_error


def load_sam3(checkpoint: str = "checkpoints/sam3.pt"):
    global _model, _processor, _load_error
    if _processor is not None:
        return _processor

    ckpt = Path(checkpoint)
    if not ckpt.exists():
        _load_error = (
            f"SAM3 checkpoint not found at {ckpt.resolve()}. "
            "Download sam3.pt from https://huggingface.co/facebook/sam3 "
            f"and place it at {ckpt}."
        )
        logger.warning(_load_error)
        return None

    try:
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
    except ImportError as e:
        _load_error = (
            f"Meta SAM3 package not installed: {e}. "
            "Run: git clone https://github.com/facebookresearch/sam3.git ~/sam3_repo "
            "&& cd ~/sam3_repo && pip install -e ."
        )
        logger.error(_load_error)
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("SAM3 will run on CPU — inference will be slow.")

    try:
        model = build_sam3_image_model(
            checkpoint_path=str(ckpt),
            load_from_HF=True,
            device=device,
        )
        processor = Sam3Processor(model)
        _model = model
        _processor = processor
        _load_error = None
        logger.info("SAM3 loaded on %s", device.upper())
        return _processor
    except Exception as e:
        _load_error = f"SAM3 load failed: {e}"
        logger.exception(_load_error)
        return None


def get_sam3_processor():
    if _processor is None:
        load_sam3(settings.SAM3_CHECKPOINT)
    if _processor is None:
        raise RuntimeError(_load_error or "SAM3 not loaded. Call load_sam3() first.")
    return _processor
