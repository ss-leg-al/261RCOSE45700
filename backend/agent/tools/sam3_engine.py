"""SAM3 text-prompted PII detection (pixel-precise polygon masks)."""
from __future__ import annotations

import contextlib

import cv2
import numpy as np
from PIL import Image

_TEXT_PROMPT = {
    "face":      "human face",
    "document":  "paper document",
    "screen":    "computer screen or monitor",
    "nameplate": "name badge",
    "id_card":   "identity card with photo",
}

MASK_STRATEGY = {
    "face":      "blur",
    "document":  "blackbox",
    "screen":    "pixelate",
    "nameplate": "blackbox",
    "id_card":   "blackbox",
}


def detect_pii(
    image_path: str,
    pii_types: list[str],
    conf_threshold: float = 0.3,
) -> list[dict]:
    """Detect PII objects with SAM3 text prompts.

    Returns list of:
      {type, polygon, bbox_xyxy, bbox_xywh, confidence, mask_strategy}
    """
    import torch
    from ...models.sam3_loader import get_sam3_processor

    processor = get_sam3_processor()
    image = Image.open(image_path).convert("RGB")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast = (
        torch.autocast(device, dtype=torch.bfloat16)
        if device == "cuda"
        else contextlib.nullcontext()
    )

    results: list[dict] = []
    with autocast:
        state = processor.set_image(image)
        for pii_type in pii_types:
            prompt = _TEXT_PROMPT.get(pii_type, pii_type)
            output = processor.set_text_prompt(state=state, prompt=prompt)
            masks = output.get("masks")
            boxes = output.get("boxes")
            scores = output.get("scores")
            if masks is None or boxes is None or scores is None:
                continue

            masks_np  = masks.detach().cpu().float().numpy()   # [N,1,H,W]
            boxes_np  = boxes.detach().cpu().float().numpy()   # [N,4] xyxy
            scores_np = scores.detach().cpu().float().numpy()  # [N]

            for i in range(len(scores_np)):
                conf = float(scores_np[i])
                if conf < conf_threshold:
                    continue
                x1, y1, x2, y2 = (int(v) for v in boxes_np[i])
                x1, y1 = max(0, x1), max(0, y1)
                polygon = _mask_to_polygon((masks_np[i, 0] > 0.5).astype(np.uint8) * 255)
                results.append({
                    "type":          pii_type,
                    "polygon":       polygon,
                    "bbox_xyxy":     [x1, y1, x2, y2],
                    "bbox_xywh":     [x1, y1, max(1, x2 - x1), max(1, y2 - y1)],
                    "confidence":    conf,
                    "mask_strategy": MASK_STRATEGY.get(pii_type, "blackbox"),
                })
    return results


def _mask_to_polygon(binary_mask: np.ndarray) -> list[list[int]] | None:
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    return largest.reshape(-1, 2).tolist()
