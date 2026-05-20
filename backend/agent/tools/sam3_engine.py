"""SAM3 text-prompted PII detection (pixel-precise polygon masks)."""
from __future__ import annotations

import contextlib

import cv2
import numpy as np
from PIL import Image

# Internal PII keys stay short and stable for the API/UI.
# Each prompt stays as one concise object phrase; no prompt mixes alternatives.
_TEXT_PROMPTS = {
    "face": (
        "person face",
        "human face",
    ),
    "document": (
        "paper document",
        "printed page",
        "form",
    ),
    "screen": (
        "display screen",
        "computer monitor",
        "laptop screen",
        "phone screen",
        "tablet screen",
    ),
    "nameplate": (
        "nameplate",
        "name tag",
        "name badge",
    ),
    "id_card": (
        "identity card",
        "driver license",
        "passport",
    ),
}

_DEDUP_IOU_THRESHOLD = 0.5

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
            prompts = _TEXT_PROMPTS.get(pii_type, (pii_type,))
            for prompt in prompts:
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
    return _dedupe_detections(results)


def _mask_to_polygon(binary_mask: np.ndarray) -> list[list[int]] | None:
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    return largest.reshape(-1, 2).tolist()


def _dedupe_detections(detections: list[dict]) -> list[dict]:
    """Keep the highest-confidence box when prompts for the same PII type overlap."""
    kept: list[dict] = []
    for obj in sorted(detections, key=lambda item: item["confidence"], reverse=True):
        duplicate = any(
            obj["type"] == kept_obj["type"]
            and _bbox_iou(obj["bbox_xyxy"], kept_obj["bbox_xyxy"]) >= _DEDUP_IOU_THRESHOLD
            for kept_obj in kept
        )
        if not duplicate:
            kept.append(obj)
    return kept


def _bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
