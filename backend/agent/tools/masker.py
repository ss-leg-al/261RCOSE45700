"""Apply SAM3 polygon masks and fallback bbox blurs."""
from __future__ import annotations

import cv2
import numpy as np


def apply_polygon_mask(
    img: np.ndarray,
    polygon,
    strategy: str,
    overlay_color: tuple[int, int, int] | None = None,
    overlay_alpha: float = 0.38,
) -> np.ndarray:
    """Apply privacy masking within a SAM3 polygon and optionally tint it.

    ``overlay_color`` is an OpenCV BGR tuple. The tint is applied *after* the
    privacy transform so the colored result makes changed regions visible
    without exposing the original pixels.
    """
    if polygon is None or len(polygon) < 3:
        return img
    pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    region = mask == 255
    if not region.any():
        return img

    if strategy == "blur":
        blurred = cv2.GaussianBlur(img, (51, 51), 15)
        img[region] = blurred[region]
    elif strategy == "blackbox":
        img[region] = 0
    elif strategy == "pixelate":
        ys, xs = np.where(region)
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return img
        rh, rw = roi.shape[:2]
        block = max(2, min(rw, rh) // 12 or 2)
        small = cv2.resize(roi, (max(1, rw // block), max(1, rh // block)))
        pix = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
        sub = mask[y1:y2, x1:x2] == 255
        roi[sub] = pix[sub]
        img[y1:y2, x1:x2] = roi
    elif strategy == "overlay_only":
        pass
    if overlay_color is not None:
        _apply_colored_overlay(img, mask, overlay_color, overlay_alpha)
        thickness = max(2, min(h, w) // 360)
        cv2.polylines(img, [pts], isClosed=True, color=overlay_color, thickness=thickness)
    return img


def apply_binary_mask(
    img: np.ndarray,
    mask: np.ndarray,
    strategy: str,
    overlay_color: tuple[int, int, int] | None = None,
    overlay_alpha: float = 0.38,
) -> np.ndarray:
    """Apply privacy masking within a binary mask and optionally tint it."""
    if mask is None:
        return img

    h, w = img.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    mask = ((mask > 0).astype(np.uint8) * 255)
    region = mask == 255
    if not region.any():
        return img

    if strategy == "blur":
        blurred = cv2.GaussianBlur(img, (51, 51), 15)
        img[region] = blurred[region]
    elif strategy == "blackbox":
        img[region] = 0
    elif strategy == "pixelate":
        ys, xs = np.where(region)
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return img
        rh, rw = roi.shape[:2]
        block = max(2, min(rw, rh) // 12 or 2)
        small = cv2.resize(roi, (max(1, rw // block), max(1, rh // block)))
        pix = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
        sub = mask[y1:y2, x1:x2] == 255
        roi[sub] = pix[sub]
        img[y1:y2, x1:x2] = roi
    elif strategy == "overlay_only":
        pass

    if overlay_color is not None:
        _apply_colored_overlay(img, mask, overlay_color, overlay_alpha)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            thickness = max(2, min(h, w) // 360)
            cv2.drawContours(img, contours, -1, overlay_color, thickness)
    return img


def blur_bbox(
    img: np.ndarray,
    bbox_xyxy: list[float],
    strength: int = 51,
    overlay_color: tuple[int, int, int] | None = None,
    overlay_alpha: float = 0.32,
) -> np.ndarray:
    """Fallback rectangular blur when SAM3 polygon is unavailable."""
    x1, y1, x2, y2 = (int(v) for v in bbox_xyxy)
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return img
    k = strength | 1
    img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (k, k), 0)
    if overlay_color is not None:
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        _apply_colored_overlay(img, mask, overlay_color, overlay_alpha)
        thickness = max(2, min(h, w) // 360)
        cv2.rectangle(img, (x1, y1), (x2, y2), overlay_color, thickness)
    return img


def _apply_colored_overlay(
    img: np.ndarray,
    mask: np.ndarray,
    color_bgr: tuple[int, int, int],
    alpha: float,
) -> None:
    region = mask == 255
    if not region.any():
        return
    alpha = max(0.0, min(1.0, alpha))
    overlay = img.copy()
    overlay[region] = color_bgr
    blended = cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0)
    img[region] = blended[region]
