"""Helpers for low-resolution keyframe SAM3 masks and interpolation."""
from __future__ import annotations

import cv2
import numpy as np


def resize_for_sam3(
    frame: np.ndarray,
    max_side: int,
) -> tuple[np.ndarray, float, float]:
    """Resize a frame for SAM3 while preserving aspect ratio.

    Returns the resized frame and the scale factors from low-res coordinates
    back to the original frame coordinates.
    """
    h, w = frame.shape[:2]
    if max_side <= 0 or max(h, w) <= max_side:
        return frame.copy(), 1.0, 1.0

    scale = max_side / float(max(h, w))
    low_w = max(1, int(round(w * scale)))
    low_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (low_w, low_h), interpolation=cv2.INTER_AREA)
    return resized, w / float(low_w), h / float(low_h)


def upscale_mask_to_frame(mask: np.ndarray, frame_shape: tuple[int, ...]) -> np.ndarray:
    """Scale a binary mask to a frame's dimensions with nearest-neighbor edges."""
    h, w = frame_shape[:2]
    if mask.shape[:2] == (h, w):
        return mask
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def dilate_binary_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Expand a binary mask by roughly ``pixels`` in all directions."""
    if pixels <= 0:
        return mask
    kernel_size = pixels * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel, iterations=1)


def scale_bbox_to_frame(obj: dict, frame_shape: tuple[int, ...]) -> list[float] | None:
    """Scale a detection bbox from its mask space to the full frame space."""
    bbox = obj.get("bbox_xyxy")
    if not bbox or len(bbox) != 4:
        return None

    binary_mask = obj.get("binary_mask")
    if binary_mask is None:
        return [float(v) for v in bbox]

    mask_h, mask_w = binary_mask.shape[:2]
    frame_h, frame_w = frame_shape[:2]
    scale_x = frame_w / float(max(1, mask_w))
    scale_y = frame_h / float(max(1, mask_h))
    x1, y1, x2, y2 = bbox
    return [
        float(x1) * scale_x,
        float(y1) * scale_y,
        float(x2) * scale_x,
        float(y2) * scale_y,
    ]


def mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask == 255)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def detection_centroid(obj: dict) -> tuple[float, float] | None:
    binary_mask = obj.get("binary_mask")
    if binary_mask is not None:
        centroid = mask_centroid(binary_mask)
        if centroid is not None:
            return centroid

    bbox = obj.get("bbox_xyxy")
    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)
    return None


def translated_mask(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
    h, w = mask.shape[:2]
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        mask,
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def bbox_iou_xyxy(a: list[float], b: list[float]) -> float:
    if len(a) != 4 or len(b) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask == 255)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def clone_detection_with_mask(source: dict, mask: np.ndarray) -> dict:
    result = dict(source)
    result["binary_mask"] = mask
    bbox = bbox_from_mask(mask)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        result["bbox_xyxy"] = bbox
        result["bbox_xywh"] = [x1, y1, max(1, x2 - x1), max(1, y2 - y1)]
    return result


def match_detections_by_motion(
    start_detections: list[dict],
    end_detections: list[dict],
    mask_shape: tuple[int, int],
) -> list[tuple[int, int]]:
    """Greedily pair same-type detections across adjacent keyframes."""
    h, w = mask_shape[:2]
    diag = max(1.0, float((w * w + h * h) ** 0.5))
    candidates: list[tuple[float, int, int]] = []
    for start_idx, start_obj in enumerate(start_detections):
        start_center = detection_centroid(start_obj)
        if start_center is None:
            continue
        for end_idx, end_obj in enumerate(end_detections):
            if start_obj.get("type") != end_obj.get("type"):
                continue
            end_center = detection_centroid(end_obj)
            if end_center is None:
                continue
            distance = float(
                ((start_center[0] - end_center[0]) ** 2 + (start_center[1] - end_center[1]) ** 2) ** 0.5
            )
            distance_score = max(0.0, 1.0 - distance / (diag * 0.25))
            iou = bbox_iou_xyxy(start_obj.get("bbox_xyxy", []), end_obj.get("bbox_xyxy", []))
            score = iou + distance_score
            if score > 0.3:
                candidates.append((score, start_idx, end_idx))

    matches: list[tuple[int, int]] = []
    used_start: set[int] = set()
    used_end: set[int] = set()
    for _score, start_idx, end_idx in sorted(candidates, reverse=True):
        if start_idx in used_start or end_idx in used_end:
            continue
        used_start.add(start_idx)
        used_end.add(end_idx)
        matches.append((start_idx, end_idx))
    return matches


def interpolate_detection_masks(
    start_detections: list[dict],
    end_detections: list[dict],
    alpha: float,
) -> list[dict]:
    """Move matched keyframe masks along linear centroid motion.

    ``alpha`` is the relative position between the start and end keyframes.
    The function is privacy-biased for true intermediate frames: unmatched
    start/end masks are kept so an appearing/disappearing object is not missed.
    At exact keyframes, it returns that keyframe's detections without fallback.
    """
    if alpha <= 0:
        return start_detections
    if alpha >= 1:
        return end_detections

    sample_mask = next(
        (
            obj.get("binary_mask")
            for obj in [*start_detections, *end_detections]
            if obj.get("binary_mask") is not None
        ),
        None,
    )
    if sample_mask is None:
        return start_detections

    matches = match_detections_by_motion(start_detections, end_detections, sample_mask.shape)
    matched_start = {start_idx for start_idx, _end_idx in matches}
    matched_end = {end_idx for _start_idx, end_idx in matches}

    interpolated: list[dict] = []
    for start_idx, end_idx in matches:
        start_obj = start_detections[start_idx]
        end_obj = end_detections[end_idx]
        start_mask = start_obj.get("binary_mask")
        end_mask = end_obj.get("binary_mask")
        start_center = detection_centroid(start_obj)
        end_center = detection_centroid(end_obj)
        if start_mask is None or end_mask is None or start_center is None or end_center is None:
            continue

        dx = end_center[0] - start_center[0]
        dy = end_center[1] - start_center[1]
        shifted_start = translated_mask(start_mask, dx * alpha, dy * alpha)
        shifted_end = translated_mask(end_mask, -dx * (1.0 - alpha), -dy * (1.0 - alpha))
        merged = np.maximum(shifted_start, shifted_end)
        source = start_obj if float(start_obj.get("confidence", 0.0)) >= float(end_obj.get("confidence", 0.0)) else end_obj
        interpolated_obj = clone_detection_with_mask(source, merged)
        selected_id = start_obj.get("selected_pii_object_id") or end_obj.get("selected_pii_object_id")
        if selected_id is not None:
            interpolated_obj["selected_pii_object_id"] = selected_id
        interpolated.append(interpolated_obj)

    for start_idx, start_obj in enumerate(start_detections):
        if start_idx not in matched_start and start_obj.get("binary_mask") is not None:
            interpolated.append(start_obj)
    for end_idx, end_obj in enumerate(end_detections):
        if end_idx not in matched_end and end_obj.get("binary_mask") is not None:
            interpolated.append(end_obj)

    return interpolated
