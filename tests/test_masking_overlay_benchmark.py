from __future__ import annotations

import gc
import json
import os
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from backend.agent.tools.sam3_engine import _mask_to_polygon, detect_pii
from backend.agent.tools.video_tools import compose_video
from backend.config import settings
from backend.models.sam3_loader import get_sam3_processor, is_available as sam3_available


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JJI_VIDEO = REPO_ROOT / "local_benchmarks" / "sam3_masking_time" / "jji.mp4"
DEFAULT_BENCHMARK_OUTPUT_ROOT = (
    REPO_ROOT / "local_benchmarks" / "sam3_masking_time" / "runs" / "mask_overlay_outputs"
)
MASK_VISUAL_COLORS_BGR = {
    "face": (247, 85, 168),
    "document": (11, 158, 245),
    "screen": (212, 182, 6),
    "nameplate": (68, 68, 239),
    "id_card": (129, 185, 16),
    "license_plate": (22, 115, 249),
}


def _video_metadata(video_path: Path) -> dict[str, float | int]:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"could not open video: {video_path}")
        return {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        }
    finally:
        cap.release()


def _scale_bbox(
    ratios: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = ratios
    return (
        int(round(x1 * width)),
        int(round(y1 * height)),
        int(round(x2 * width)),
        int(round(y2 * height)),
    )


def _rect_binary_mask(
    image_shape: tuple[int, int],
    bbox_xyxy: tuple[int, int, int, int],
) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.rectangle(mask, bbox_xyxy[:2], bbox_xyxy[2:], 255, -1)
    return mask


def _ellipse_binary_mask(
    image_shape: tuple[int, int],
    bbox_xyxy: tuple[int, int, int, int],
) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.uint8)
    x1, y1, x2, y2 = bbox_xyxy
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    axes = (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask


def _overlay_binary_mask(
    img: np.ndarray,
    mask: np.ndarray,
    color_bgr: tuple[int, int, int],
    alpha: float = 0.38,
) -> np.ndarray:
    region = mask == 255
    if not region.any():
        return img
    overlay = img.copy()
    overlay[region] = color_bgr
    blended = cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0)
    img[region] = blended[region]
    return img


def _overlay_polygon_mask(
    img: np.ndarray,
    polygon: list[list[int]],
    color_bgr: tuple[int, int, int],
) -> np.ndarray:
    pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    _overlay_binary_mask(img, mask, color_bgr)
    thickness = max(2, min(img.shape[:2]) // 360)
    cv2.polylines(img, [pts], isClosed=True, color=color_bgr, thickness=thickness)
    return img


def _draw_mask_outline(
    img: np.ndarray,
    mask: np.ndarray,
    color_bgr: tuple[int, int, int],
) -> None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return
    thickness = max(2, min(img.shape[:2]) // 360)
    cv2.drawContours(img, contours, -1, color_bgr, thickness)


def _overlay_detection_masks(
    frame: np.ndarray,
    detections: list[dict],
    *,
    dilate_px: int = 0,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for obj in detections:
        pii_type = obj["type"]
        binary_mask = obj.get("binary_mask")
        if binary_mask is None:
            continue
        if binary_mask.shape[:2] != frame.shape[:2]:
            binary_mask = _upscale_mask_to_frame(binary_mask, frame.shape)
        if dilate_px > 0:
            binary_mask = _dilate_binary_mask(binary_mask, dilate_px)
        color_bgr = MASK_VISUAL_COLORS_BGR.get(pii_type, (0, 255, 255))
        _overlay_binary_mask(frame, binary_mask, color_bgr)
        _draw_mask_outline(frame, binary_mask, color_bgr)
        counts[pii_type] = counts.get(pii_type, 0) + 1
    return counts


def _resize_for_sam3(
    frame: np.ndarray,
    max_side: int,
) -> tuple[np.ndarray, float, float]:
    h, w = frame.shape[:2]
    if max_side <= 0 or max(h, w) <= max_side:
        return frame.copy(), 1.0, 1.0
    scale = max_side / float(max(h, w))
    low_w = max(1, int(round(w * scale)))
    low_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (low_w, low_h), interpolation=cv2.INTER_AREA)
    return resized, w / float(low_w), h / float(low_h)


def _upscale_mask_to_frame(mask: np.ndarray, frame_shape: tuple[int, int, int]) -> np.ndarray:
    h, w = frame_shape[:2]
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def _dilate_binary_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask
    kernel_size = pixels * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel, iterations=1)


def _parse_target_types(value: str | None) -> list[str]:
    if not value:
        return ["screen", "nameplate", "face"]
    return [item.strip() for item in value.split(",") if item.strip()]


def _write_overlay_frame_sequence(
    video_path: Path,
    frames_dir: Path,
    max_frames: int,
    process_frame,
    *,
    pass_index: bool = False,
) -> tuple[float, int]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frames_dir.glob("*.jpg"):
        old_frame.unlink()

    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    count = 0
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"could not open video: {video_path}")
        start = time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if pass_index:
                process_frame(frame, count)
            else:
                process_frame(frame)
            count += 1
            out_path = frames_dir / f"{count:04d}.jpg"
            if not cv2.imwrite(str(out_path), frame):
                raise RuntimeError(f"could not write frame: {out_path}")
            if max_frames > 0 and count >= max_frames:
                break
        return time.perf_counter() - start, count
    finally:
        cap.release()
        if was_enabled:
            gc.enable()


def _compose_timed(frames_dir: Path, out_path: Path, fps: float) -> float:
    start = time.perf_counter()
    compose_video(frames_dir, out_path, fps=fps)
    return time.perf_counter() - start


def _union_detection_masks(
    detections: list[dict],
    frame_shape: tuple[int, int, int],
    *,
    dilate_px: int = 0,
) -> np.ndarray:
    union = np.zeros(frame_shape[:2], dtype=np.uint8)
    for obj in detections:
        binary_mask = obj.get("binary_mask")
        if binary_mask is None:
            continue
        if binary_mask.shape[:2] != frame_shape[:2]:
            binary_mask = _upscale_mask_to_frame(binary_mask, frame_shape)
        if dilate_px > 0:
            binary_mask = _dilate_binary_mask(binary_mask, dilate_px)
        union[binary_mask == 255] = 255
    return union


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a == 255
    b = mask_b == 255
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(a, b).sum()
    return float(intersection / union)


def _mask_coverage(reference_mask: np.ndarray, candidate_mask: np.ndarray) -> float:
    reference = reference_mask == 255
    reference_area = reference.sum()
    if reference_area == 0:
        return 1.0
    candidate = candidate_mask == 255
    return float(np.logical_and(reference, candidate).sum() / reference_area)


def _mask_area_ratio(reference_mask: np.ndarray, candidate_mask: np.ndarray) -> float | None:
    reference_area = int((reference_mask == 255).sum())
    if reference_area == 0:
        return None
    candidate_area = int((candidate_mask == 255).sum())
    return float(candidate_area / reference_area)


def _bbox_iou_xyxy(a: list[float], b: list[float]) -> float:
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


def _mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask == 255)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _detection_centroid(obj: dict) -> tuple[float, float] | None:
    binary_mask = obj.get("binary_mask")
    if binary_mask is not None:
        centroid = _mask_centroid(binary_mask)
        if centroid is not None:
            return centroid
    bbox = obj.get("bbox_xyxy")
    if bbox:
        x1, y1, x2, y2 = bbox
        return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)
    return None


def _translated_mask(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
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


def _bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask == 255)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def _clone_detection_with_mask(source: dict, mask: np.ndarray) -> dict:
    result = dict(source)
    result["binary_mask"] = mask
    bbox = _bbox_from_mask(mask)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        result["bbox_xyxy"] = bbox
        result["bbox_xywh"] = [x1, y1, max(1, x2 - x1), max(1, y2 - y1)]
    return result


def _match_detections_by_motion(
    start_detections: list[dict],
    end_detections: list[dict],
    mask_shape: tuple[int, int],
) -> list[tuple[int, int]]:
    """Greedily pair same-type detections across adjacent keyframes."""
    h, w = mask_shape[:2]
    diag = max(1.0, float((w * w + h * h) ** 0.5))
    candidates: list[tuple[float, int, int]] = []
    for start_idx, start_obj in enumerate(start_detections):
        start_center = _detection_centroid(start_obj)
        if start_center is None:
            continue
        for end_idx, end_obj in enumerate(end_detections):
            if start_obj.get("type") != end_obj.get("type"):
                continue
            end_center = _detection_centroid(end_obj)
            if end_center is None:
                continue
            distance = float(
                ((start_center[0] - end_center[0]) ** 2 + (start_center[1] - end_center[1]) ** 2) ** 0.5
            )
            distance_score = max(0.0, 1.0 - distance / (diag * 0.25))
            iou = _bbox_iou_xyxy(start_obj.get("bbox_xyxy", []), end_obj.get("bbox_xyxy", []))
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


def _interpolate_detection_masks(
    start_detections: list[dict],
    end_detections: list[dict],
    alpha: float,
) -> list[dict]:
    """Move matched keyframe masks along linear centroid motion for an intermediate frame."""
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

    matches = _match_detections_by_motion(start_detections, end_detections, sample_mask.shape)
    matched_start = {start_idx for start_idx, _end_idx in matches}
    matched_end = {end_idx for _start_idx, end_idx in matches}

    interpolated: list[dict] = []
    for start_idx, end_idx in matches:
        start_obj = start_detections[start_idx]
        end_obj = end_detections[end_idx]
        start_mask = start_obj.get("binary_mask")
        end_mask = end_obj.get("binary_mask")
        start_center = _detection_centroid(start_obj)
        end_center = _detection_centroid(end_obj)
        if start_mask is None or end_mask is None or start_center is None or end_center is None:
            continue
        dx = end_center[0] - start_center[0]
        dy = end_center[1] - start_center[1]
        shifted_start = _translated_mask(start_mask, dx * alpha, dy * alpha)
        shifted_end = _translated_mask(end_mask, -dx * (1.0 - alpha), -dy * (1.0 - alpha))
        merged = np.maximum(shifted_start, shifted_end)
        source = start_obj if float(start_obj.get("confidence", 0.0)) >= float(end_obj.get("confidence", 0.0)) else end_obj
        interpolated.append(_clone_detection_with_mask(source, merged))

    # Privacy-biased fallback: keep unmatched masks around rather than dropping
    # a newly appearing or disappearing object between keyframes.
    for start_idx, start_obj in enumerate(start_detections):
        if start_idx not in matched_start and start_obj.get("binary_mask") is not None:
            interpolated.append(start_obj)
    for end_idx, end_obj in enumerate(end_detections):
        if end_idx not in matched_end and end_obj.get("binary_mask") is not None:
            interpolated.append(end_obj)

    return interpolated


def _put_label(
    img: np.ndarray,
    text: str,
    origin: tuple[int, int],
) -> None:
    cv2.putText(img, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"could not write image: {path}")


def _read_first_frame(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"could not open video: {video_path}")
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise RuntimeError(f"could not read first frame: {video_path}")
    return frame


def _warm_sam3_for_benchmark(
    inputs: list[tuple[str, Path]],
    target_types: list[str],
    conf_threshold: float,
) -> dict:
    """Separate one-time SAM3 load/warm-up from timed benchmark measurements."""
    load_start = time.perf_counter()
    get_sam3_processor()
    load_seconds = time.perf_counter() - load_start

    runs = []
    total_detection_warmup_seconds = 0.0
    for label, image_path in inputs:
        warmup_start = time.perf_counter()
        detections = detect_pii(
            str(image_path),
            target_types,
            conf_threshold,
            include_binary_mask=True,
        )
        warmup_seconds = time.perf_counter() - warmup_start
        total_detection_warmup_seconds += warmup_seconds
        runs.append({
            "label": label,
            "image": str(image_path),
            "seconds": round(warmup_seconds, 6),
            "detections": len(detections),
        })

    return {
        "load_seconds": round(load_seconds, 6),
        "detection_warmup_seconds": round(total_detection_warmup_seconds, 6),
        "runs": runs,
        "note": "Excluded from reported original/lowres timed detection seconds.",
    }


@unittest.skipUnless(
    os.environ.get("RUN_MASK_BENCHMARK") == "1",
    "Set RUN_MASK_BENCHMARK=1 to generate local colored-mask overlay outputs.",
)
class MaskingOverlayBenchmarkTests(unittest.TestCase):
    def test_jji_colored_mask_overlay_output_runtime(self) -> None:
        video_path = Path(os.environ.get("MASK_BENCHMARK_VIDEO", DEFAULT_JJI_VIDEO))
        if not video_path.exists():
            self.skipTest(f"benchmark video not found: {video_path}")

        metadata = _video_metadata(video_path)
        width = int(metadata["width"])
        height = int(metadata["height"])
        fps = float(metadata["fps"])
        max_frames = int(os.environ.get("MASK_BENCHMARK_MAX_FRAMES", "0"))
        run_id = os.environ.get("MASK_BENCHMARK_RUN_ID") or time.strftime("%Y%m%d_%H%M%S")
        output_root = Path(os.environ.get("MASK_BENCHMARK_OUTPUT_ROOT", DEFAULT_BENCHMARK_OUTPUT_ROOT))
        run_root = output_root / f"jji_{run_id}"

        # Static mask fixtures measured from the first jji.mp4 frame. This
        # output is intentionally a colored-mask preview only: no blur,
        # pixelate, blackbox, bbox shape-prior, or SAM3 inference happens here.
        screen_bbox = _scale_bbox((0.273, 0.465, 0.406, 0.663), width, height)
        nameplate_bbox = _scale_bbox((0.055, 0.810, 0.126, 0.875), width, height)
        face_bbox = _scale_bbox((0.728, 0.025, 0.805, 0.238), width, height)

        image_shape = (height, width)
        mask_fixtures = [
            (_rect_binary_mask(image_shape, screen_bbox), MASK_VISUAL_COLORS_BGR["screen"]),
            (_rect_binary_mask(image_shape, nameplate_bbox), MASK_VISUAL_COLORS_BGR["nameplate"]),
            (_ellipse_binary_mask(image_shape, face_bbox), MASK_VISUAL_COLORS_BGR["face"]),
        ]
        polygon_fixtures = []
        for binary_mask, color_bgr in mask_fixtures:
            polygon = _mask_to_polygon(binary_mask)
            self.assertIsNotNone(polygon)
            polygon_fixtures.append((polygon, color_bgr))

        def apply_colored_mask_overlay(frame: np.ndarray) -> None:
            for polygon, color_bgr in polygon_fixtures:
                _overlay_polygon_mask(frame, polygon, color_bgr)

        frames_dir = run_root / "frames"
        output_video = run_root / "jji_colored_mask_overlay.mp4"
        mask_write_seconds, frame_count = _write_overlay_frame_sequence(
            video_path,
            frames_dir,
            max_frames,
            apply_colored_mask_overlay,
        )
        compose_seconds = _compose_timed(frames_dir, output_video, fps=fps)
        total_seconds = mask_write_seconds + compose_seconds
        self.assertTrue(output_video.exists())
        self.assertGreater(output_video.stat().st_size, 0)

        report = {
            "video": str(video_path),
            "metadata": metadata,
            "max_frames": max_frames,
            "run_root": str(run_root),
            "frames": frame_count,
            "mask_overlay_and_jpeg_write_seconds": round(mask_write_seconds, 6),
            "compose_seconds": round(compose_seconds, 6),
            "total_seconds": round(total_seconds, 6),
            "output_video": str(output_video),
            "output_bytes": output_video.stat().st_size,
            "note": (
                "Colored mask overlay preview only. This benchmark does not run SAM3 "
                "and does not apply blur, pixelate, blackbox, or shape-prior masking."
            ),
        }
        report_path = run_root / "report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["report_path"] = str(report_path)
        print("\nJJI_COLORED_MASK_OVERLAY " + json.dumps(report, ensure_ascii=False, sort_keys=True))

    def test_jji_lowres_sam3_colored_mask_overlay_output_runtime(self) -> None:
        if not sam3_available():
            self.skipTest("SAM3 checkpoint/package is not available in this environment.")

        video_path = Path(os.environ.get("MASK_BENCHMARK_VIDEO", DEFAULT_JJI_VIDEO))
        if not video_path.exists():
            self.skipTest(f"benchmark video not found: {video_path}")

        metadata = _video_metadata(video_path)
        fps = float(metadata["fps"])
        max_frames_env = os.environ.get("MASK_BENCHMARK_MAX_FRAMES")
        # Full-video SAM3 is intentionally opt-in. Without an explicit value,
        # keep this real-SAM3 test to a short smoke run.
        max_frames = int(max_frames_env) if max_frames_env is not None else 2
        lowres_max_side = int(os.environ.get("MASK_BENCHMARK_LOWRES_MAX_SIDE", "960"))
        target_types = _parse_target_types(os.environ.get("MASK_BENCHMARK_TARGET_TYPES"))
        conf_threshold = float(
            os.environ.get("MASK_BENCHMARK_CONF_THRESHOLD", settings.SAM3_CONFIDENCE_THRESHOLD)
        )
        run_id = os.environ.get("MASK_BENCHMARK_RUN_ID") or time.strftime("%Y%m%d_%H%M%S")
        output_root = Path(os.environ.get("MASK_BENCHMARK_OUTPUT_ROOT", DEFAULT_BENCHMARK_OUTPUT_ROOT))
        run_root = output_root / f"jji_lowres_sam3_{run_id}"
        lowres_dir = run_root / "lowres_frames"
        lowres_dir.mkdir(parents=True, exist_ok=True)
        for old_frame in lowres_dir.glob("*.jpg"):
            old_frame.unlink()

        stats = {
            "sam3_detection_seconds": 0.0,
            "detections": 0,
            "detections_by_type": {},
            "lowres_size": None,
        }

        def apply_lowres_sam3_overlay(frame: np.ndarray, frame_index: int) -> None:
            lowres_frame, _scale_x, _scale_y = _resize_for_sam3(frame, lowres_max_side)
            stats["lowres_size"] = [int(lowres_frame.shape[1]), int(lowres_frame.shape[0])]
            lowres_path = lowres_dir / f"{frame_index + 1:04d}.jpg"
            if not cv2.imwrite(str(lowres_path), lowres_frame):
                raise RuntimeError(f"could not write low-res frame: {lowres_path}")

            detect_start = time.perf_counter()
            detections = detect_pii(
                str(lowres_path),
                target_types,
                conf_threshold,
                include_binary_mask=True,
            )
            stats["sam3_detection_seconds"] += time.perf_counter() - detect_start
            stats["detections"] += len(detections)

            for obj in detections:
                pii_type = obj["type"]
                stats["detections_by_type"][pii_type] = (
                    stats["detections_by_type"].get(pii_type, 0) + 1
                )
                binary_mask = obj.get("binary_mask")
                if binary_mask is None:
                    continue
                highres_mask = _upscale_mask_to_frame(binary_mask, frame.shape)
                color_bgr = MASK_VISUAL_COLORS_BGR.get(pii_type, (0, 255, 255))
                _overlay_binary_mask(frame, highres_mask, color_bgr)
                _draw_mask_outline(frame, highres_mask, color_bgr)

        frames_dir = run_root / "frames"
        output_video = run_root / "jji_lowres_sam3_colored_mask_overlay.mp4"
        mask_write_seconds, frame_count = _write_overlay_frame_sequence(
            video_path,
            frames_dir,
            max_frames,
            apply_lowres_sam3_overlay,
            pass_index=True,
        )
        compose_seconds = _compose_timed(frames_dir, output_video, fps=fps)
        total_seconds = mask_write_seconds + compose_seconds
        self.assertTrue(output_video.exists())
        self.assertGreater(output_video.stat().st_size, 0)

        report = {
            "video": str(video_path),
            "metadata": metadata,
            "max_frames": max_frames,
            "lowres_max_side": lowres_max_side,
            "lowres_size": stats["lowres_size"],
            "target_types": target_types,
            "conf_threshold": conf_threshold,
            "run_root": str(run_root),
            "frames": frame_count,
            "detections": stats["detections"],
            "detections_by_type": stats["detections_by_type"],
            "sam3_detection_seconds": round(stats["sam3_detection_seconds"], 6),
            "mask_overlay_and_jpeg_write_seconds": round(mask_write_seconds, 6),
            "compose_seconds": round(compose_seconds, 6),
            "total_seconds": round(total_seconds, 6),
            "output_video": str(output_video),
            "output_bytes": output_video.stat().st_size,
            "note": (
                "Runs SAM3 on downscaled frames, upscales the returned binary masks "
                "to the source frame size, and renders colored mask overlay on the "
                "original-resolution frames."
            ),
        }
        report_path = run_root / "report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["report_path"] = str(report_path)
        print("\nJJI_LOWRES_SAM3_COLORED_MASK_OVERLAY " + json.dumps(report, ensure_ascii=False, sort_keys=True))

    def test_jji_first_frame_original_vs_lowres_sam3_overlay(self) -> None:
        if not sam3_available():
            self.skipTest("SAM3 checkpoint/package is not available in this environment.")

        video_path = Path(os.environ.get("MASK_BENCHMARK_VIDEO", DEFAULT_JJI_VIDEO))
        if not video_path.exists():
            self.skipTest(f"benchmark video not found: {video_path}")

        target_types = _parse_target_types(os.environ.get("MASK_BENCHMARK_TARGET_TYPES"))
        conf_threshold = float(
            os.environ.get("MASK_BENCHMARK_CONF_THRESHOLD", settings.SAM3_CONFIDENCE_THRESHOLD)
        )
        lowres_max_side = int(os.environ.get("MASK_BENCHMARK_LOWRES_MAX_SIDE", "960"))
        run_id = os.environ.get("MASK_BENCHMARK_RUN_ID") or time.strftime("%Y%m%d_%H%M%S")
        output_root = Path(os.environ.get("MASK_BENCHMARK_OUTPUT_ROOT", DEFAULT_BENCHMARK_OUTPUT_ROOT))
        run_root = output_root / f"jji_first_frame_compare_{run_id}"
        run_root.mkdir(parents=True, exist_ok=True)

        frame = _read_first_frame(video_path)

        source_frame_path = run_root / "jji_first_frame_original.jpg"
        _write_image(source_frame_path, frame)

        lowres_frame, _scale_x, _scale_y = _resize_for_sam3(frame, lowres_max_side)
        lowres_frame_path = run_root / "jji_first_frame_lowres_input.jpg"
        _write_image(lowres_frame_path, lowres_frame)

        warmup = _warm_sam3_for_benchmark(
            [
                ("original_first_frame", source_frame_path),
                ("lowres_first_frame", lowres_frame_path),
            ],
            target_types,
            conf_threshold,
        )

        original_overlay = frame.copy()
        original_start = time.perf_counter()
        original_detections = detect_pii(
            str(source_frame_path),
            target_types,
            conf_threshold,
            include_binary_mask=True,
        )
        original_seconds = time.perf_counter() - original_start
        original_counts = _overlay_detection_masks(original_overlay, original_detections)
        original_output = run_root / "jji_first_frame_original_sam3_overlay.jpg"
        _write_image(original_output, original_overlay)

        lowres_overlay = frame.copy()
        lowres_start = time.perf_counter()
        lowres_detections = detect_pii(
            str(lowres_frame_path),
            target_types,
            conf_threshold,
            include_binary_mask=True,
        )
        lowres_seconds = time.perf_counter() - lowres_start
        lowres_counts = _overlay_detection_masks(lowres_overlay, lowres_detections)
        lowres_output = run_root / "jji_first_frame_lowres_sam3_upscaled_overlay.jpg"
        _write_image(lowres_output, lowres_overlay)

        side_by_side = np.concatenate(
            [
                cv2.resize(original_overlay, (960, 540), interpolation=cv2.INTER_AREA),
                cv2.resize(lowres_overlay, (960, 540), interpolation=cv2.INTER_AREA),
            ],
            axis=1,
        )
        side_by_side_output = run_root / "jji_first_frame_original_vs_lowres_sam3.jpg"
        _write_image(side_by_side_output, side_by_side)

        report = {
            "video": str(video_path),
            "target_types": target_types,
            "conf_threshold": conf_threshold,
            "warmup": warmup,
            "source_size": [int(frame.shape[1]), int(frame.shape[0])],
            "lowres_max_side": lowres_max_side,
            "lowres_size": [int(lowres_frame.shape[1]), int(lowres_frame.shape[0])],
            "original": {
                "sam3_seconds": round(original_seconds, 6),
                "detections": len(original_detections),
                "detections_by_type": original_counts,
                "output_image": str(original_output),
            },
            "lowres_upscaled": {
                "sam3_seconds": round(lowres_seconds, 6),
                "detections": len(lowres_detections),
                "detections_by_type": lowres_counts,
                "output_image": str(lowres_output),
            },
            "comparison": {
                "sam3_speedup_x": round(original_seconds / lowres_seconds, 2)
                if lowres_seconds > 0 else None,
                "sam3_reduction_percent": round((1 - lowres_seconds / original_seconds) * 100, 1)
                if original_seconds > 0 else None,
            },
            "side_by_side_image": str(side_by_side_output),
        }
        report_path = run_root / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["report_path"] = str(report_path)
        print("\nJJI_FIRST_FRAME_ORIGINAL_VS_LOWRES_SAM3 " + json.dumps(report, ensure_ascii=False, sort_keys=True))

    def test_jji_full_video_original_vs_lowres_sam3_quality(self) -> None:
        if not sam3_available():
            self.skipTest("SAM3 checkpoint/package is not available in this environment.")

        video_path = Path(os.environ.get("MASK_BENCHMARK_VIDEO", DEFAULT_JJI_VIDEO))
        if not video_path.exists():
            self.skipTest(f"benchmark video not found: {video_path}")

        metadata = _video_metadata(video_path)
        fps = float(metadata["fps"])
        target_types = _parse_target_types(os.environ.get("MASK_BENCHMARK_TARGET_TYPES"))
        conf_threshold = float(
            os.environ.get("MASK_BENCHMARK_CONF_THRESHOLD", settings.SAM3_CONFIDENCE_THRESHOLD)
        )
        lowres_max_side = int(os.environ.get("MASK_BENCHMARK_LOWRES_MAX_SIDE", "960"))
        max_frames_env = os.environ.get("MASK_BENCHMARK_MAX_FRAMES")
        max_frames = int(max_frames_env) if max_frames_env is not None else 2
        run_id = os.environ.get("MASK_BENCHMARK_RUN_ID") or time.strftime("%Y%m%d_%H%M%S")
        output_root = Path(os.environ.get("MASK_BENCHMARK_OUTPUT_ROOT", DEFAULT_BENCHMARK_OUTPUT_ROOT))
        run_root = output_root / f"jji_full_compare_{run_id}"
        frames_dir = run_root / "side_by_side_frames"
        input_dir = run_root / "sam3_inputs"
        frames_dir.mkdir(parents=True, exist_ok=True)
        input_dir.mkdir(parents=True, exist_ok=True)
        for old_frame in frames_dir.glob("*.jpg"):
            old_frame.unlink()

        original_input = input_dir / "original_current.jpg"
        lowres_input = input_dir / "lowres_current.jpg"
        warmup_frame = _read_first_frame(video_path)
        original_warmup_input = input_dir / "original_warmup.jpg"
        lowres_warmup_input = input_dir / "lowres_warmup.jpg"
        _write_image(original_warmup_input, warmup_frame)
        lowres_warmup_frame, _scale_x, _scale_y = _resize_for_sam3(warmup_frame, lowres_max_side)
        _write_image(lowres_warmup_input, lowres_warmup_frame)
        warmup = _warm_sam3_for_benchmark(
            [
                ("original_first_frame", original_warmup_input),
                ("lowres_first_frame", lowres_warmup_input),
            ],
            target_types,
            conf_threshold,
        )

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.fail(f"could not open video: {video_path}")

        frame_metrics: list[dict] = []
        original_detection_seconds = 0.0
        lowres_detection_seconds = 0.0
        original_counts_total: dict[str, int] = {}
        lowres_counts_total: dict[str, int] = {}
        count_mismatch_frames = 0
        lowres_size: list[int] | None = None
        started = time.perf_counter()
        frame_count = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_index = frame_count
                frame_count += 1

                if not cv2.imwrite(str(original_input), frame):
                    raise RuntimeError(f"could not write frame: {original_input}")
                lowres_frame, _scale_x, _scale_y = _resize_for_sam3(frame, lowres_max_side)
                lowres_size = [int(lowres_frame.shape[1]), int(lowres_frame.shape[0])]
                if not cv2.imwrite(str(lowres_input), lowres_frame):
                    raise RuntimeError(f"could not write frame: {lowres_input}")

                original_start = time.perf_counter()
                original_detections = detect_pii(
                    str(original_input),
                    target_types,
                    conf_threshold,
                    include_binary_mask=True,
                )
                original_detection_seconds += time.perf_counter() - original_start

                lowres_start = time.perf_counter()
                lowres_detections = detect_pii(
                    str(lowres_input),
                    target_types,
                    conf_threshold,
                    include_binary_mask=True,
                )
                lowres_detection_seconds += time.perf_counter() - lowres_start

                original_overlay = frame.copy()
                lowres_overlay = frame.copy()
                original_counts = _overlay_detection_masks(original_overlay, original_detections)
                lowres_counts = _overlay_detection_masks(lowres_overlay, lowres_detections)
                for key, value in original_counts.items():
                    original_counts_total[key] = original_counts_total.get(key, 0) + value
                for key, value in lowres_counts.items():
                    lowres_counts_total[key] = lowres_counts_total.get(key, 0) + value

                if sum(original_counts.values()) != sum(lowres_counts.values()):
                    count_mismatch_frames += 1

                original_union = _union_detection_masks(original_detections, frame.shape)
                lowres_union = _union_detection_masks(lowres_detections, frame.shape)
                iou = _mask_iou(original_union, lowres_union)

                left = cv2.resize(original_overlay, (960, 540), interpolation=cv2.INTER_AREA)
                right = cv2.resize(lowres_overlay, (960, 540), interpolation=cv2.INTER_AREA)
                _put_label(left, f"original SAM3 f={frame_index:04d}", (24, 42))
                _put_label(right, f"lowres SAM3 {lowres_size[0]}x{lowres_size[1]} IoU={iou:.3f}", (24, 42))
                side_by_side = np.concatenate([left, right], axis=1)
                side_by_side_path = frames_dir / f"{frame_count:04d}.jpg"
                if not cv2.imwrite(str(side_by_side_path), side_by_side):
                    raise RuntimeError(f"could not write frame: {side_by_side_path}")

                frame_metrics.append({
                    "frame_index": frame_index,
                    "original_detections": len(original_detections),
                    "lowres_detections": len(lowres_detections),
                    "iou": round(iou, 6),
                })

                if max_frames > 0 and frame_count >= max_frames:
                    break
        finally:
            cap.release()

        compose_start = time.perf_counter()
        output_video = run_root / "jji_original_vs_lowres_sam3_quality.mp4"
        compose_seconds = _compose_timed(frames_dir, output_video, fps=fps)
        total_seconds = time.perf_counter() - started
        self.assertTrue(output_video.exists())
        self.assertGreater(output_video.stat().st_size, 0)

        ious = [item["iou"] for item in frame_metrics]
        report = {
            "video": str(video_path),
            "metadata": metadata,
            "frames": frame_count,
            "max_frames": max_frames,
            "target_types": target_types,
            "conf_threshold": conf_threshold,
            "warmup": warmup,
            "lowres_max_side": lowres_max_side,
            "lowres_size": lowres_size,
            "original_detection_seconds": round(original_detection_seconds, 6),
            "lowres_detection_seconds": round(lowres_detection_seconds, 6),
            "sam3_speedup_x": round(original_detection_seconds / lowres_detection_seconds, 2)
            if lowres_detection_seconds > 0 else None,
            "sam3_reduction_percent": round((1 - lowres_detection_seconds / original_detection_seconds) * 100, 1)
            if original_detection_seconds > 0 else None,
            "compose_seconds": round(compose_seconds, 6),
            "compose_started_after_seconds": round(compose_start - started, 6),
            "total_seconds": round(total_seconds, 6),
            "original_detections_by_type": original_counts_total,
            "lowres_detections_by_type": lowres_counts_total,
            "count_mismatch_frames": count_mismatch_frames,
            "mean_mask_iou": round(float(np.mean(ious)), 6) if ious else None,
            "min_mask_iou": round(float(np.min(ious)), 6) if ious else None,
            "median_mask_iou": round(float(np.median(ious)), 6) if ious else None,
            "output_video": str(output_video),
            "output_bytes": output_video.stat().st_size,
            "frame_metrics": frame_metrics,
        }
        report_path = run_root / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["report_path"] = str(report_path)
        print("\nJJI_FULL_VIDEO_ORIGINAL_VS_LOWRES_SAM3_QUALITY " + json.dumps(report, ensure_ascii=False, sort_keys=True))

    def test_jji_lowres_sam3_keyframe_interval_quality(self) -> None:
        if not sam3_available():
            self.skipTest("SAM3 checkpoint/package is not available in this environment.")

        video_path = Path(os.environ.get("MASK_BENCHMARK_VIDEO", DEFAULT_JJI_VIDEO))
        if not video_path.exists():
            self.skipTest(f"benchmark video not found: {video_path}")

        metadata = _video_metadata(video_path)
        fps = float(metadata["fps"])
        target_types = _parse_target_types(os.environ.get("MASK_BENCHMARK_TARGET_TYPES"))
        conf_threshold = float(
            os.environ.get("MASK_BENCHMARK_CONF_THRESHOLD", settings.SAM3_CONFIDENCE_THRESHOLD)
        )
        lowres_max_side = int(os.environ.get("MASK_BENCHMARK_LOWRES_MAX_SIDE", "960"))
        interval = int(os.environ.get("MASK_BENCHMARK_KEYFRAME_INTERVAL", "3"))
        if interval <= 0:
            raise ValueError(f"MASK_BENCHMARK_KEYFRAME_INTERVAL must be positive, got {interval}")
        dilate_px = int(os.environ.get("MASK_BENCHMARK_KEYFRAME_DILATE_PX", "0"))
        max_frames_env = os.environ.get("MASK_BENCHMARK_MAX_FRAMES")
        max_frames = int(max_frames_env) if max_frames_env is not None else 2
        run_id = os.environ.get("MASK_BENCHMARK_RUN_ID") or time.strftime("%Y%m%d_%H%M%S")
        output_root = Path(os.environ.get("MASK_BENCHMARK_OUTPUT_ROOT", DEFAULT_BENCHMARK_OUTPUT_ROOT))
        run_root = output_root / f"jji_keyframe_{interval}_{run_id}"
        frames_dir = run_root / "side_by_side_frames"
        input_dir = run_root / "sam3_inputs"
        frames_dir.mkdir(parents=True, exist_ok=True)
        input_dir.mkdir(parents=True, exist_ok=True)
        for old_frame in frames_dir.glob("*.jpg"):
            old_frame.unlink()

        warmup_frame = _read_first_frame(video_path)
        lowres_warmup_frame, _scale_x, _scale_y = _resize_for_sam3(warmup_frame, lowres_max_side)
        lowres_warmup_input = input_dir / "lowres_warmup.jpg"
        _write_image(lowres_warmup_input, lowres_warmup_frame)
        warmup = _warm_sam3_for_benchmark(
            [("lowres_first_frame", lowres_warmup_input)],
            target_types,
            conf_threshold,
        )

        lowres_input = input_dir / "lowres_current.jpg"
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.fail(f"could not open video: {video_path}")

        frame_metrics: list[dict] = []
        baseline_detection_seconds = 0.0
        keyframe_detection_seconds = 0.0
        baseline_counts_total: dict[str, int] = {}
        keyframe_counts_total: dict[str, int] = {}
        count_mismatch_frames = 0
        keyframe_count = 0
        lowres_size: list[int] | None = None
        last_keyframe_detections: list[dict] | None = None
        last_keyframe_index: int | None = None
        started = time.perf_counter()
        frame_count = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_index = frame_count
                frame_count += 1

                lowres_frame, _scale_x, _scale_y = _resize_for_sam3(frame, lowres_max_side)
                lowres_size = [int(lowres_frame.shape[1]), int(lowres_frame.shape[0])]
                _write_image(lowres_input, lowres_frame)

                baseline_start = time.perf_counter()
                baseline_detections = detect_pii(
                    str(lowres_input),
                    target_types,
                    conf_threshold,
                    include_binary_mask=True,
                )
                baseline_detection_seconds += time.perf_counter() - baseline_start

                is_keyframe = frame_index % interval == 0 or last_keyframe_detections is None
                if is_keyframe:
                    keyframe_start = time.perf_counter()
                    last_keyframe_detections = detect_pii(
                        str(lowres_input),
                        target_types,
                        conf_threshold,
                        include_binary_mask=True,
                    )
                    keyframe_detection_seconds += time.perf_counter() - keyframe_start
                    last_keyframe_index = frame_index
                    keyframe_count += 1

                keyframe_detections = last_keyframe_detections or []

                baseline_overlay = frame.copy()
                keyframe_overlay = frame.copy()
                baseline_counts = _overlay_detection_masks(baseline_overlay, baseline_detections)
                keyframe_counts = _overlay_detection_masks(
                    keyframe_overlay,
                    keyframe_detections,
                    dilate_px=dilate_px,
                )
                for key, value in baseline_counts.items():
                    baseline_counts_total[key] = baseline_counts_total.get(key, 0) + value
                for key, value in keyframe_counts.items():
                    keyframe_counts_total[key] = keyframe_counts_total.get(key, 0) + value

                if sum(baseline_counts.values()) != sum(keyframe_counts.values()):
                    count_mismatch_frames += 1

                baseline_union = _union_detection_masks(baseline_detections, frame.shape)
                keyframe_union_raw = _union_detection_masks(keyframe_detections, frame.shape)
                keyframe_union = _union_detection_masks(
                    keyframe_detections,
                    frame.shape,
                    dilate_px=dilate_px,
                )
                raw_iou = _mask_iou(baseline_union, keyframe_union_raw)
                padded_iou = _mask_iou(baseline_union, keyframe_union)
                coverage = _mask_coverage(baseline_union, keyframe_union)
                area_ratio = _mask_area_ratio(baseline_union, keyframe_union)

                left = cv2.resize(baseline_overlay, (960, 540), interpolation=cv2.INTER_AREA)
                right = cv2.resize(keyframe_overlay, (960, 540), interpolation=cv2.INTER_AREA)
                _put_label(left, f"lowres every frame f={frame_index:04d}", (24, 42))
                _put_label(
                    right,
                    f"every {interval}f from {last_keyframe_index:04d} cov={coverage:.3f}",
                    (24, 42),
                )
                side_by_side = np.concatenate([left, right], axis=1)
                side_by_side_path = frames_dir / f"{frame_count:04d}.jpg"
                _write_image(side_by_side_path, side_by_side)

                frame_metrics.append({
                    "frame_index": frame_index,
                    "source_keyframe_index": last_keyframe_index,
                    "is_keyframe": is_keyframe,
                    "baseline_detections": len(baseline_detections),
                    "keyframe_detections": len(keyframe_detections),
                    "raw_iou": round(raw_iou, 6),
                    "padded_iou": round(padded_iou, 6),
                    "baseline_coverage": round(coverage, 6),
                    "padded_area_ratio": round(area_ratio, 6) if area_ratio is not None else None,
                })

                if max_frames > 0 and frame_count >= max_frames:
                    break
        finally:
            cap.release()

        output_video = run_root / f"jji_lowres_every_frame_vs_keyframe_{interval}.mp4"
        compose_seconds = _compose_timed(frames_dir, output_video, fps=fps)
        total_seconds = time.perf_counter() - started
        self.assertTrue(output_video.exists())
        self.assertGreater(output_video.stat().st_size, 0)

        raw_ious = [item["raw_iou"] for item in frame_metrics]
        padded_ious = [item["padded_iou"] for item in frame_metrics]
        coverages = [item["baseline_coverage"] for item in frame_metrics]
        area_ratios = [
            item["padded_area_ratio"]
            for item in frame_metrics
            if item["padded_area_ratio"] is not None
        ]
        report = {
            "video": str(video_path),
            "metadata": metadata,
            "frames": frame_count,
            "max_frames": max_frames,
            "target_types": target_types,
            "conf_threshold": conf_threshold,
            "warmup": warmup,
            "lowres_max_side": lowres_max_side,
            "lowres_size": lowres_size,
            "keyframe_interval": interval,
            "keyframe_update_fps": round(fps / interval, 6),
            "keyframe_dilate_px": dilate_px,
            "keyframes_detected": keyframe_count,
            "baseline_lowres_detection_seconds": round(baseline_detection_seconds, 6),
            "keyframe_detection_seconds": round(keyframe_detection_seconds, 6),
            "sam3_speedup_x": round(baseline_detection_seconds / keyframe_detection_seconds, 2)
            if keyframe_detection_seconds > 0 else None,
            "sam3_reduction_percent": round((1 - keyframe_detection_seconds / baseline_detection_seconds) * 100, 1)
            if baseline_detection_seconds > 0 else None,
            "compose_seconds": round(compose_seconds, 6),
            "total_seconds": round(total_seconds, 6),
            "baseline_detections_by_type": baseline_counts_total,
            "keyframe_detections_by_type": keyframe_counts_total,
            "count_mismatch_frames": count_mismatch_frames,
            "mean_raw_iou": round(float(np.mean(raw_ious)), 6) if raw_ious else None,
            "min_raw_iou": round(float(np.min(raw_ious)), 6) if raw_ious else None,
            "mean_padded_iou": round(float(np.mean(padded_ious)), 6) if padded_ious else None,
            "min_padded_iou": round(float(np.min(padded_ious)), 6) if padded_ious else None,
            "mean_baseline_coverage": round(float(np.mean(coverages)), 6) if coverages else None,
            "min_baseline_coverage": round(float(np.min(coverages)), 6) if coverages else None,
            "frames_below_95_coverage": sum(1 for value in coverages if value < 0.95),
            "mean_padded_area_ratio": round(float(np.mean(area_ratios)), 6) if area_ratios else None,
            "output_video": str(output_video),
            "output_bytes": output_video.stat().st_size,
            "frame_metrics": frame_metrics,
        }
        report_path = run_root / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["report_path"] = str(report_path)
        print("\nJJI_LOWRES_SAM3_KEYFRAME_INTERVAL_QUALITY " + json.dumps(report, ensure_ascii=False, sort_keys=True))

    def test_jji_lowres_sam3_keyframe_interpolated_quality(self) -> None:
        if not sam3_available():
            self.skipTest("SAM3 checkpoint/package is not available in this environment.")

        video_path = Path(os.environ.get("MASK_BENCHMARK_VIDEO", DEFAULT_JJI_VIDEO))
        if not video_path.exists():
            self.skipTest(f"benchmark video not found: {video_path}")

        metadata = _video_metadata(video_path)
        fps = float(metadata["fps"])
        target_types = _parse_target_types(os.environ.get("MASK_BENCHMARK_TARGET_TYPES"))
        conf_threshold = float(
            os.environ.get("MASK_BENCHMARK_CONF_THRESHOLD", settings.SAM3_CONFIDENCE_THRESHOLD)
        )
        lowres_max_side = int(os.environ.get("MASK_BENCHMARK_LOWRES_MAX_SIDE", "960"))
        interval = int(os.environ.get("MASK_BENCHMARK_KEYFRAME_INTERVAL", "3"))
        if interval <= 0:
            raise ValueError(f"MASK_BENCHMARK_KEYFRAME_INTERVAL must be positive, got {interval}")
        dilate_px = int(os.environ.get("MASK_BENCHMARK_KEYFRAME_DILATE_PX", "24"))
        max_frames_env = os.environ.get("MASK_BENCHMARK_MAX_FRAMES")
        max_frames = int(max_frames_env) if max_frames_env is not None else 2
        run_id = os.environ.get("MASK_BENCHMARK_RUN_ID") or time.strftime("%Y%m%d_%H%M%S")
        output_root = Path(os.environ.get("MASK_BENCHMARK_OUTPUT_ROOT", DEFAULT_BENCHMARK_OUTPUT_ROOT))
        run_root = output_root / f"jji_keyframe_interp_{interval}_{run_id}"
        frames_dir = run_root / "side_by_side_frames"
        input_dir = run_root / "sam3_inputs"
        frames_dir.mkdir(parents=True, exist_ok=True)
        input_dir.mkdir(parents=True, exist_ok=True)
        for old_frame in frames_dir.glob("*.jpg"):
            old_frame.unlink()

        warmup_frame = _read_first_frame(video_path)
        lowres_warmup_frame, _scale_x, _scale_y = _resize_for_sam3(warmup_frame, lowres_max_side)
        lowres_warmup_input = input_dir / "lowres_warmup.jpg"
        _write_image(lowres_warmup_input, lowres_warmup_frame)
        warmup = _warm_sam3_for_benchmark(
            [("lowres_first_frame", lowres_warmup_input)],
            target_types,
            conf_threshold,
        )

        lowres_input = input_dir / "lowres_current.jpg"

        def detect_lowres(frame: np.ndarray) -> tuple[list[dict], float, list[int]]:
            lowres_frame, _scale_x, _scale_y = _resize_for_sam3(frame, lowres_max_side)
            lowres_size = [int(lowres_frame.shape[1]), int(lowres_frame.shape[0])]
            _write_image(lowres_input, lowres_frame)
            start = time.perf_counter()
            detections = detect_pii(
                str(lowres_input),
                target_types,
                conf_threshold,
                include_binary_mask=True,
            )
            return detections, time.perf_counter() - start, lowres_size

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.fail(f"could not open video: {video_path}")

        ok, previous_keyframe = cap.read()
        if not ok:
            cap.release()
            self.fail(f"could not read first frame: {video_path}")

        previous_keyframe_index = 0
        previous_keyframe_detections, detect_seconds, lowres_size = detect_lowres(previous_keyframe)
        baseline_detection_seconds = detect_seconds
        interpolated_detection_seconds = detect_seconds
        keyframe_count = 1
        frame_count = 0
        output_count = 0
        frame_metrics: list[dict] = []
        baseline_counts_total: dict[str, int] = {}
        interpolated_counts_total: dict[str, int] = {}
        count_mismatch_frames = 0
        started = time.perf_counter()

        def render_frame(
            frame_index: int,
            frame: np.ndarray,
            baseline_detections: list[dict],
            interpolated_detections: list[dict],
            *,
            source_keyframe_index: int,
            next_keyframe_index: int | None,
            is_keyframe: bool,
        ) -> None:
            nonlocal output_count, count_mismatch_frames
            baseline_overlay = frame.copy()
            interpolated_overlay = frame.copy()
            baseline_counts = _overlay_detection_masks(baseline_overlay, baseline_detections)
            interpolated_counts = _overlay_detection_masks(
                interpolated_overlay,
                interpolated_detections,
                dilate_px=dilate_px,
            )
            for key, value in baseline_counts.items():
                baseline_counts_total[key] = baseline_counts_total.get(key, 0) + value
            for key, value in interpolated_counts.items():
                interpolated_counts_total[key] = interpolated_counts_total.get(key, 0) + value

            if sum(baseline_counts.values()) != sum(interpolated_counts.values()):
                count_mismatch_frames += 1

            baseline_union = _union_detection_masks(baseline_detections, frame.shape)
            interpolated_union_raw = _union_detection_masks(interpolated_detections, frame.shape)
            interpolated_union = _union_detection_masks(
                interpolated_detections,
                frame.shape,
                dilate_px=dilate_px,
            )
            raw_iou = _mask_iou(baseline_union, interpolated_union_raw)
            padded_iou = _mask_iou(baseline_union, interpolated_union)
            coverage = _mask_coverage(baseline_union, interpolated_union)
            area_ratio = _mask_area_ratio(baseline_union, interpolated_union)

            left = cv2.resize(baseline_overlay, (960, 540), interpolation=cv2.INTER_AREA)
            right = cv2.resize(interpolated_overlay, (960, 540), interpolation=cv2.INTER_AREA)
            _put_label(left, f"lowres every frame f={frame_index:04d}", (24, 42))
            next_label = "tail" if next_keyframe_index is None else f"{next_keyframe_index:04d}"
            _put_label(
                right,
                f"interp {source_keyframe_index:04d}->{next_label} cov={coverage:.3f}",
                (24, 42),
            )
            side_by_side = np.concatenate([left, right], axis=1)
            output_count += 1
            _write_image(frames_dir / f"{output_count:04d}.jpg", side_by_side)

            frame_metrics.append({
                "frame_index": frame_index,
                "source_keyframe_index": source_keyframe_index,
                "next_keyframe_index": next_keyframe_index,
                "is_keyframe": is_keyframe,
                "baseline_detections": len(baseline_detections),
                "interpolated_detections": len(interpolated_detections),
                "raw_iou": round(raw_iou, 6),
                "padded_iou": round(padded_iou, 6),
                "baseline_coverage": round(coverage, 6),
                "padded_area_ratio": round(area_ratio, 6) if area_ratio is not None else None,
            })

        try:
            first_segment = True
            while True:
                segment: list[tuple[int, np.ndarray, list[dict], bool]] = []
                if first_segment:
                    segment.append((
                        previous_keyframe_index,
                        previous_keyframe,
                        previous_keyframe_detections,
                        True,
                    ))
                    frame_count = 1

                next_keyframe_index: int | None = None
                next_keyframe_frame: np.ndarray | None = None
                next_keyframe_detections: list[dict] | None = None

                for step in range(1, interval + 1):
                    if max_frames > 0 and frame_count >= max_frames:
                        break
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frame_index = previous_keyframe_index + step
                    baseline_detections, detect_seconds, lowres_size = detect_lowres(frame)
                    baseline_detection_seconds += detect_seconds
                    frame_count += 1
                    is_next_keyframe = step == interval
                    if is_next_keyframe:
                        # Reuse the baseline detection at the keyframe. This is the
                        # exact SAM3 call the optimized path would have made.
                        interpolated_detection_seconds += detect_seconds
                        keyframe_count += 1
                        next_keyframe_index = frame_index
                        next_keyframe_frame = frame
                        next_keyframe_detections = baseline_detections
                    segment.append((frame_index, frame, baseline_detections, is_next_keyframe))

                for frame_index, frame, baseline_detections, is_keyframe in segment:
                    if frame_index == previous_keyframe_index:
                        interpolated_detections = previous_keyframe_detections
                    elif next_keyframe_index is not None and next_keyframe_detections is not None:
                        alpha = (frame_index - previous_keyframe_index) / float(
                            next_keyframe_index - previous_keyframe_index
                        )
                        interpolated_detections = _interpolate_detection_masks(
                            previous_keyframe_detections,
                            next_keyframe_detections,
                            alpha,
                        )
                    else:
                        interpolated_detections = previous_keyframe_detections

                    render_frame(
                        frame_index,
                        frame,
                        baseline_detections,
                        interpolated_detections,
                        source_keyframe_index=previous_keyframe_index,
                        next_keyframe_index=next_keyframe_index,
                        is_keyframe=is_keyframe,
                    )

                if next_keyframe_index is None or next_keyframe_frame is None or next_keyframe_detections is None:
                    break
                previous_keyframe_index = next_keyframe_index
                previous_keyframe = next_keyframe_frame
                previous_keyframe_detections = next_keyframe_detections
                first_segment = False
        finally:
            cap.release()

        output_video = run_root / f"jji_lowres_every_frame_vs_interpolated_keyframe_{interval}.mp4"
        compose_seconds = _compose_timed(frames_dir, output_video, fps=fps)
        total_seconds = time.perf_counter() - started
        self.assertTrue(output_video.exists())
        self.assertGreater(output_video.stat().st_size, 0)

        raw_ious = [item["raw_iou"] for item in frame_metrics]
        padded_ious = [item["padded_iou"] for item in frame_metrics]
        coverages = [item["baseline_coverage"] for item in frame_metrics]
        area_ratios = [
            item["padded_area_ratio"]
            for item in frame_metrics
            if item["padded_area_ratio"] is not None
        ]
        report = {
            "video": str(video_path),
            "metadata": metadata,
            "frames": output_count,
            "max_frames": max_frames,
            "target_types": target_types,
            "conf_threshold": conf_threshold,
            "warmup": warmup,
            "lowres_max_side": lowres_max_side,
            "lowres_size": lowres_size,
            "keyframe_interval": interval,
            "keyframe_update_fps": round(fps / interval, 6),
            "keyframe_dilate_px": dilate_px,
            "keyframes_detected": keyframe_count,
            "baseline_lowres_detection_seconds": round(baseline_detection_seconds, 6),
            "interpolated_detection_seconds": round(interpolated_detection_seconds, 6),
            "sam3_speedup_x": round(baseline_detection_seconds / interpolated_detection_seconds, 2)
            if interpolated_detection_seconds > 0 else None,
            "sam3_reduction_percent": round((1 - interpolated_detection_seconds / baseline_detection_seconds) * 100, 1)
            if baseline_detection_seconds > 0 else None,
            "compose_seconds": round(compose_seconds, 6),
            "total_seconds": round(total_seconds, 6),
            "baseline_detections_by_type": baseline_counts_total,
            "interpolated_detections_by_type": interpolated_counts_total,
            "count_mismatch_frames": count_mismatch_frames,
            "mean_raw_iou": round(float(np.mean(raw_ious)), 6) if raw_ious else None,
            "min_raw_iou": round(float(np.min(raw_ious)), 6) if raw_ious else None,
            "mean_padded_iou": round(float(np.mean(padded_ious)), 6) if padded_ious else None,
            "min_padded_iou": round(float(np.min(padded_ious)), 6) if padded_ious else None,
            "mean_baseline_coverage": round(float(np.mean(coverages)), 6) if coverages else None,
            "min_baseline_coverage": round(float(np.min(coverages)), 6) if coverages else None,
            "frames_below_95_coverage": sum(1 for value in coverages if value < 0.95),
            "mean_padded_area_ratio": round(float(np.mean(area_ratios)), 6) if area_ratios else None,
            "output_video": str(output_video),
            "output_bytes": output_video.stat().st_size,
            "frame_metrics": frame_metrics,
        }
        report_path = run_root / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["report_path"] = str(report_path)
        print("\nJJI_LOWRES_SAM3_KEYFRAME_INTERPOLATED_QUALITY " + json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
