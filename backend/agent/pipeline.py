"""
Two-phase pipeline:
  Phase 1 (detection): GPT-4o → SAM3 + InsightFace detect & cluster → awaiting_selection
  Phase 2 (masking):   per-frame SAM3 pixel mask + InsightFace identity check → done
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from .job_store import FaceCluster, PIICandidate, get_store
from .log_emitter import emit_log, write_status
from .report_builder import build_final_report, build_intermediate_scene_analysis, write_report
from .tools.face_engine import (
    bbox_iou,
    cluster_embeddings,
    detect_faces,
    find_best_cluster,
)
from .tools.guideline_generator import generate_guideline
from .tools.masker import apply_binary_mask, apply_polygon_mask, blur_bbox
from .tools.sam3_engine import detect_pii
from .tools.scene_analyzer import analyze_scene
from .tools.temporal_masks import (
    dilate_binary_mask,
    interpolate_detection_masks,
    match_detections_by_motion,
    resize_for_sam3,
    scale_bbox_to_frame,
    upscale_mask_to_frame,
)
from .tools.video_tools import compose_video, extract_frames, get_video_fps
from ..config import settings
from ..models.sam3_loader import is_available as sam3_available

MASK_COLOR_HEX = {
    "face":          "#a855f7",
    "document":      "#f59e0b",
    "screen":        "#06b6d4",
    "nameplate":     "#ef4444",
    "id_card":       "#10b981",
    "license_plate": "#f97316",
    "brand_logo":    "#ec4899",
}


# ---------------------------------------------------------------------------
# Phase 1 — Detection & clustering
# ---------------------------------------------------------------------------

def run_detection_phase(job_id: str) -> None:
    store = get_store(job_id)
    store.status = "detecting"
    write_status(job_id, "detecting")

    try:
        # 1. Extract frames at SAMPLE_FPS for detection only
        native_fps = get_video_fps(store.video_path)
        store.native_fps = native_fps
        frames_dir = settings.upload_path / job_id / "frames"
        frames = extract_frames(store.video_path, frames_dir)
        store.frames_dir = str(frames_dir)
        emit_log(job_id, {
            "step": "extract",
            "message": f"{len(frames)}개 프레임 추출 완료 (원본 {native_fps:.2f}fps)",
        })

        # 2. Analyze scene (GPT-4o) — evenly-sampled multi-frame
        n = min(settings.SCENE_ANALYSIS_FRAMES, len(frames))
        sample_indices = list(dict.fromkeys(
            min(int(len(frames) * i / n), len(frames) - 1) for i in range(n)
        ))
        emit_log(job_id, {
            "step": "scene",
            "message": f"씬 분석 중 ({len(sample_indices)}개 프레임 병렬 처리)...",
        })

        scene_results: list[dict] = []
        scene_type_votes: list[str] = []
        expected_pii_union: set[str] = set()

        with ThreadPoolExecutor(max_workers=len(sample_indices)) as pool:
            futures = {pool.submit(analyze_scene, str(frames[i])): i for i in sample_indices}
            for fut in as_completed(futures):
                result = fut.result()
                result["frame_index"] = futures[fut]
                scene_results.append(result)
                scene_type_votes.append(result.get("scene_type", "other"))
                expected_pii_union.update(result.get("expected_pii", []))

        scene_type  = Counter(scene_type_votes).most_common(1)[0][0]
        expected_pii = list(expected_pii_union)
        detection_pii_types = _pii_types_for_detection(expected_pii)
        deterministic_pii_types_added = [
            pii_type for pii_type in detection_pii_types if pii_type not in expected_pii
        ]
        store.scene_type  = scene_type
        store.expected_pii = expected_pii
        store.scene_analysis = build_intermediate_scene_analysis(
            sorted(scene_results, key=lambda r: r.get("frame_index", 0)),
            scene_type,
            expected_pii,
        )
        store.detection_pii_types = detection_pii_types
        store.deterministic_pii_types_added = deterministic_pii_types_added
        emit_log(job_id, {
            "step": "scene",
            "message": f"씬 분석 완료: {scene_type} → {expected_pii} ({len(sample_indices)}프레임 합산)",
        })

        thumb_dir = settings.upload_path / job_id / "thumbnails"
        thumb_dir.mkdir(parents=True, exist_ok=True)

        # 3. SAM3 detect PII on 5 evenly-spaced frames (face + non-face)
        # sam3_detections: list of (frame_path, detection_dict)
        sam3_detections: list[tuple[int, dict]] = []  # (frame_idx, face_obj) for thumbnail matching
        sam3_frame_detections: list[tuple] = []       # (frame_idx, frame_path, obj) for all detections
        if sam3_available() and detection_pii_types:
            n_samples = min(5, len(frames))
            sample_indices = [int(len(frames) * i / n_samples) for i in range(n_samples)]
            total_detected = 0
            for si in sample_indices:
                detections = detect_pii(
                    str(frames[si]),
                    detection_pii_types,
                    settings.SAM3_CONFIDENCE_THRESHOLD,
                )
                for obj in detections:
                    sam3_frame_detections.append((si, frames[si], obj))
                    if obj["type"] == "face":
                        sam3_detections.append((si, obj))
                total_detected += len(detections)
            emit_log(job_id, {
                "step": "sam3",
                "message": f"SAM3 탐지 완료: {n_samples}개 샘플 프레임, {total_detected}개 객체",
            })

            # Save example thumbnails for each non-face PII category.  Full
            # video masking is category/type-level, not per-object: SAM3 image
            # detection does not provide stable object IDs across frames.
            pii_candidates: list[PIICandidate] = []
            for frame_idx, frame_path, obj in sam3_frame_detections:
                if obj["type"] == "face":
                    continue
                pii_type = obj["type"]
                object_id = len(pii_candidates)
                thumb_name = f"pii_{object_id}_{pii_type}.jpg"
                if not _save_pii_context_thumbnail(frame_path, obj, thumb_dir / thumb_name):
                    continue
                pii_candidates.append(PIICandidate(
                    object_id=object_id,
                    pii_type=pii_type,
                    thumbnail=thumb_name,
                    confidence=obj["confidence"],
                    frame_index=frame_idx,
                    bbox_xyxy=[float(v) for v in obj["bbox_xyxy"]],
                    mask_strategy=obj.get("mask_strategy"),
                ))
            store.pii_candidates = pii_candidates
            emit_log(job_id, {
                "step": "pii",
                "message": f"비얼굴 PII {len(pii_candidates)}개 후보 추출",
            })

        # 4. InsightFace detect + cluster (faces only)
        face_clusters: list[FaceCluster] = []
        cluster_embs: dict[int, list] = {}

        if "face" in expected_pii:
            all_embeddings: list[list[float]] = []
            face_records: list[tuple] = []  # (frame_idx, bbox_xyxy, emb_idx, score)

            interval = settings.FACE_DETECT_INTERVAL
            for frame_idx, fp in enumerate(frames):
                if frame_idx % interval != 0:
                    continue
                for face in detect_faces(str(fp)):
                    idx = len(all_embeddings)
                    all_embeddings.append(face["embedding"])
                    face_records.append((frame_idx, face["bbox_xyxy"], idx, face["score"]))

            emit_log(job_id, {
                "step": "insightface",
                "message": f"InsightFace: {len(all_embeddings)}개 얼굴 감지",
            })

            if all_embeddings:
                labels = cluster_embeddings(all_embeddings)
                cluster_dict: dict[int, list[tuple]] = {}
                for rec_idx, record in enumerate(face_records):
                    label = int(labels[rec_idx])
                    cluster_dict.setdefault(label, []).append(record)

                emit_log(job_id, {
                    "step": "cluster",
                    "message": f"{len(cluster_dict)}명의 인물 구분 완료",
                })

                for label, records in sorted(cluster_dict.items()):
                    best = max(records, key=lambda r: r[3])
                    frame_idx, bbox_xyxy, emb_idx, _ = best

                    # Thumbnail: match SAM3 face polygon if available, else use bbox crop
                    img = cv2.imread(str(frames[frame_idx]))
                    thumb_name = f"face_{label}.jpg"
                    if img is not None:
                        # Try to use SAM3 polygon crop for higher quality thumbnail
                        sam3_face = _find_sam3_face(bbox_xyxy, sam3_detections, frame_idx)
                        if sam3_face and sam3_face.get("polygon"):
                            # Crop from polygon bounding box
                            x1, y1, x2, y2 = sam3_face["bbox_xyxy"]
                        else:
                            x1, y1, x2, y2 = (int(v) for v in bbox_xyxy)
                        pad = 20
                        h, w = img.shape[:2]
                        crop = img[
                            max(0, y1 - pad):min(h, y2 + pad),
                            max(0, x1 - pad):min(w, x2 + pad),
                        ]
                        cv2.imwrite(str(thumb_dir / thumb_name), crop)

                    cluster_embs[label] = [all_embeddings[r[2]] for r in records]
                    face_clusters.append(FaceCluster(
                        cluster_id=label,
                        thumbnail=thumb_name,
                        count=len(records),
                        frame_index=frame_idx,
                        bbox_xyxy=[float(v) for v in bbox_xyxy],
                    ))

        store.face_clusters = face_clusters
        store.cluster_embeddings = cluster_embs

        # Generate guideline before handing off to user
        store.status = "generating_guideline"
        write_status(job_id, "generating_guideline")
        emit_log(job_id, {"step": "guideline", "message": "편집 가이드라인 생성 중..."})
        guideline = generate_guideline(
            scene_type=store.scene_type or "other",
            expected_pii=store.expected_pii,
            face_clusters=face_clusters,
            pii_candidates=store.pii_candidates,
        )

        # Prepend algorithmic borderline-cluster warnings (no GPT-4o needed)
        borderline = _find_borderline_clusters(cluster_embs, settings.FACE_SIMILARITY_THRESHOLD)
        for cid_a, cid_b, sim in borderline:
            guideline.insert(0, {
                "level":    "warning",
                "category": "face",
                "message":  f"인물 {cid_a + 1}번과 {cid_b + 1}번이 동일인일 수 있습니다 (유사도 {sim:.2f})",
            })
        if borderline:
            emit_log(job_id, {
                "step": "guideline",
                "message": f"경계값 얼굴 클러스터 {len(borderline)}쌍 감지됨",
            })

        store.guideline = guideline
        emit_log(job_id, {
            "step": "guideline",
            "message": f"가이드라인 {len(guideline)}개 항목 생성 완료",
        })

        store.status = "awaiting_selection"
        write_status(job_id, "awaiting_selection")
        emit_log(job_id, {"step": "ready", "message": "인물/PII 선택을 기다리고 있습니다"})

    except Exception as exc:
        store.status = "failed"
        store.error = str(exc)
        write_status(job_id, "failed", str(exc))
        emit_log(job_id, {"step": "error", "message": str(exc)})


def _save_pii_context_thumbnail(frame_path: Path, obj: dict, out_path: Path) -> bool:
    """Save a zoomed-in PII thumbnail with only a red outline on the target.

    The review UI should not show the whole frame, but a focused crop around the
    detected object.  Keep a small amount of nearby background for orientation
    and draw the exact SAM polygon (or bbox fallback) in red.
    """
    img = cv2.imread(str(frame_path))
    if img is None:
        return False

    img_h, img_w = img.shape[:2]
    bbox = _clamp_bbox(obj.get("bbox_xyxy", []), img_w, img_h)
    if bbox is None:
        return False
    x1, y1, x2, y2 = bbox

    left, top, right, bottom = _focused_crop_bounds(bbox, img_w, img_h)
    crop = img[top:bottom, left:right].copy()
    if crop.size == 0:
        return False

    thumb_w, thumb_h = 320, 200
    scale_x = thumb_w / max(1, right - left)
    scale_y = thumb_h / max(1, bottom - top)
    thumb = cv2.resize(crop, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

    polygon = obj.get("polygon")
    if polygon and len(polygon) >= 3:
        pts = np.array([
            [
                [
                    int(np.clip((point[0] - left) * scale_x, 0, thumb_w - 1)),
                    int(np.clip((point[1] - top) * scale_y, 0, thumb_h - 1)),
                ]
                for point in polygon
            ]
        ], dtype=np.int32)
        cv2.polylines(thumb, pts, isClosed=True, color=(0, 0, 255), thickness=4, lineType=cv2.LINE_AA)
    else:
        pt1 = (
            int(np.clip((x1 - left) * scale_x, 0, thumb_w - 1)),
            int(np.clip((y1 - top) * scale_y, 0, thumb_h - 1)),
        )
        pt2 = (
            int(np.clip((x2 - left) * scale_x, 0, thumb_w - 1)),
            int(np.clip((y2 - top) * scale_y, 0, thumb_h - 1)),
        )
        cv2.rectangle(thumb, pt1, pt2, color=(0, 0, 255), thickness=4, lineType=cv2.LINE_AA)

    return bool(cv2.imwrite(str(out_path), thumb))


def _clamp_bbox(bbox_xyxy: list | tuple, img_w: int, img_h: int) -> tuple[int, int, int, int] | None:
    if len(bbox_xyxy) != 4:
        return None
    x1, y1, x2, y2 = (int(round(float(v))) for v in bbox_xyxy)
    x1 = int(np.clip(x1, 0, img_w - 1))
    y1 = int(np.clip(y1, 0, img_h - 1))
    x2 = int(np.clip(x2, x1 + 1, img_w))
    y2 = int(np.clip(y2, y1 + 1, img_h))
    return x1, y1, x2, y2


def _focused_crop_bounds(
    bbox: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    target_aspect: float = 1.6,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # Tight enough to enlarge the target area, but not so tight that the red
    # outline loses nearby cues.  The aspect correction below preserves the
    # review-card thumbnail ratio.
    crop_w = max(box_w * 2.2, 120.0)
    crop_h = max(box_h * 2.2, 75.0)
    if crop_w / crop_h < target_aspect:
        crop_w = crop_h * target_aspect
    else:
        crop_h = crop_w / target_aspect

    crop_w = min(crop_w, float(img_w))
    crop_h = min(crop_h, float(img_h))

    left = int(round(cx - crop_w / 2))
    top = int(round(cy - crop_h / 2))
    left = max(0, min(left, img_w - int(round(crop_w))))
    top = max(0, min(top, img_h - int(round(crop_h))))
    right = min(img_w, left + int(round(crop_w)))
    bottom = min(img_h, top + int(round(crop_h)))
    return left, top, right, bottom


def _find_borderline_clusters(
    cluster_embs: dict[int, list],
    threshold: float,
    margin: float = 0.12,
) -> list[tuple[int, int, float]]:
    """Return (cid_a, cid_b, similarity) for cluster pairs near the merge threshold.

    These are clusters that almost got merged — they may be the same person
    captured at different angles or lighting conditions.
    """
    ids = list(cluster_embs.keys())
    if len(ids) < 2:
        return []

    centroids = {
        cid: np.mean(embs, axis=0) / (np.linalg.norm(np.mean(embs, axis=0)) + 1e-9)
        for cid, embs in cluster_embs.items() if embs
    }

    borderline: list[tuple[int, int, float]] = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            sim = float(np.dot(centroids[ids[i]], centroids[ids[j]]))
            if threshold - margin <= sim < threshold:
                borderline.append((ids[i], ids[j], sim))

    return sorted(borderline, key=lambda x: -x[2])


def _find_sam3_face(
    insightface_bbox: list[float],
    sam3_detections: list[tuple[int, dict]],
    target_frame_idx: int,
) -> dict | None:
    """Find the SAM3 face detection that best overlaps with an InsightFace bbox.

    sam3_detections is a list of (frame_idx, obj) so we can match only
    detections from the same frame as the InsightFace best-score detection.
    """
    best_iou, best = 0.0, None
    for frame_idx, obj in sam3_detections:
        if frame_idx != target_frame_idx:
            continue
        iou = bbox_iou(insightface_bbox, obj["bbox_xyxy"])
        if iou > best_iou:
            best_iou, best = iou, obj
    return best if best_iou > 0.3 else None



# ---------------------------------------------------------------------------
# Phase 2 — Per-frame masking
# ---------------------------------------------------------------------------

def run_masking_phase(job_id: str) -> None:
    store = get_store(job_id)
    store.status = "masking"
    write_status(job_id, "masking")

    try:
        # Extract all frames at native fps for full-quality output.
        native_fps = getattr(store, "native_fps", None) or get_video_fps(store.video_path)
        all_frames_dir = settings.upload_path / job_id / "all_frames"
        frames = extract_frames(store.video_path, all_frames_dir, fps=native_fps)

        masked_dir = settings.upload_path / job_id / "masked_frames"
        masked_dir.mkdir(parents=True, exist_ok=True)
        for old_frame in masked_dir.glob("*.jpg"):
            old_frame.unlink()

        mask_preview_dir = settings.upload_path / job_id / "mask_preview_frames"
        mask_preview_dir.mkdir(parents=True, exist_ok=True)
        for old_frame in mask_preview_dir.glob("*.jpg"):
            old_frame.unlink()

        selected_pii_types = _selected_pii_types(store)
        pii_selection = _build_pii_selection(store)
        # The downloadable output must stay privacy-preserving only
        # (blur/pixelate/blackbox/ambient_fill). Colored masks are generated as a separate
        # completed-state preview video so users can see what was affected.
        output_overlay = False

        pii_types_to_detect: list[str] = []
        if "face" in store.expected_pii:
            pii_types_to_detect.append("face")
        pii_types_to_detect += selected_pii_types
        colored_preview_enabled = bool(pii_types_to_detect)

        use_keyframe_interpolation = (
            bool(settings.SAM3_MASK_INTERPOLATION_ENABLED)
            and sam3_available()
            and bool(pii_types_to_detect)
        )

        if use_keyframe_interpolation:
            emit_log(job_id, {
                "step": "mask",
                "message": (
                    f"{len(frames)}개 프레임 low-res SAM3 keyframe 보간 마스킹 시작 "
                    f"({native_fps:.2f}fps, {settings.SAM3_MASK_KEYFRAME_INTERVAL}프레임 간격)"
                ),
            })
            total_faces, total_pii, masking_stats = _mask_frames_with_keyframe_interpolation(
                job_id=job_id,
                store=store,
                frames=frames,
                masked_dir=masked_dir,
                preview_dir=mask_preview_dir if colored_preview_enabled else None,
                selected_pii_types=selected_pii_types,
                pii_selection=pii_selection,
                pii_types_to_detect=pii_types_to_detect,
                debug_overlay=output_overlay,
            )
        else:
            emit_log(job_id, {
                "step": "mask",
                "message": f"{len(frames)}개 프레임 per-frame SAM3 마스킹 시작 ({native_fps:.2f}fps)",
            })
            total_faces, total_pii, masking_stats = _mask_frames_per_frame(
                job_id=job_id,
                store=store,
                frames=frames,
                masked_dir=masked_dir,
                preview_dir=mask_preview_dir if colored_preview_enabled else None,
                selected_pii_types=selected_pii_types,
                pii_selection=pii_selection,
                pii_types_to_detect=pii_types_to_detect,
                debug_overlay=output_overlay,
            )

        store.masked_frames_dir = str(masked_dir)
        store.mask_preview_frames_dir = str(mask_preview_dir) if colored_preview_enabled else None
        store.total_faces_blurred = total_faces
        store.total_pii_masked = total_pii
        emit_log(job_id, {
            "step": "mask",
            "message": f"마스킹 완료 — 얼굴 {total_faces}개, 비얼굴 PII {total_pii}개",
        })

        # Compose video at native fps.
        out_dir = settings.output_path / job_id
        out_path = out_dir / "output.mp4"
        compose_video(masked_dir, out_path, fps=native_fps)
        store.output_video_path = str(out_path)

        preview_path = out_dir / "mask_preview.mp4" if colored_preview_enabled else None
        if preview_path is not None:
            compose_video(mask_preview_dir, preview_path, fps=native_fps)
            store.mask_preview_video_path = str(preview_path)
        else:
            store.mask_preview_video_path = None
        emit_log(job_id, {"step": "compose", "message": "영상 합성 완료"})

        report = build_final_report(
            store=store,
            job_id=job_id,
            total_faces_blurred=total_faces,
            total_pii_masked=total_pii,
            output_video_path=str(out_path),
        )
        report.update({
            "detection_pii_types":                      _detection_pii_types_for_report(store),
            "deterministic_pii_types_added":            _deterministic_pii_types_added(store),
            "masked_pii_object_ids":                    sorted(pii_selection["selected_object_ids"]),
            "selected_pii_category_count":              len(selected_pii_types),
            "selected_pii_object_count":                len(pii_selection["selected_object_ids"]),
            "total_pii_candidates_detected":            len(store.pii_candidates),
            "colored_mask_enabled":                     colored_preview_enabled,
            "colored_mask_preview_enabled":             colored_preview_enabled,
            "mask_preview_max_side":                    settings.MASK_PREVIEW_MAX_SIDE if colored_preview_enabled else None,
            "debug_mask_overlay_enabled":               False,
            "sam3_video_tracking_enabled":              False,
            "temporal_mask_cache_enabled":              False,
            "temporal_mask_interpolation_enabled":      use_keyframe_interpolation,
            "lowres_sam3_keyframe_interpolation_enabled": use_keyframe_interpolation,
            **masking_stats,
            "mask_colors": {
                k: MASK_COLOR_HEX[k]
                for k in ["face", *selected_pii_types]
                if k in MASK_COLOR_HEX
            } if colored_preview_enabled else {},
            "mask_preview_video_path": str(preview_path) if preview_path is not None else None,
        })
        write_report(report, str(out_path))
        store.report = report

        store.status = "done"
        write_status(job_id, "done")
        emit_log(job_id, {"event": "done", "message": "처리 완료! 영상을 다운로드하세요."})

    except Exception as exc:
        store.status = "failed"
        store.error = str(exc)
        write_status(job_id, "failed", str(exc))
        emit_log(job_id, {"step": "error", "message": str(exc)})


def _mask_frames_per_frame(
    *,
    job_id: str,
    store,
    frames: list[Path],
    masked_dir: Path,
    preview_dir: Path | None,
    selected_pii_types: list[str],
    pii_selection: dict,
    pii_types_to_detect: list[str],
    debug_overlay: bool,
) -> tuple[int, int, dict]:
    """Original safe fallback: run SAM3 on every output frame."""
    total_faces = 0
    total_pii = 0
    sam3_ready = sam3_available()

    for idx, fp in enumerate(frames):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        preview_ops: list[dict] | None = [] if preview_dir is not None else None

        sam3_objs: list[dict] = []
        if sam3_ready and pii_types_to_detect:
            sam3_objs = detect_pii(
                str(fp),
                pii_types_to_detect,
                settings.SAM3_CONFIDENCE_THRESHOLD,
            )

        face_count, pii_count = _apply_sam3_detections_to_frame(
            img=img,
            frame_path=fp,
            detections=sam3_objs,
            store=store,
            selected_pii_types=selected_pii_types,
            pii_selection=pii_selection,
            debug_overlay=debug_overlay,
            preview_ops=preview_ops,
            dilate_px=0,
        )
        total_faces += face_count
        total_pii += pii_count

        if "face" in store.expected_pii and not sam3_ready:
            for face in detect_faces(str(fp)):
                cid = find_best_cluster(
                    face["embedding"],
                    store.cluster_embeddings,
                    settings.FACE_SIMILARITY_THRESHOLD,
                )
                if cid is None or cid not in store.protected_face_cluster_ids:
                    img = blur_bbox(
                        img,
                        face["bbox_xyxy"],
                        overlay_color=_overlay_color("face", debug_overlay),
                    )
                    if preview_ops is not None:
                        preview_ops.append({
                            "kind": "bbox",
                            "bbox_xyxy": face["bbox_xyxy"],
                            "pii_type": "face",
                        })
                    total_faces += 1

        cv2.imwrite(str(masked_dir / fp.name), img)
        if preview_dir is not None:
            _write_colored_preview_frame(preview_dir, fp.name, img, preview_ops or [])
        _emit_mask_progress(job_id, idx + 1, len(frames))

    mode = "per_frame_sam3" if sam3_ready and pii_types_to_detect else "insightface_bbox_fallback"
    return total_faces, total_pii, {
        "masking_mode": mode,
        "sam3_mask_lowres_max_side": None,
        "sam3_mask_keyframe_interval": 1 if mode == "per_frame_sam3" else None,
        "sam3_mask_keyframe_dilate_px": 0,
        "sam3_mask_keyframes_detected": len(frames) if mode == "per_frame_sam3" else 0,
    }


def _mask_frames_with_keyframe_interpolation(
    *,
    job_id: str,
    store,
    frames: list[Path],
    masked_dir: Path,
    preview_dir: Path | None,
    selected_pii_types: list[str],
    pii_selection: dict,
    pii_types_to_detect: list[str],
    debug_overlay: bool,
) -> tuple[int, int, dict]:
    """Run SAM3 on low-res keyframes and interpolate masks for in-between frames."""
    if not frames:
        return 0, 0, {
            "masking_mode": "lowres_keyframe_interpolation",
            "sam3_mask_lowres_max_side": settings.SAM3_MASK_LOWRES_MAX_SIDE,
            "sam3_mask_keyframe_interval": settings.SAM3_MASK_KEYFRAME_INTERVAL,
            "sam3_mask_keyframe_dilate_px": settings.SAM3_MASK_DILATE_PX,
            "sam3_mask_keyframes_detected": 0,
        }

    interval = max(1, int(settings.SAM3_MASK_KEYFRAME_INTERVAL))
    dilate_px = max(0, int(settings.SAM3_MASK_DILATE_PX))
    input_dir = settings.upload_path / job_id / "sam3_mask_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in input_dir.glob("*.jpg"):
        old_frame.unlink()

    total_faces = 0
    total_pii = 0
    processed = 0
    keyframes_detected = 0

    previous_keyframe_index = 0
    previous_keyframe_path = frames[0]
    previous_keyframe_frame = cv2.imread(str(previous_keyframe_path))
    if previous_keyframe_frame is None:
        raise RuntimeError(f"Could not read frame: {previous_keyframe_path}")
    previous_keyframe_detections = _detect_lowres_keyframe(
        frame_path=previous_keyframe_path,
        frame=previous_keyframe_frame,
        input_dir=input_dir,
        pii_types_to_detect=pii_types_to_detect,
    )
    _tag_selected_pii_detections(
        previous_keyframe_detections,
        previous_keyframe_frame.shape,
        pii_selection,
    )
    keyframes_detected += 1

    first_segment = True
    while True:
        segment: list[tuple[int, Path, np.ndarray]] = []
        if first_segment:
            segment.append((previous_keyframe_index, previous_keyframe_path, previous_keyframe_frame))

        next_keyframe_index: int | None = None
        next_keyframe_path: Path | None = None
        next_keyframe_frame: np.ndarray | None = None
        next_keyframe_detections: list[dict] | None = None

        for step in range(1, interval + 1):
            frame_index = previous_keyframe_index + step
            if frame_index >= len(frames):
                break

            frame_path = frames[frame_index]
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            segment.append((frame_index, frame_path, frame))
            is_interval_keyframe = step == interval
            is_tail_keyframe = frame_index == len(frames) - 1
            if is_interval_keyframe or is_tail_keyframe:
                next_keyframe_index = frame_index
                next_keyframe_path = frame_path
                next_keyframe_frame = frame
                next_keyframe_detections = _detect_lowres_keyframe(
                    frame_path=frame_path,
                    frame=frame,
                    input_dir=input_dir,
                    pii_types_to_detect=pii_types_to_detect,
                )
                _tag_selected_pii_detections(
                    next_keyframe_detections,
                    frame.shape,
                    pii_selection,
                    previous_detections=previous_keyframe_detections,
                )
                keyframes_detected += 1
                break

        for frame_index, frame_path, frame in segment:
            preview_ops: list[dict] | None = [] if preview_dir is not None else None
            if frame_index == previous_keyframe_index:
                detections = previous_keyframe_detections
            elif next_keyframe_index is not None and next_keyframe_detections is not None:
                alpha = (frame_index - previous_keyframe_index) / float(
                    max(1, next_keyframe_index - previous_keyframe_index)
                )
                detections = interpolate_detection_masks(
                    previous_keyframe_detections,
                    next_keyframe_detections,
                    alpha,
                )
            else:
                detections = previous_keyframe_detections

            face_count, pii_count = _apply_sam3_detections_to_frame(
                img=frame,
                frame_path=frame_path,
                detections=detections,
                store=store,
                selected_pii_types=selected_pii_types,
                pii_selection=pii_selection,
                debug_overlay=debug_overlay,
                preview_ops=preview_ops,
                dilate_px=dilate_px,
            )
            total_faces += face_count
            total_pii += pii_count

            cv2.imwrite(str(masked_dir / frame_path.name), frame)
            if preview_dir is not None:
                _write_colored_preview_frame(preview_dir, frame_path.name, frame, preview_ops or [])
            processed += 1
            _emit_mask_progress(job_id, processed, len(frames))

        if next_keyframe_index is None or next_keyframe_path is None or next_keyframe_frame is None:
            break

        previous_keyframe_index = next_keyframe_index
        previous_keyframe_path = next_keyframe_path
        previous_keyframe_frame = next_keyframe_frame
        previous_keyframe_detections = next_keyframe_detections or []
        first_segment = False

    return total_faces, total_pii, {
        "masking_mode": "lowres_keyframe_interpolation",
        "sam3_mask_lowres_max_side": settings.SAM3_MASK_LOWRES_MAX_SIDE,
        "sam3_mask_keyframe_interval": interval,
        "sam3_mask_keyframe_dilate_px": dilate_px,
        "sam3_mask_keyframes_detected": keyframes_detected,
    }


def _detect_lowres_keyframe(
    *,
    frame_path: Path,
    frame: np.ndarray,
    input_dir: Path,
    pii_types_to_detect: list[str],
) -> list[dict]:
    lowres_frame, _scale_x, _scale_y = resize_for_sam3(
        frame,
        int(settings.SAM3_MASK_LOWRES_MAX_SIDE),
    )
    lowres_path = input_dir / frame_path.name
    if not cv2.imwrite(str(lowres_path), lowres_frame):
        raise RuntimeError(f"Could not write SAM3 low-res input: {lowres_path}")

    return detect_pii(
        str(lowres_path),
        pii_types_to_detect,
        settings.SAM3_CONFIDENCE_THRESHOLD,
        include_binary_mask=True,
    )


def _apply_sam3_detections_to_frame(
    *,
    img: np.ndarray,
    frame_path: Path,
    detections: list[dict],
    store,
    selected_pii_types: list[str],
    pii_selection: dict,
    debug_overlay: bool,
    preview_ops: list[dict] | None,
    dilate_px: int,
) -> tuple[int, int]:
    face_count = 0
    pii_count = 0

    face_objs = [obj for obj in detections if obj.get("type") == "face"]
    if "face" in store.expected_pii and face_objs:
        full_faces = [] if not store.protected_face_cluster_ids else detect_faces(str(frame_path))
        for obj in face_objs:
            if _should_mask_face_detection(obj, img.shape, full_faces, store):
                _apply_detection_mask(
                    img,
                    obj,
                    strategy="blur",
                    pii_type="face",
                    debug_overlay=debug_overlay,
                    dilate_px=dilate_px,
                )
                if preview_ops is not None:
                    preview_ops.append({
                        "kind": "detection",
                        "obj": obj,
                        "pii_type": "face",
                        "dilate_px": dilate_px,
                    })
                face_count += 1

    for obj in detections:
        pii_type = obj.get("type")
        if pii_type == "face":
            continue
        if _should_mask_pii_detection(obj, img.shape, pii_selection):
            _apply_detection_mask(
                img,
                obj,
                strategy=obj.get("mask_strategy", "blackbox"),
                pii_type=pii_type,
                debug_overlay=debug_overlay,
                dilate_px=dilate_px,
            )
            if preview_ops is not None:
                preview_ops.append({
                    "kind": "detection",
                    "obj": obj,
                    "pii_type": pii_type,
                    "dilate_px": dilate_px,
                })
            pii_count += 1

    return face_count, pii_count


def _write_colored_preview_frame(
    preview_dir: Path,
    frame_name: str,
    masked_img: np.ndarray,
    preview_ops: list[dict],
) -> None:
    """Write a lightweight colored preview derived from the private output.

    The preview starts from the already-masked frame so it does not expose the
    original pixels. We only add colored overlays here; blur/pixelate/blackbox/ambient_fill
    have already been applied once to the downloadable output frame.
    """
    preview = masked_img.copy()
    for op in preview_ops:
        pii_type = op.get("pii_type", "document")
        if op.get("kind") == "bbox":
            blur_bbox(
                preview,
                op.get("bbox_xyxy", []),
                strength=1,
                overlay_color=_overlay_color(pii_type, True),
            )
            continue
        if op.get("kind") == "detection":
            _apply_detection_mask(
                preview,
                op["obj"],
                strategy="overlay_only",
                pii_type=pii_type,
                debug_overlay=True,
                dilate_px=int(op.get("dilate_px", 0)),
            )

    max_side = int(getattr(settings, "MASK_PREVIEW_MAX_SIDE", 0) or 0)
    if max_side > 0 and max(preview.shape[:2]) > max_side:
        preview, _scale_x, _scale_y = resize_for_sam3(preview, max_side)
    cv2.imwrite(str(preview_dir / frame_name), preview)


def _build_pii_selection(store) -> dict:
    """Build non-face PII selection state.

    Backward-compatible behavior:
    - no ``masked_pii_object_ids`` means type/category selection masks all
      detections of each selected type.
    - when object IDs are present, each selected thumbnail is treated as an
      individual anchor. If all candidates of a type are selected, that type is
      promoted back to all-type masking.
    """
    selected_object_ids = set(getattr(store, "masked_pii_object_ids", []) or [])
    type_level = set(getattr(store, "masked_pii_types", []) or [])
    candidates_by_type: dict[str, list[PIICandidate]] = {}
    selected_candidates_by_type: dict[str, list[PIICandidate]] = {}

    for candidate in store.pii_candidates:
        candidates_by_type.setdefault(candidate.pii_type, []).append(candidate)
        if candidate.object_id in selected_object_ids:
            selected_candidates_by_type.setdefault(candidate.pii_type, []).append(candidate)

    all_types: set[str] = set()
    if not selected_object_ids:
        all_types.update(type_level)
    else:
        for pii_type, candidates in candidates_by_type.items():
            candidate_ids = {candidate.object_id for candidate in candidates}
            selected_ids = {
                candidate.object_id
                for candidate in selected_candidates_by_type.get(pii_type, [])
            }
            if candidate_ids and candidate_ids.issubset(selected_ids):
                all_types.add(pii_type)
        for pii_type in type_level:
            if pii_type not in candidates_by_type:
                all_types.add(pii_type)

    return {
        "selected_object_ids": selected_object_ids,
        "all_types": all_types,
        "selected_candidates_by_type": selected_candidates_by_type,
        "object_mode": bool(selected_object_ids),
    }


def _tag_selected_pii_detections(
    detections: list[dict],
    frame_shape: tuple[int, ...],
    pii_selection: dict,
    *,
    previous_detections: list[dict] | None = None,
) -> None:
    """Attach selected PII object IDs to current keyframe detections in-place."""
    if not pii_selection["object_mode"]:
        return

    selected_candidates_by_type: dict[str, list[PIICandidate]] = pii_selection[
        "selected_candidates_by_type"
    ]
    for pii_type, candidates in selected_candidates_by_type.items():
        type_detections = [
            (idx, obj)
            for idx, obj in enumerate(detections)
            if obj.get("type") == pii_type and pii_type not in pii_selection["all_types"]
        ]
        matches = _match_candidates_to_detections(
            candidates,
            [obj for _idx, obj in type_detections],
            frame_shape,
        )
        for candidate_index, detection_index in matches:
            original_index = type_detections[detection_index][0]
            detections[original_index]["selected_pii_object_id"] = candidates[candidate_index].object_id

    if previous_detections:
        previous_selected = [
            obj
            for obj in previous_detections
            if obj.get("selected_pii_object_id") in pii_selection["selected_object_ids"]
        ]
        untagged_current = [
            obj
            for obj in detections
            if obj.get("selected_pii_object_id") is None
            and obj.get("type") not in pii_selection["all_types"]
        ]
        for previous_idx, current_idx in match_detections_by_motion(
            previous_selected,
            untagged_current,
            _sample_mask_shape(previous_selected, untagged_current),
        ):
            untagged_current[current_idx]["selected_pii_object_id"] = previous_selected[
                previous_idx
            ].get("selected_pii_object_id")


def _sample_mask_shape(*groups: list[dict]) -> tuple[int, int]:
    for group in groups:
        for obj in group:
            mask = obj.get("binary_mask")
            if mask is not None:
                return mask.shape[:2]
    return (1, 1)


def _should_mask_pii_detection(
    obj: dict,
    frame_shape: tuple[int, ...],
    pii_selection: dict,
) -> bool:
    pii_type = obj.get("type")
    if pii_type in pii_selection["all_types"]:
        return True
    if not pii_selection["object_mode"]:
        return False
    if obj.get("selected_pii_object_id") in pii_selection["selected_object_ids"]:
        return True

    candidates = pii_selection["selected_candidates_by_type"].get(pii_type, [])
    return bool(_match_candidates_to_detections(candidates, [obj], frame_shape))


def _match_candidates_to_detections(
    candidates: list[PIICandidate],
    detections: list[dict],
    frame_shape: tuple[int, ...],
) -> list[tuple[int, int]]:
    """Greedily match selected thumbnail anchors to current detections."""
    if not candidates or not detections:
        return []

    h, w = frame_shape[:2]
    diag = max(1.0, float((w * w + h * h) ** 0.5))
    scored: list[tuple[float, int, int]] = []
    for candidate_index, candidate in enumerate(candidates):
        if not candidate.bbox_xyxy:
            continue
        candidate_center = _bbox_center(candidate.bbox_xyxy)
        for detection_index, detection in enumerate(detections):
            detection_bbox = scale_bbox_to_frame(detection, frame_shape)
            if detection_bbox is None:
                continue
            detection_center = _bbox_center(detection_bbox)
            distance = float(
                (
                    (candidate_center[0] - detection_center[0]) ** 2
                    + (candidate_center[1] - detection_center[1]) ** 2
                )
                ** 0.5
            )
            distance_score = max(0.0, 1.0 - distance / (diag * 0.35))
            iou = bbox_iou(candidate.bbox_xyxy, detection_bbox)
            score = (iou * 2.0) + distance_score
            if iou >= 0.05 or distance_score >= 0.35:
                scored.append((score, candidate_index, detection_index))

    matches: list[tuple[int, int]] = []
    used_candidates: set[int] = set()
    used_detections: set[int] = set()
    for _score, candidate_index, detection_index in sorted(scored, reverse=True):
        if candidate_index in used_candidates or detection_index in used_detections:
            continue
        used_candidates.add(candidate_index)
        used_detections.add(detection_index)
        matches.append((candidate_index, detection_index))
    return matches


def _bbox_center(bbox_xyxy: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)


def _should_mask_face_detection(
    obj: dict,
    frame_shape: tuple[int, ...],
    full_faces: list[dict],
    store,
) -> bool:
    if not store.protected_face_cluster_ids:
        return True

    obj_bbox = scale_bbox_to_frame(obj, frame_shape)
    if obj_bbox is None:
        return True

    best_iou, best_emb = 0.0, None
    for face in full_faces:
        iou = bbox_iou(obj_bbox, face["bbox_xyxy"])
        if iou > best_iou:
            best_iou, best_emb = iou, face["embedding"]
    emb = best_emb if best_iou > 0.3 else None

    cid = (
        find_best_cluster(emb, store.cluster_embeddings, settings.FACE_SIMILARITY_THRESHOLD)
        if emb is not None else None
    )
    return cid is None or cid not in store.protected_face_cluster_ids


def _apply_detection_mask(
    img: np.ndarray,
    obj: dict,
    *,
    strategy: str,
    pii_type: str,
    debug_overlay: bool,
    dilate_px: int,
) -> None:
    binary_mask = obj.get("binary_mask")
    if binary_mask is not None:
        frame_mask = upscale_mask_to_frame(binary_mask, img.shape)
        if dilate_px > 0:
            frame_mask = dilate_binary_mask(frame_mask, dilate_px)
        apply_binary_mask(
            img,
            frame_mask,
            strategy,
            overlay_color=_overlay_color(pii_type, debug_overlay),
        )
        return

    apply_polygon_mask(
        img,
        obj.get("polygon"),
        strategy,
        overlay_color=_overlay_color(pii_type, debug_overlay),
    )


def _emit_mask_progress(job_id: str, processed: int, total: int) -> None:
    if processed % 10 == 0 or processed == total:
        emit_log(job_id, {
            "step": "mask",
            "message": f"{processed}/{total} 프레임 처리 중...",
        })


def _mask_color_bgr(pii_type: str) -> tuple[int, int, int]:
    hex_color = MASK_COLOR_HEX.get(pii_type, "#f59e0b").lstrip("#")
    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)
    return blue, green, red


def _overlay_color(pii_type: str, enabled: bool) -> tuple[int, int, int] | None:
    return _mask_color_bgr(pii_type) if enabled else None


def _selected_pii_types(store) -> list[str]:
    selected = list(getattr(store, "masked_pii_types", []) or [])
    selected_object_ids = set(getattr(store, "masked_pii_object_ids", []) or [])
    for candidate in store.pii_candidates:
        if candidate.object_id in selected_object_ids:
            selected.append(candidate.pii_type)
    return list(dict.fromkeys(selected))


def _pii_types_for_detection(expected_pii: list[str]) -> list[str]:
    """Return PII types SAM3 should consider during candidate discovery.

    Brand logos are privacy/compliance-sensitive enough that discovery should
    not rely solely on scene-analyzer recall. Keeping this as a feature flag
    preserves a cheap opt-out for local benchmarking or prompt tuning.
    """
    pii_types = list(expected_pii or [])
    if settings.BRAND_LOGO_DETECTION_ENABLED and "brand_logo" not in pii_types:
        pii_types.append("brand_logo")
    return pii_types


def _detection_pii_types_for_report(store) -> list[str]:
    detection_pii_types = getattr(store, "detection_pii_types", None)
    if detection_pii_types is not None:
        return list(detection_pii_types)
    return _pii_types_for_detection(getattr(store, "expected_pii", []) or [])


def _deterministic_pii_types_added(store) -> list[str]:
    added = getattr(store, "deterministic_pii_types_added", None)
    if added is not None:
        return list(added)
    detection_pii_types = _detection_pii_types_for_report(store)
    expected_pii = set(getattr(store, "expected_pii", []) or [])
    return [pii_type for pii_type in detection_pii_types if pii_type not in expected_pii]
