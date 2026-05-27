"""
Two-phase pipeline:
  Phase 1 (detection): GPT-4o → SAM3 + InsightFace detect & cluster → awaiting_selection
  Phase 2 (masking):   per-frame SAM3 pixel mask + InsightFace identity check → done
"""
from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from .job_store import FaceCluster, PIICandidate, get_store
from .log_emitter import emit_log, write_status
from .tools.face_engine import (
    bbox_iou,
    cluster_embeddings,
    detect_faces,
    find_best_cluster,
)
from .tools.guideline_generator import generate_guideline
from .tools.masker import apply_polygon_mask, blur_bbox
from .tools.sam3_engine import detect_pii
from .tools.scene_analyzer import analyze_scene
from .tools.video_tools import compose_video, extract_frames, get_video_fps
from ..config import settings
from ..models.sam3_loader import is_available as sam3_available


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
        emit_log(job_id, {"step": "extract", "message": f"{len(frames)}개 프레임 추출 완료 (원본 {native_fps:.2f}fps)"})

        # 2. Analyze scene (GPT-4o) — evenly-sampled multi-frame
        n = min(settings.SCENE_ANALYSIS_FRAMES, len(frames))
        sample_indices = list(dict.fromkeys(
            min(int(len(frames) * i / n), len(frames) - 1) for i in range(n)
        ))
        emit_log(job_id, {
            "step": "scene",
            "message": f"씬 분석 중 ({len(sample_indices)}개 프레임 병렬 처리)...",
        })

        scene_type_votes: list[str] = []
        expected_pii_union: set[str] = set()

        with ThreadPoolExecutor(max_workers=len(sample_indices)) as pool:
            futures = {pool.submit(analyze_scene, str(frames[i])): i for i in sample_indices}
            for fut in as_completed(futures):
                st, pii = fut.result()
                scene_type_votes.append(st)
                expected_pii_union.update(pii)

        scene_type  = Counter(scene_type_votes).most_common(1)[0][0]
        expected_pii = list(expected_pii_union)
        store.scene_type  = scene_type
        store.expected_pii = expected_pii
        emit_log(job_id, {
            "step": "scene",
            "message": f"씬 분석 완료: {scene_type} → {expected_pii} ({len(sample_indices)}프레임 합산)",
        })

        thumb_dir = settings.upload_path / job_id / "thumbnails"
        thumb_dir.mkdir(parents=True, exist_ok=True)

        # 3. SAM3 detect PII on 5 evenly-spaced frames (face + non-face)
        # sam3_detections: list of (frame_path, detection_dict)
        sam3_detections: list[tuple[int, dict]] = []  # (frame_idx, face_obj) for thumbnail matching
        sam3_frame_detections: list[tuple] = []       # (frame_path, obj) for all detections
        if sam3_available() and expected_pii:
            n_samples = min(5, len(frames))
            sample_indices = [int(len(frames) * i / n_samples) for i in range(n_samples)]
            total_detected = 0
            for si in sample_indices:
                detections = detect_pii(
                    str(frames[si]),
                    expected_pii,
                    settings.SAM3_CONFIDENCE_THRESHOLD,
                )
                for obj in detections:
                    sam3_frame_detections.append((frames[si], obj))
                    if obj["type"] == "face":
                        sam3_detections.append((si, obj))
                total_detected += len(detections)
            emit_log(job_id, {
                "step": "sam3",
                "message": f"SAM3 탐지 완료: {n_samples}개 샘플 프레임, {total_detected}개 객체",
            })

            # Save best thumbnail per PII type (highest confidence across sampled frames)
            best_per_type: dict[str, tuple] = {}  # type → (frame_path, obj)
            for frame_path, obj in sam3_frame_detections:
                if obj["type"] == "face":
                    continue
                t = obj["type"]
                if t not in best_per_type or obj["confidence"] > best_per_type[t][1]["confidence"]:
                    best_per_type[t] = (frame_path, obj)

            pii_candidates: list[PIICandidate] = []
            for pii_type, (frame_path, obj) in best_per_type.items():
                x1, y1, x2, y2 = obj["bbox_xyxy"]
                img = cv2.imread(str(frame_path))
                if img is not None:
                    crop = img[max(0, y1):y2, max(0, x1):x2]
                    thumb_name = f"pii_{len(pii_candidates)}.jpg"
                    cv2.imwrite(str(thumb_dir / thumb_name), crop)
                    pii_candidates.append(PIICandidate(
                        object_id=len(pii_candidates),
                        pii_type=pii_type,
                        thumbnail=thumb_name,
                        confidence=obj["confidence"],
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
                "message": f"경계값 클러스터 {len(borderline)}쌍 감지됨",
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
        # Extract all frames at native fps for full-quality output
        native_fps = getattr(store, "native_fps", None) or get_video_fps(store.video_path)
        all_frames_dir = settings.upload_path / job_id / "all_frames"
        frames = extract_frames(store.video_path, all_frames_dir, fps=native_fps)

        masked_dir = settings.upload_path / job_id / "masked_frames"
        masked_dir.mkdir(parents=True, exist_ok=True)

        emit_log(job_id, {"step": "mask", "message": f"{len(frames)}개 프레임 마스킹 시작 ({native_fps:.2f}fps)"})

        total_faces = 0
        total_pii = 0

        # Determine what to detect per frame with SAM3
        pii_types_to_detect = []
        if "face" in store.expected_pii:
            pii_types_to_detect.append("face")
        pii_types_to_detect += store.masked_pii_types

        # Cache last-known polygon per cluster_id / pii_type for temporal propagation.
        # If SAM3 misses a frame the cached polygon is reused for up to PII_CACHE_TTL frames.
        PII_CACHE_TTL = 30
        face_cache: dict[int, dict] = {}  # cluster_id → {polygon, last_seen}
        pii_cache: dict[str, dict] = {}   # pii_type   → {polygon, mask_strategy, last_seen}

        for idx, fp in enumerate(frames):
            img = cv2.imread(str(fp))
            if img is None:
                continue

            # SAM3 detect all relevant PII in this frame
            sam3_objs: list[dict] = []
            if sam3_available() and pii_types_to_detect:
                sam3_objs = detect_pii(
                    str(fp),
                    pii_types_to_detect,
                    settings.SAM3_CONFIDENCE_THRESHOLD,
                )

            # — Face masking —
            # Strategy: SAM3 polygon (high quality) + InsightFace identity check
            face_objs = [o for o in sam3_objs if o["type"] == "face"]
            if "face" in store.expected_pii:
                masked_cids: set[int] = set()

                if face_objs:
                    full_faces = detect_faces(str(fp))
                    for obj in face_objs:
                        best_iou, best_emb = 0.0, None
                        for f in full_faces:
                            iou = bbox_iou(obj["bbox_xyxy"], f["bbox_xyxy"])
                            if iou > best_iou:
                                best_iou, best_emb = iou, f["embedding"]
                        emb = best_emb if best_iou > 0.3 else None

                        cid = (
                            find_best_cluster(emb, store.cluster_embeddings, settings.FACE_SIMILARITY_THRESHOLD)
                            if emb is not None else None
                        )

                        # Update face cache for every identified face (protected or not)
                        if cid is not None and obj["polygon"] is not None:
                            face_cache[cid] = {"polygon": obj["polygon"], "last_seen": idx}

                        if cid is None or cid not in store.protected_face_cluster_ids:
                            img = apply_polygon_mask(img, obj["polygon"], "blur")
                            total_faces += 1
                            if cid is not None:
                                masked_cids.add(cid)
                elif sam3_available():
                    pass  # SAM3 available but detected no faces this frame — use cache below
                else:
                    # SAM3 not available → InsightFace bbox fallback (no cache)
                    for face in detect_faces(str(fp)):
                        cid = find_best_cluster(
                            face["embedding"],
                            store.cluster_embeddings,
                            settings.FACE_SIMILARITY_THRESHOLD,
                        )
                        if cid is None or cid not in store.protected_face_cluster_ids:
                            img = blur_bbox(img, face["bbox_xyxy"])
                            total_faces += 1

                # TTL fallback: reuse cached polygon for faces SAM3 missed this frame
                for cid, cached in face_cache.items():
                    if cid in masked_cids:
                        continue  # already masked this frame
                    if cid in store.protected_face_cluster_ids:
                        continue  # protected, never mask
                    if (idx - cached["last_seen"]) <= PII_CACHE_TTL:
                        img = apply_polygon_mask(img, cached["polygon"], "blur")
                        total_faces += 1

            # — Non-face PII masking —
            detected_pii_types: set[str] = set()
            for obj in sam3_objs:
                if obj["type"] == "face":
                    continue
                if obj["type"] in store.masked_pii_types:
                    img = apply_polygon_mask(img, obj["polygon"], obj["mask_strategy"])
                    total_pii += 1
                    detected_pii_types.add(obj["type"])
                    pii_cache[obj["type"]] = {
                        "polygon":       obj["polygon"],
                        "mask_strategy": obj["mask_strategy"],
                        "last_seen":     idx,
                    }

            # Fallback: propagate cached polygon for PII types SAM3 missed this frame
            for pii_type in store.masked_pii_types:
                if pii_type in detected_pii_types:
                    continue
                cached = pii_cache.get(pii_type)
                if cached and (idx - cached["last_seen"]) <= PII_CACHE_TTL:
                    img = apply_polygon_mask(img, cached["polygon"], cached["mask_strategy"])
                    total_pii += 1

            cv2.imwrite(str(masked_dir / fp.name), img)

            if (idx + 1) % 10 == 0 or idx + 1 == len(frames):
                emit_log(job_id, {
                    "step": "mask",
                    "message": f"{idx + 1}/{len(frames)} 프레임 처리 중...",
                })

        store.masked_frames_dir = str(masked_dir)
        store.total_faces_blurred = total_faces
        store.total_pii_masked = total_pii
        emit_log(job_id, {
            "step": "mask",
            "message": f"마스킹 완료 — 얼굴 {total_faces}개, 비얼굴 PII {total_pii}개",
        })

        # Compose video at native fps
        out_dir = settings.output_path / job_id
        out_path = out_dir / "output.mp4"
        compose_video(masked_dir, out_path, fps=native_fps)
        store.output_video_path = str(out_path)
        emit_log(job_id, {"step": "compose", "message": "영상 합성 완료"})

        # Report
        report = {
            "job_id": job_id,
            "scene_type": store.scene_type,
            "expected_pii": store.expected_pii,
            "total_people_detected": len(store.face_clusters),
            "protected_face_cluster_ids": store.protected_face_cluster_ids,
            "masked_pii_types": store.masked_pii_types,
            "total_faces_blurred": total_faces,
            "total_pii_masked": total_pii,
            "output_video_path": str(out_path),
        }
        (out_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        store.report = report

        store.status = "done"
        write_status(job_id, "done")
        emit_log(job_id, {"event": "done", "message": "처리 완료! 영상을 다운로드하세요."})

    except Exception as exc:
        store.status = "failed"
        store.error = str(exc)
        write_status(job_id, "failed", str(exc))
        emit_log(job_id, {"step": "error", "message": str(exc)})
