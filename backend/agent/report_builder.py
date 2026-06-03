from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_PII_LABELS = {
    "face": "얼굴",
    "document": "문서",
    "screen": "화면/스크린",
    "nameplate": "명패",
    "id_card": "신분증",
}

_SCENE_LABELS = {
    "meeting": "회의",
    "lecture": "강의",
    "interview": "인터뷰",
    "public": "공공장소",
    "other": "기타",
}


def build_intermediate_scene_analysis(
    frame_results: list[dict[str, Any]],
    scene_type: str,
    expected_pii: list[str],
) -> dict[str, Any]:
    risk_counter: Counter[str] = Counter()
    focus_counter: Counter[str] = Counter()
    confidence_values: list[float] = []

    for result in frame_results:
        for risk in result.get("privacy_risks", []):
            risk_counter[str(risk)] += 1
        for focus in result.get("recommended_focus", []):
            focus_counter[str(focus)] += 1
        confidence = result.get("confidence")
        if isinstance(confidence, (int, float)):
            confidence_values.append(float(confidence))

    avg_confidence = (
        round(sum(confidence_values) / len(confidence_values), 2)
        if confidence_values else None
    )

    return {
        "scene_type": scene_type,
        "scene_label": _SCENE_LABELS.get(scene_type, scene_type),
        "expected_pii": expected_pii,
        "expected_pii_labels": [_PII_LABELS.get(t, t) for t in expected_pii],
        "confidence": avg_confidence,
        "frame_count": len(frame_results),
        "top_privacy_risks": [
            {"message": risk, "votes": votes}
            for risk, votes in risk_counter.most_common(5)
        ],
        "recommended_focus": [
            {"message": focus, "votes": votes}
            for focus, votes in focus_counter.most_common(5)
        ],
        "frames": frame_results,
    }


def build_final_report(
    *,
    store,
    job_id: str,
    total_faces_blurred: int,
    total_pii_masked: int,
    output_video_path: str | None,
    skipped: bool = False,
) -> dict[str, Any]:
    protected_ids = list(store.protected_face_cluster_ids)
    masked_pii_types = list(store.masked_pii_types)
    detected_pii_types = sorted({p.pii_type for p in store.pii_candidates})
    unmasked_pii_types = [
        pii_type for pii_type in detected_pii_types if pii_type not in masked_pii_types
    ]

    face_count = len(store.face_clusters)
    protected_count = len(protected_ids)
    blur_target_count = max(face_count - protected_count, 0)
    risk_score = _estimate_risk_score(
        face_count=face_count,
        pii_count=len(store.pii_candidates),
        unmasked_pii_count=len(unmasked_pii_types),
        skipped=skipped,
    )

    report = {
        "metadata": {
            "job_id": job_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "skipped" if skipped else "completed",
            "source_video_path": store.video_path,
            "output_video_path": output_video_path,
            "native_fps": store.native_fps,
        },
        "executive_summary": _build_summary(
            scene_type=store.scene_type,
            face_count=face_count,
            protected_count=protected_count,
            pii_types=detected_pii_types,
            skipped=skipped,
        ),
        "risk_assessment": {
            "score": risk_score,
            "level": _risk_level(risk_score),
            "reasons": _risk_reasons(
                face_count=face_count,
                detected_pii_types=detected_pii_types,
                unmasked_pii_types=unmasked_pii_types,
                skipped=skipped,
            ),
        },
        "scene_analysis": store.scene_analysis or {
            "scene_type": store.scene_type,
            "expected_pii": store.expected_pii,
        },
        "detection_summary": {
            "total_people_detected": face_count,
            "face_clusters": [
                {
                    "cluster_id": c.cluster_id,
                    "display_name": f"인물 {c.cluster_id + 1}",
                    "appearances": c.count,
                    "protected": c.cluster_id in protected_ids,
                    "planned_action": (
                        "보호: 원본 유지"
                        if c.cluster_id in protected_ids
                        else "마스킹: 얼굴 블러"
                    ),
                }
                for c in store.face_clusters
            ],
            "total_pii_candidates": len(store.pii_candidates),
            "pii_candidates": [
                {
                    "object_id": p.object_id,
                    "pii_type": p.pii_type,
                    "label": _PII_LABELS.get(p.pii_type, p.pii_type),
                    "confidence": round(float(p.confidence), 3),
                    "selected_for_masking": p.pii_type in masked_pii_types,
                    "planned_action": (
                        "마스킹 적용"
                        if p.pii_type in masked_pii_types
                        else "사용자 선택으로 유지"
                    ),
                }
                for p in store.pii_candidates
            ],
        },
        "user_decisions": {
            "protected_face_cluster_ids": protected_ids,
            "masked_pii_types": masked_pii_types,
            "unmasked_pii_types": unmasked_pii_types,
            "skipped": skipped,
        },
        "processing_result": {
            "face_clusters_to_blur": blur_target_count,
            "total_faces_blurred": 0 if skipped else total_faces_blurred,
            "total_pii_masked": 0 if skipped else total_pii_masked,
            "output_video_path": output_video_path,
        },
        "review_checklist": _build_review_checklist(
            store=store,
            unmasked_pii_types=unmasked_pii_types,
            skipped=skipped,
        ),
        "guideline_items": list(store.guideline),
        "legacy": {
            "job_id": job_id,
            "scene_type": store.scene_type,
            "expected_pii": store.expected_pii,
            "total_people_detected": face_count,
            "protected_face_cluster_ids": protected_ids,
            "masked_pii_types": masked_pii_types,
            "total_faces_blurred": 0 if skipped else total_faces_blurred,
            "total_pii_masked": 0 if skipped else total_pii_masked,
            "skipped": skipped,
        },
    }

    # Keep old top-level keys for the current frontend and any external callers.
    report.update(report["legacy"])
    return report


def write_report(report: dict[str, Any], output_video_path: str | None) -> None:
    if not output_video_path:
        return
    import json

    out_dir = Path(output_video_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _estimate_risk_score(
    *,
    face_count: int,
    pii_count: int,
    unmasked_pii_count: int,
    skipped: bool,
) -> int:
    score = min(100, face_count * 12 + pii_count * 18 + unmasked_pii_count * 20)
    if skipped and (face_count or pii_count):
        score = min(100, score + 25)
    return score


def _risk_level(score: int) -> str:
    if score >= 70:
        return "높음"
    if score >= 35:
        return "중간"
    return "낮음"


def _build_summary(
    *,
    scene_type: str | None,
    face_count: int,
    protected_count: int,
    pii_types: list[str],
    skipped: bool,
) -> str:
    scene_label = _SCENE_LABELS.get(scene_type or "other", scene_type or "기타")
    pii_text = ", ".join(_PII_LABELS.get(t, t) for t in pii_types) or "비얼굴 PII 없음"
    if skipped:
        return (
            f"{scene_label} 영상에서 인물 {face_count}명과 {pii_text} 항목이 확인됐지만 "
            "사용자 선택으로 편집 없이 원본을 완료 처리했습니다."
        )
    return (
        f"{scene_label} 영상에서 인물 {face_count}명을 감지했고 "
        f"{protected_count}명은 보호 대상으로 유지했습니다. 비얼굴 PII 후보는 {pii_text}입니다."
    )


def _risk_reasons(
    *,
    face_count: int,
    detected_pii_types: list[str],
    unmasked_pii_types: list[str],
    skipped: bool,
) -> list[str]:
    reasons: list[str] = []
    if face_count:
        reasons.append(f"영상 내 식별 가능한 인물 {face_count}명이 감지되었습니다.")
    if detected_pii_types:
        labels = ", ".join(_PII_LABELS.get(t, t) for t in detected_pii_types)
        reasons.append(f"비얼굴 개인정보 후보로 {labels} 항목이 감지되었습니다.")
    if unmasked_pii_types:
        labels = ", ".join(_PII_LABELS.get(t, t) for t in unmasked_pii_types)
        reasons.append(f"{labels} 항목은 사용자 선택에 따라 마스킹 대상에서 제외되었습니다.")
    if skipped:
        reasons.append("편집 스킵으로 원본 영상이 그대로 결과물에 사용되었습니다.")
    return reasons or ["자동 탐지 기준에서 큰 개인정보 노출 위험은 확인되지 않았습니다."]


def _build_review_checklist(*, store, unmasked_pii_types: list[str], skipped: bool) -> list[dict[str, str]]:
    checklist = [
        {
            "title": "보호 인물 확인",
            "status": "확인 필요" if store.face_clusters else "해당 없음",
            "detail": "보호로 선택한 인물이 본인 또는 공개 동의 대상인지 검토하세요.",
        },
        {
            "title": "짧게 지나가는 PII 확인",
            "status": "확인 필요" if store.pii_candidates else "해당 없음",
            "detail": "썸네일에 잡히지 않은 문서, 화면, 명패가 순간적으로 등장하는지 재생 확인이 필요합니다.",
        },
    ]
    if unmasked_pii_types:
        labels = ", ".join(_PII_LABELS.get(t, t) for t in unmasked_pii_types)
        checklist.append({
            "title": "마스킹 제외 항목 재확인",
            "status": "주의",
            "detail": f"{labels} 항목은 결과 영상에 남을 수 있습니다.",
        })
    if skipped:
        checklist.append({
            "title": "스킵 결정 근거 보관",
            "status": "주의",
            "detail": "편집 없이 공개 가능한 영상인지 최종 책임자가 한 번 더 확인해야 합니다.",
        })
    return checklist
