"""GPT-4o guideline generation — analyzes detection results and returns actionable warnings."""
from __future__ import annotations

import json

from ...config import settings

_PROMPT = """\
당신은 영상 프라이버시 컴플라이언스 어시스턴트입니다.
자동 PII 탐지 결과를 검토하고, 편집자가 최종 마스킹 전에 확인해야 할 사항을 안내합니다.

탐지 결과:
{detection_summary}

다음 기준으로 가이드라인 항목을 생성하세요:
- 등장 횟수가 적은 인물 (방문자/불특정인 — 보호 여부 확인 필요)
- 신뢰도가 낮은 PII 탐지 (수동 확인 권장)
- 씬 유형에 따른 특이 리스크 (회의실의 화면, 의료 씬의 문서 등)
- 차량/도로/주차장 씬의 번호판 노출 리스크
- 탐지 결과의 이상 패턴

JSON으로만 응답하세요:
{{"items": [
  {{"level": "warning" 또는 "info", "category": "face" 또는 "pii" 또는 "scene", "message": "..."}}
]}}

규칙: 메시지는 한국어, 40자 이내. 최대 6개. 실질적으로 유용한 항목만 포함.\
"""


def generate_guideline(
    scene_type: str,
    expected_pii: list[str],
    face_clusters: list,
    pii_candidates: list,
) -> list[dict]:
    if not settings.OPENAI_API_KEY:
        return []

    from openai import OpenAI

    face_summary = [
        {"cluster_id": fc.cluster_id, "appearances": fc.count}
        for fc in face_clusters
    ]
    pii_summary = [
        {
            "object_id": p.object_id,
            "type": p.pii_type,
            "confidence": round(p.confidence, 2),
            "frame_index": p.frame_index,
        }
        for p in pii_candidates
    ]
    detection_summary = json.dumps(
        {
            "scene_type": scene_type,
            "expected_pii": expected_pii,
            "total_people": len(face_clusters),
            "face_clusters": face_summary,
            "pii_candidates": pii_summary,
        },
        ensure_ascii=False,
    )

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": _PROMPT.format(detection_summary=detection_summary),
        }],
        response_format={"type": "json_object"},
    )

    raw = resp.choices[0].message.content
    if not raw:
        return []

    parsed = json.loads(raw)
    return [
        item for item in parsed.get("items", [])
        if isinstance(item, dict) and "level" in item and "message" in item
    ]
