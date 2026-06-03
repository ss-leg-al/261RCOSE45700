"""GPT-4o Vision scene analysis."""
import base64
import json
from pathlib import Path

from openai import OpenAI

from ...config import settings

_PII_TYPES = {"face", "document", "screen", "nameplate", "id_card"}

_PROMPT = (
    "You are a privacy analysis agent. Given a video frame, identify the scene type "
    "and privacy-sensitive objects likely present. Be specific enough for a video "
    "editor to understand what to inspect before masking.\n"
    "Respond ONLY as JSON:\n"
    '{"scene_type": "meeting|lecture|interview|public|other", '
    '"expected_pii": ["face","document","screen","nameplate","id_card"], '
    '"visible_context": "short scene description", '
    '"privacy_risks": ["risk sentence"], '
    '"recommended_focus": ["review target"], '
    '"confidence": 0.0}\n'
    "Only include PII types that are actually visible or highly likely in this scene. "
    "Use Korean for visible_context, privacy_risks, and recommended_focus."
)


def analyze_scene(frame_path: str) -> dict:
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    b64 = base64.b64encode(Path(frame_path).read_bytes()).decode()

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": _PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content
    if not raw:
        return _fallback_result()

    parsed = json.loads(raw)
    scene_type = parsed.get("scene_type", "other")
    expected_pii = [p for p in parsed.get("expected_pii", []) if p in _PII_TYPES]
    return {
        "scene_type": scene_type if scene_type in {"meeting", "lecture", "interview", "public", "other"} else "other",
        "expected_pii": expected_pii,
        "visible_context": str(parsed.get("visible_context") or "프레임 맥락 정보 없음"),
        "privacy_risks": _string_list(parsed.get("privacy_risks")),
        "recommended_focus": _string_list(parsed.get("recommended_focus")),
        "confidence": _confidence(parsed.get("confidence")),
    }


def _fallback_result() -> dict:
    return {
        "scene_type": "other",
        "expected_pii": [],
        "visible_context": "분석 결과를 가져오지 못했습니다.",
        "privacy_risks": [],
        "recommended_focus": [],
        "confidence": None,
    }


def _string_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item]


def _confidence(value) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return max(0.0, min(1.0, float(value)))
