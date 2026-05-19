"""GPT-4o Vision scene analysis — returns (scene_type, expected_pii_types)."""
import base64
import json
from pathlib import Path

from openai import OpenAI

from ...config import settings

_PII_TYPES = {"face", "document", "screen", "nameplate", "id_card"}

_PROMPT = (
    "You are a privacy analysis agent. Given a video frame, identify the scene type "
    "and privacy-sensitive objects likely present.\n"
    "Respond ONLY as JSON:\n"
    '{"scene_type": "meeting|lecture|interview|public|other", '
    '"expected_pii": ["face","document","screen","nameplate","id_card"], '
    '"reasoning": "..."}\n'
    "Only include PII types that are actually visible or highly likely in this scene."
)


def analyze_scene(frame_path: str) -> tuple[str, list[str]]:
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
        return "other", []

    parsed = json.loads(raw)
    scene_type = parsed.get("scene_type", "other")
    expected_pii = [p for p in parsed.get("expected_pii", []) if p in _PII_TYPES]
    return scene_type, expected_pii
