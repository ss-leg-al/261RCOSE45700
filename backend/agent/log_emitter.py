import json
from pathlib import Path

from ..config import settings


def emit_log(job_id: str, payload: dict) -> None:
    log_path = settings.upload_path / job_id / "logs.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_status(job_id: str, status: str, error: str | None = None) -> None:
    path = settings.upload_path / job_id / "status.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(status if not error else f"{status}\n{error}", encoding="utf-8")
