from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class FaceCluster:
    cluster_id: int
    thumbnail: str   # filename under uploads/{job_id}/thumbnails/
    count: int       # number of detections in this cluster


@dataclass
class PIICandidate:
    object_id: int
    pii_type: str    # document | screen | nameplate | id_card
    thumbnail: str   # filename under uploads/{job_id}/thumbnails/
    confidence: float


@dataclass
class JobStore:
    job_id: str
    video_path: str | None = None
    frames_dir: str | None = None
    status: str = "pending"  # pending|detecting|awaiting_selection|masking|done|failed
    error: str | None = None

    # Video metadata
    native_fps: float | None = None

    # GPT-4o results
    scene_type: str | None = None
    expected_pii: list[str] = field(default_factory=list)

    # InsightFace clustering
    face_clusters: list[FaceCluster] = field(default_factory=list)
    cluster_embeddings: dict[int, list] = field(default_factory=dict)

    # SAM3 non-face PII candidates
    pii_candidates: list[PIICandidate] = field(default_factory=list)

    # User selection
    protected_face_cluster_ids: list[int] = field(default_factory=list)
    masked_pii_types: list[str] = field(default_factory=list)

    # Guideline (generated after Phase 1)
    guideline: list[dict] = field(default_factory=list)

    # Output
    masked_frames_dir: str | None = None
    output_video_path: str | None = None
    report: dict | None = None
    total_faces_blurred: int = 0
    total_pii_masked: int = 0


_stores: dict[str, JobStore] = {}
_lock = threading.Lock()


def get_store(job_id: str) -> JobStore:
    with _lock:
        if job_id not in _stores:
            _stores[job_id] = JobStore(job_id=job_id)
        return _stores[job_id]


def list_stores() -> list[JobStore]:
    with _lock:
        return list(_stores.values())


def reset_store(job_id: str) -> None:
    with _lock:
        _stores.pop(job_id, None)
