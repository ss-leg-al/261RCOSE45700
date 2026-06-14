"""DB-backed job store with in-memory working cache.

The pipeline mutates a dataclass-style JobStore in place for performance, and
calls save_store(job_id) at phase boundaries to persist to PostgreSQL.
"""
from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field

from sqlalchemy import select

from ..db.models import FaceClusterRow, Job, PIICandidateRow
from ..db.session import SessionLocal


@dataclass
class FaceCluster:
    cluster_id: int
    thumbnail: str
    count: int
    frame_index: int | None = None
    bbox_xyxy: list[float] | None = None


@dataclass
class PIICandidate:
    object_id: int
    pii_type: str
    thumbnail: str
    confidence: float
    frame_index: int | None = None
    bbox_xyxy: list[float] | None = None
    mask_strategy: str | None = None


@dataclass
class JobStore:
    job_id: str
    video_path: str | None = None
    frames_dir: str | None = None
    status: str = "pending"
    error: str | None = None

    native_fps: float | None = None

    scene_type: str | None = None
    expected_pii: list[str] = field(default_factory=list)
    scene_analysis: dict = field(default_factory=dict)
    detection_pii_types: list[str] | None = None
    deterministic_pii_types_added: list[str] | None = None

    face_clusters: list[FaceCluster] = field(default_factory=list)
    cluster_embeddings: dict[int, list] = field(default_factory=dict)

    pii_candidates: list[PIICandidate] = field(default_factory=list)

    protected_face_cluster_ids: list[int] = field(default_factory=list)
    masked_pii_types: list[str] = field(default_factory=list)
    masked_pii_object_ids: list[int] = field(default_factory=list)
    sam3_mode: str | None = None

    guideline: list[dict] = field(default_factory=list)

    masked_frames_dir: str | None = None
    mask_preview_frames_dir: str | None = None
    output_video_path: str | None = None
    mask_preview_video_path: str | None = None
    report: dict | None = None
    total_faces_blurred: int = 0
    total_pii_masked: int = 0


_cache: dict[str, JobStore] = {}
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Dataclass <-> ORM mapping
# ---------------------------------------------------------------------------

_SCALAR_FIELDS = (
    "status", "error",
    "video_path", "frames_dir",
    "masked_frames_dir", "mask_preview_frames_dir",
    "output_video_path", "mask_preview_video_path",
    "native_fps",
    "scene_type", "expected_pii", "scene_analysis",
    "detection_pii_types", "deterministic_pii_types_added",
    "protected_face_cluster_ids", "masked_pii_types", "masked_pii_object_ids",
    "sam3_mode", "guideline", "report",
    "total_faces_blurred", "total_pii_masked",
)


def _row_to_store(row: Job) -> JobStore:
    store = JobStore(job_id=row.job_id)
    for f in _SCALAR_FIELDS:
        setattr(store, f, getattr(row, f))
    store.face_clusters = [
        FaceCluster(
            cluster_id=c.cluster_id,
            thumbnail=c.thumbnail,
            count=c.count,
            frame_index=c.frame_index,
            bbox_xyxy=c.bbox_xyxy,
        )
        for c in sorted(row.face_clusters, key=lambda c: c.cluster_id)
    ]
    store.cluster_embeddings = {
        c.cluster_id: list(c.embeddings or []) for c in row.face_clusters
    }
    store.pii_candidates = [
        PIICandidate(
            object_id=p.object_id,
            pii_type=p.pii_type,
            thumbnail=p.thumbnail,
            confidence=p.confidence,
            frame_index=p.frame_index,
            bbox_xyxy=p.bbox_xyxy,
            mask_strategy=p.mask_strategy,
        )
        for p in sorted(row.pii_candidates, key=lambda p: p.object_id)
    ]
    return store


def _apply_store_to_row(store: JobStore, row: Job) -> None:
    for f in _SCALAR_FIELDS:
        setattr(row, f, getattr(store, f))

    # Rewrite face_clusters + pii_candidates collections wholesale.
    # Cluster/object counts are small (typically <50) so this is cheap.
    row.face_clusters.clear()
    for fc in store.face_clusters:
        row.face_clusters.append(FaceClusterRow(
            cluster_id=fc.cluster_id,
            thumbnail=fc.thumbnail,
            count=fc.count,
            frame_index=fc.frame_index,
            bbox_xyxy=fc.bbox_xyxy,
            embeddings=store.cluster_embeddings.get(fc.cluster_id, []),
        ))
    row.pii_candidates.clear()
    for pc in store.pii_candidates:
        row.pii_candidates.append(PIICandidateRow(
            object_id=pc.object_id,
            pii_type=pc.pii_type,
            thumbnail=pc.thumbnail,
            confidence=pc.confidence,
            frame_index=pc.frame_index,
            bbox_xyxy=pc.bbox_xyxy,
            mask_strategy=pc.mask_strategy,
        ))


# ---------------------------------------------------------------------------
# Public API (matches previous in-memory interface)
# ---------------------------------------------------------------------------

def get_store(job_id: str) -> JobStore:
    """Return cached JobStore, loading from DB on cache miss; create if absent."""
    with _lock:
        cached = _cache.get(job_id)
        if cached is not None:
            return cached

    with SessionLocal() as session:
        row = session.get(Job, job_id)
        if row is None:
            row = Job(job_id=job_id)
            session.add(row)
            session.commit()
            session.refresh(row)
        store = _row_to_store(row)

    with _lock:
        # Re-check after acquiring the lock in case of races
        cached = _cache.get(job_id)
        if cached is not None:
            return cached
        _cache[job_id] = store
        return store


def save_store(job_id: str) -> None:
    """Persist the in-memory store to the DB."""
    with _lock:
        store = _cache.get(job_id)
    if store is None:
        return

    with SessionLocal() as session:
        row = session.get(Job, job_id)
        if row is None:
            row = Job(job_id=job_id)
            session.add(row)
        _apply_store_to_row(store, row)
        session.commit()


def list_stores() -> list[JobStore]:
    """Return all jobs from the DB (preferring cached versions when present)."""
    with SessionLocal() as session:
        rows = session.execute(select(Job).order_by(Job.created_at.desc())).scalars().all()
        stores = []
        with _lock:
            for row in rows:
                cached = _cache.get(row.job_id)
                stores.append(cached if cached is not None else _row_to_store(row))
        return stores


def reset_store(job_id: str) -> None:
    """Drop the cached store (does not delete the DB row)."""
    with _lock:
        _cache.pop(job_id, None)
