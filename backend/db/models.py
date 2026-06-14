"""ORM models for jobs, face clusters, PII candidates, and selection profiles."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .session import Base


class Job(Base):
    __tablename__ = "jobs"

    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Filesystem paths
    video_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    frames_dir: Mapped[str | None] = mapped_column(Text, nullable=True)
    masked_frames_dir: Mapped[str | None] = mapped_column(Text, nullable=True)
    mask_preview_frames_dir: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_video_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    mask_preview_video_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Video metadata
    native_fps: Mapped[float | None] = mapped_column(Float, nullable=True)

    # GPT-4o results
    scene_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    expected_pii: Mapped[list] = mapped_column(JSONB, default=list)
    scene_analysis: Mapped[dict] = mapped_column(JSONB, default=dict)
    detection_pii_types: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    deterministic_pii_types_added: Mapped[list | None] = mapped_column(JSONB, nullable=True)

    # User selection
    protected_face_cluster_ids: Mapped[list] = mapped_column(JSONB, default=list)
    masked_pii_types: Mapped[list] = mapped_column(JSONB, default=list)
    masked_pii_object_ids: Mapped[list] = mapped_column(JSONB, default=list)
    sam3_mode: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # Guideline + report
    guideline: Mapped[list] = mapped_column(JSONB, default=list)
    report: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Counters
    total_faces_blurred: Mapped[int] = mapped_column(Integer, default=0)
    total_pii_masked: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now()
    )

    face_clusters: Mapped[list["FaceClusterRow"]] = relationship(
        back_populates="job", cascade="all, delete-orphan", lazy="selectin"
    )
    pii_candidates: Mapped[list["PIICandidateRow"]] = relationship(
        back_populates="job", cascade="all, delete-orphan", lazy="selectin"
    )


class FaceClusterRow(Base):
    __tablename__ = "face_clusters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(
        ForeignKey("jobs.job_id", ondelete="CASCADE"), index=True
    )
    cluster_id: Mapped[int] = mapped_column(Integer)
    thumbnail: Mapped[str] = mapped_column(Text)
    count: Mapped[int] = mapped_column(Integer, default=0)
    frame_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bbox_xyxy: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    embeddings: Mapped[list] = mapped_column(JSONB, default=list)

    job: Mapped[Job] = relationship(back_populates="face_clusters")


class PIICandidateRow(Base):
    __tablename__ = "pii_candidates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(
        ForeignKey("jobs.job_id", ondelete="CASCADE"), index=True
    )
    object_id: Mapped[int] = mapped_column(Integer)
    pii_type: Mapped[str] = mapped_column(String(64))
    thumbnail: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    frame_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bbox_xyxy: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    mask_strategy: Mapped[str | None] = mapped_column(String(32), nullable=True)

    job: Mapped[Job] = relationship(back_populates="pii_candidates")


class Profile(Base):
    __tablename__ = "profiles"

    profile_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    masked_pii_types: Mapped[list] = mapped_column(JSONB, default=list)
    protected_face_embeddings: Mapped[list] = mapped_column(JSONB, default=list)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
