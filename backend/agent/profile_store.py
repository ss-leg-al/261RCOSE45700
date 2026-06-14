"""DB-backed selection profile store."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from sqlalchemy import select

from ..db.models import Profile
from ..db.session import SessionLocal


@dataclass
class SelectionProfile:
    profile_id: str
    name: str
    masked_pii_types: list[str] = field(default_factory=list)
    protected_face_embeddings: list[list[float]] = field(default_factory=list)


def _to_dataclass(row: Profile) -> SelectionProfile:
    return SelectionProfile(
        profile_id=row.profile_id,
        name=row.name,
        masked_pii_types=list(row.masked_pii_types or []),
        protected_face_embeddings=list(row.protected_face_embeddings or []),
    )


def list_profiles() -> list[SelectionProfile]:
    with SessionLocal() as session:
        rows = session.execute(
            select(Profile).order_by(Profile.created_at.desc())
        ).scalars().all()
        return [_to_dataclass(r) for r in rows]


def get_profile(profile_id: str) -> SelectionProfile | None:
    with SessionLocal() as session:
        row = session.get(Profile, profile_id)
        return _to_dataclass(row) if row else None


def save_profile(
    name: str,
    masked_pii_types: list[str],
    protected_face_embeddings: list[list[float]],
) -> SelectionProfile:
    profile_id = str(uuid.uuid4())
    with SessionLocal() as session:
        row = Profile(
            profile_id=profile_id,
            name=name,
            masked_pii_types=masked_pii_types,
            protected_face_embeddings=protected_face_embeddings,
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return _to_dataclass(row)


def delete_profile(profile_id: str) -> bool:
    with SessionLocal() as session:
        row = session.get(Profile, profile_id)
        if row is None:
            return False
        session.delete(row)
        session.commit()
        return True
