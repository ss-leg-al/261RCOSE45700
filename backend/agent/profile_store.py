"""In-memory selection profile store."""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field


@dataclass
class SelectionProfile:
    profile_id: str
    name: str
    masked_pii_types: list[str] = field(default_factory=list)
    # Mean embedding per saved protected face (one entry per person role)
    protected_face_embeddings: list[list[float]] = field(default_factory=list)


_profiles: dict[str, SelectionProfile] = {}
_lock = threading.Lock()


def list_profiles() -> list[SelectionProfile]:
    with _lock:
        return list(_profiles.values())


def get_profile(profile_id: str) -> SelectionProfile | None:
    with _lock:
        return _profiles.get(profile_id)


def save_profile(
    name: str,
    masked_pii_types: list[str],
    protected_face_embeddings: list[list[float]],
) -> SelectionProfile:
    profile = SelectionProfile(
        profile_id=str(uuid.uuid4()),
        name=name,
        masked_pii_types=masked_pii_types,
        protected_face_embeddings=protected_face_embeddings,
    )
    with _lock:
        _profiles[profile.profile_id] = profile
    return profile


def delete_profile(profile_id: str) -> bool:
    with _lock:
        return _profiles.pop(profile_id, None) is not None
