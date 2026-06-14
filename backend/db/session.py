"""SQLAlchemy engine + session factory."""
from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from ..config import settings


class Base(DeclarativeBase):
    pass


engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

SessionLocal = sessionmaker(engine, expire_on_commit=False)


def init_db() -> None:
    from . import models  # noqa: F401 — ensure models register on Base.metadata
    Base.metadata.create_all(engine)
