"""Engine e sessões SQLAlchemy.

``DATABASE_URL`` decide o destino:
- ausente → SQLite local ``flowbuilder.db`` (zero-config, dev);
- ``postgresql+psycopg://...`` → Neon ou qualquer Postgres (produção).

Credenciais SEMPRE via ``.env`` (não versionado); nunca em código ou config
commitada.
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DEFAULT_SQLITE_URL = "sqlite:///flowbuilder.db"


class Base(DeclarativeBase):
    pass


def make_engine(database_url: str | None = None):
    url = database_url or os.environ.get("DATABASE_URL") or DEFAULT_SQLITE_URL
    # Neon fornece URLs "postgresql://"; o driver instalado é o psycopg 3.
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args, pool_pre_ping=True)


def make_session_factory(engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
