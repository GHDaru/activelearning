"""Modelo de dados do FlowBuilder.

Um ``Run`` registra uma execução de fluxo (v0: avaliação de oráculo estilo E0)
com sua config completa e o relatório final — o banco guarda o ÍNDICE da
execução; os artefatos brutos (JSONL de anotações) ficam em disco, versionáveis,
apontados por ``artifacts_dir`` (Princípio I: reprodutibilidade).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(12), primary_key=True, default=_new_id)
    name: Mapped[str] = mapped_column(String(200))
    kind: Mapped[str] = mapped_column(String(40), default="oracle-eval")
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    report: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifacts_dir: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    def to_dict(self, with_report: bool = False) -> dict:
        data = {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "status": self.status,
            "config": self.config,
            "error": self.error,
            "artifacts_dir": self.artifacts_dir,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }
        if with_report:
            data["report"] = self.report
        return data
