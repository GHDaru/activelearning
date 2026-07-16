"""FastAPI do FlowBuilder.

Sobe com:
    uv run uvicorn --factory activelearning.adapters.api.app:create_app --reload --port 8000

Banco: SQLite local por padrão; Neon/Postgres via ``DATABASE_URL`` no ``.env``.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import select

from activelearning.adapters.api import runner
from activelearning.adapters.api.settings import Settings
from activelearning.adapters.persistence.db import Base, make_engine, make_session_factory
from activelearning.adapters.persistence.models import Run


class CreateRunBody(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    sample: str = Field(default="rand", pattern="^(rand|strat)$")
    limit: int = Field(default=50, ge=1, le=5000)
    oracle: dict = Field(
        default_factory=lambda: {"provider": "simulated", "noise": 0.1},
        description="Spec de oráculo no formato dos configs do E0 "
        "(provider simulated|openai|huawei-maas|openrouter|...)",
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    engine = make_engine(settings.database_url)
    Base.metadata.create_all(engine)
    session_factory = make_session_factory(engine)

    experiment_config = json.loads(
        Path(settings.experiment_config).read_text(encoding="utf-8")
    )

    app = FastAPI(title="FALCO FlowBuilder API", version="0.1.0")
    app.state.session_factory = session_factory
    app.state.settings = settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict:
        backend = "postgres" if (settings.database_url or "").startswith("postgres") else "sqlite"
        return {"status": "ok", "database": backend}

    @app.get("/api/oracles")
    def list_oracles() -> list[dict]:
        """Specs disponíveis: simulado (offline) + entradas do config do E0 (sem chaves)."""
        specs: list[dict] = [
            {
                "provider": "simulated",
                "label": "Oráculo simulado (gold + ruído ε, offline)",
                "noise": 0.1,
            }
        ]
        for spec in experiment_config.get("oracles", []):
            specs.append(
                {
                    "provider": spec["provider"],
                    "model": spec.get("model"),
                    "mode": spec.get("mode", "enum"),
                    "label": f"{spec['provider']}:{spec.get('model', '?')}",
                    "items_per_call": spec.get("items_per_call", 1),
                }
            )
        return specs

    @app.get("/api/samples")
    def list_samples() -> list[dict]:
        cfg = experiment_config.get("samples", {})
        return [
            {"name": "rand", "description": "Aleatória simples", "size": cfg.get("random_size")},
            {
                "name": "strat",
                "description": "Estratificada (k por classe)",
                "per_class": cfg.get("stratified_per_class"),
            },
        ]

    @app.post("/api/runs", status_code=201)
    def create_run(body: CreateRunBody) -> dict:
        with session_factory() as session:
            run = Run(name=body.name, config=body.model_dump())
            session.add(run)
            session.commit()
            run_id = run.id
            payload = run.to_dict()
        runner.launch_run(run_id, session_factory, experiment_config, settings.artifacts_root)
        return payload

    @app.get("/api/runs")
    def list_runs() -> list[dict]:
        with session_factory() as session:
            rows = session.scalars(select(Run).order_by(Run.created_at.desc())).all()
            return [r.to_dict() for r in rows]

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        with session_factory() as session:
            run = session.get(Run, run_id)
            if run is None:
                raise HTTPException(status_code=404, detail="Run não encontrado")
            return run.to_dict(with_report=True)

    return app
