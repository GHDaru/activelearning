"""FastAPI do FlowBuilder.

Sobe com:
    uv run uvicorn --factory activelearning.adapters.api.app:create_app --reload --port 8000

Banco: SQLite local por padrão; Neon/Postgres via ``DATABASE_URL`` no ``.env``.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from activelearning.adapters.api import runner
from activelearning.adapters.api.settings import Settings
from activelearning.adapters.persistence.db import Base, make_engine, make_session_factory
from activelearning.adapters.persistence.models import Dataset, Run
from activelearning.application.sanitize_dataset import sanitize_csv


class CreateRunBody(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    kind: str = Field(default="oracle-eval", pattern="^(oracle-eval|active-learning)$")
    sample: str = Field(default="rand", pattern="^(rand|strat)$")
    limit: int = Field(default=50, ge=1, le=5000)
    dataset_id: str | None = Field(default=None, description="obrigatório p/ active-learning")
    params: dict = Field(
        default_factory=dict,
        description="active-learning: seed, budget, batch_size, initial_size, "
        "strategy (entropy|least_confidence|smallest_margin|random|hybrid), "
        "pool_size, test_fraction, min_per_class",
    )
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
        if body.kind == "active-learning" and not body.dataset_id:
            raise HTTPException(status_code=422, detail="active-learning exige dataset_id")
        with session_factory() as session:
            if body.dataset_id and session.get(Dataset, body.dataset_id) is None:
                raise HTTPException(status_code=404, detail="dataset não encontrado")
            run = Run(name=body.name, kind=body.kind, config=body.model_dump())
            session.add(run)
            session.commit()
            run_id = run.id
            payload = run.to_dict()
        runner.launch_run(run_id, session_factory, experiment_config, settings.artifacts_root)
        return payload

    # ------------------------------- datasets -------------------------------
    uploads_root = settings.artifacts_root.parent / "uploads"

    @app.post("/api/datasets", status_code=201)
    async def upload_dataset(
        file: UploadFile = File(...),
        name: str = Form(...),
        text_column: str = Form(...),
        label_column: str = Form(...),
        operational_labels: str = Form("inativo"),
    ) -> dict:
        import uuid

        ds_id = uuid.uuid4().hex[:12]
        ds_dir = uploads_root / ds_id
        ds_dir.mkdir(parents=True, exist_ok=True)
        original = ds_dir / "original.csv"
        original.write_bytes(await file.read())
        sanitized = ds_dir / "sanitized.csv"
        ops = tuple(x.strip() for x in operational_labels.split(",") if x.strip())
        try:
            report = sanitize_csv(
                original, sanitized, text_column=text_column,
                label_column=label_column, operational_labels=ops,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        with session_factory() as session:
            ds = Dataset(
                id=ds_id, name=name, filename=file.filename or "dataset.csv",
                text_column=text_column, label_column=label_column,
                original_path=str(original), sanitized_path=str(sanitized),
                report=report.to_dict(),
            )
            session.add(ds)
            session.commit()
            return ds.to_dict()

    @app.get("/api/datasets")
    def list_datasets() -> list[dict]:
        with session_factory() as session:
            rows = session.scalars(select(Dataset).order_by(Dataset.created_at.desc())).all()
            return [d.to_dict(with_report=False) for d in rows]

    @app.get("/api/datasets/{dataset_id}")
    def get_dataset(dataset_id: str) -> dict:
        with session_factory() as session:
            ds = session.get(Dataset, dataset_id)
            if ds is None:
                raise HTTPException(status_code=404, detail="dataset não encontrado")
            return ds.to_dict()

    @app.get("/api/datasets/{dataset_id}/stats")
    def dataset_stats(dataset_id: str) -> dict:
        """Estatísticas descritivas da base saneada (cacheadas em stats.json)."""
        import csv as _csv
        import json as _json
        import statistics as _st
        from collections import Counter as _Counter

        with session_factory() as session:
            ds = session.get(Dataset, dataset_id)
            if ds is None:
                raise HTTPException(status_code=404, detail="dataset não encontrado")
            sanitized = Path(ds.sanitized_path)
            text_col, label_col = ds.text_column, ds.label_column
        cache = sanitized.parent / "stats.json"
        if cache.exists():
            return _json.loads(cache.read_text())

        counts: _Counter[str] = _Counter()
        vocab: set[str] = set()
        char_lens: list[int] = []
        tok_lens: list[int] = []
        with sanitized.open(encoding="utf-8") as fh:
            for row in _csv.DictReader(fh):
                text = (row.get(text_col) or "").strip()
                counts[(row.get(label_col) or "").strip()] += 1
                toks = text.lower().split()
                vocab.update(toks)
                char_lens.append(len(text))
                tok_lens.append(len(toks))
        per_class = sorted(counts.values())
        stats = {
            "n_rows": sum(per_class),
            "n_classes": len(counts),
            "vocab_size": len(vocab),
            "per_class": {
                "min": per_class[0] if per_class else 0,
                "median": int(_st.median(per_class)) if per_class else 0,
                "mean": round(_st.mean(per_class), 1) if per_class else 0,
                "max": per_class[-1] if per_class else 0,
                "lt5": sum(1 for c in per_class if c < 5),
                "imbalance_ratio": (round(per_class[-1] / per_class[0], 1)
                                    if per_class and per_class[0] else None),
            },
            "text": {
                "chars_mean": round(_st.mean(char_lens), 1) if char_lens else 0,
                "chars_p50": int(_st.median(char_lens)) if char_lens else 0,
                "chars_max": max(char_lens) if char_lens else 0,
                "tokens_mean": round(_st.mean(tok_lens), 1) if tok_lens else 0,
            },
            "top_classes": [{"label": l, "n": n} for l, n in counts.most_common(10)],
        }
        cache.write_text(_json.dumps(stats, ensure_ascii=False))
        return stats

    @app.get("/api/datasets/{dataset_id}/download")
    def download_dataset(dataset_id: str, which: str = "sanitized"):
        if which not in ("sanitized", "original"):
            raise HTTPException(status_code=422, detail="which deve ser sanitized|original")
        with session_factory() as session:
            ds = session.get(Dataset, dataset_id)
            if ds is None:
                raise HTTPException(status_code=404, detail="dataset não encontrado")
            path = ds.sanitized_path if which == "sanitized" else ds.original_path
            fname = f"{ds.name}-{which}.csv".replace(" ", "_")
        return FileResponse(path, media_type="text/csv", filename=fname)

    # ---- catálogo de experimentos da tese (execução + replay) ----
    from activelearning.adapters.api import experiments_catalog as expcat

    @app.get("/api/experiments")
    def list_experiments() -> list[dict]:
        return expcat.catalog_status()

    @app.get("/api/experiments/{exp_id}/results")
    def experiment_results(exp_id: str) -> dict:
        try:
            return expcat.load_results(exp_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="experimento desconhecido")

    @app.post("/api/experiments/{exp_id}/execute", status_code=201)
    def experiment_execute(exp_id: str, body: dict) -> dict:
        try:
            return expcat.execute(exp_id, body.get("preset", ""))
        except KeyError:
            raise HTTPException(status_code=404, detail="experimento desconhecido")
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.post("/api/experiments/{exp_id}/stop")
    def experiment_stop(exp_id: str) -> dict:
        try:
            return expcat.stop(exp_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.get("/api/experiments/{exp_id}/log")
    def experiment_log(exp_id: str) -> dict:
        return {"log": expcat.tail_log(exp_id)}

    # ---- base de conhecimento (knowledge graph dos fichamentos) ----
    def _kg_dir() -> Path:
        return settings.thesis_root / "fichamentos"

    @app.get("/api/kg/summary")
    def kg_summary() -> dict:
        """Contagens do grafo por tipo de nó + aresta (para o cabeçalho da view)."""
        from collections import Counter

        kg_json = _kg_dir() / "kg.json"
        if not kg_json.exists():
            raise HTTPException(
                status_code=404,
                detail=f"kg.json não encontrado em {kg_json}. Rode build_kg.py na tese "
                "ou defina FALCO_THESIS_ROOT.",
            )
        graph = json.loads(kg_json.read_text(encoding="utf-8"))
        by_node = Counter(n.get("type", "?") for n in graph.get("nodes", []))
        by_edge = Counter(e.get("type", "?") for e in graph.get("edges", []))
        return {
            "n_nodes": len(graph.get("nodes", [])),
            "n_edges": len(graph.get("edges", [])),
            "n_artigos": by_node.get("artigo", 0),
            "n_pendentes": by_node.get("artigo-pendente", 0),
            "by_node_type": dict(by_node),
            "by_edge_type": dict(by_edge),
            "generated_at": kg_json.stat().st_mtime,
        }

    @app.get("/api/kg")
    def kg_graph() -> dict:
        """Grafo completo (nós + arestas tipadas) — consumível por qualquer cliente."""
        kg_json = _kg_dir() / "kg.json"
        if not kg_json.exists():
            raise HTTPException(status_code=404, detail="kg.json não encontrado")
        return json.loads(kg_json.read_text(encoding="utf-8"))

    @app.get("/api/kg/view")
    def kg_view():
        """Visualização autocontida (canvas force-graph) para embutir em iframe."""
        kg_html = _kg_dir() / "kg.html"
        if not kg_html.exists():
            raise HTTPException(status_code=404, detail="kg.html não encontrado")
        return FileResponse(kg_html, media_type="text/html")

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
