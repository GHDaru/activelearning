"""Execução de runs em background.

v0: um run = avaliação de oráculo (mesmo caso de uso do E0) sobre uma amostra.
Os artefatos brutos (JSONL retomável + report) vão para
``{artifacts_root}/{run_id}/`` — o banco guarda só o índice e o report final
(Princípio I: nenhum número sem artefato).
"""
from __future__ import annotations

import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import sessionmaker

from activelearning.adapters.datasets.retail_csv import build_samples, load_rows_and_schema
from activelearning.adapters.oracles.factory import build_oracle
from activelearning.adapters.persistence.models import Dataset, Run
from activelearning.application.evaluate_oracle import EvaluateOracle

_dataset_cache: dict = {}
_dataset_lock = threading.Lock()


def get_samples_and_schema(experiment_config: dict):
    """Carrega dataset+amostras uma única vez por processo (250 mil linhas)."""
    key = str(experiment_config.get("dataset", {}).get("csv_path"))
    with _dataset_lock:
        if key not in _dataset_cache:
            rows, schema = load_rows_and_schema(experiment_config)
            samples = build_samples(experiment_config, rows, schema)
            _dataset_cache[key] = (samples, schema)
        return _dataset_cache[key]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def execute_run(
    run_id: str,
    session_factory: sessionmaker,
    experiment_config: dict,
    artifacts_root: Path,
) -> None:
    session = session_factory()
    try:
        run = session.get(Run, run_id)
        if run is None:
            return
        run.status, run.started_at = "running", _now()
        session.commit()

        cfg = run.config
        if run.kind == "active-learning":
            report = _execute_active_learning(session, run, cfg, artifacts_root)
            run.status, run.finished_at = "completed", _now()
            run.report = report
            run.artifacts_dir = str(artifacts_root / run_id)
            session.commit()
            return

        samples, schema = get_samples_and_schema(experiment_config)
        sample_name = cfg.get("sample", "rand")
        instances = samples[sample_name]
        limit = int(cfg.get("limit") or 0)
        if limit > 0:
            instances = instances[:limit]

        oracle = build_oracle(cfg["oracle"])
        out_dir = artifacts_root / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        report = EvaluateOracle(oracle=oracle, output_dir=out_dir).run(instances, schema)

        run.status, run.finished_at = "completed", _now()
        run.report = report.to_summary()
        run.artifacts_dir = str(out_dir)
        session.commit()
    except Exception as exc:  # noqa: BLE001 — o erro completo vai para o registro do run
        session.rollback()
        run = session.get(Run, run_id)
        if run is not None:
            run.status, run.finished_at = "failed", _now()
            run.error = f"{exc}\n{traceback.format_exc(limit=5)}"
            session.commit()
    finally:
        session.close()


def launch_run(run_id: str, session_factory, experiment_config: dict, artifacts_root: Path):
    thread = threading.Thread(
        target=execute_run,
        args=(run_id, session_factory, experiment_config, artifacts_root),
        daemon=True,
        name=f"run-{run_id}",
    )
    thread.start()
    return thread


def _load_dataset_rows(session, dataset_id: str) -> tuple[list[tuple[str, str]], str, str]:
    ds = session.get(Dataset, dataset_id)
    if ds is None:
        raise ValueError(f"dataset {dataset_id!r} não encontrado")
    import csv as _csv

    rows: list[tuple[str, str]] = []
    with open(ds.sanitized_path, encoding="utf-8") as fh:
        for row in _csv.DictReader(fh):
            rows.append((row[ds.text_column], row[ds.label_column]))
    return rows, ds.text_column, ds.label_column


def _execute_active_learning(session, run, cfg: dict, artifacts_root: Path) -> dict:
    """Executa um fluxo de AL parametrizado pela UI sobre um dataset saneado."""
    import random

    from activelearning.adapters.classifiers.pvbin import PVBinClassifier
    from activelearning.application.run_active_learning import run_active_learning
    from activelearning.domain.instances import CategorySchema, Instance, Label

    params = cfg.get("params", {})
    seed = int(params.get("seed", 42))
    pool_cap = int(params.get("pool_size", 2000))
    test_fraction = float(params.get("test_fraction", 0.2))
    min_per_class = int(params.get("min_per_class", 2))

    rows, _, _ = _load_dataset_rows(session, cfg["dataset_id"])

    # dedupe por texto normalizado ANTES do split (regra da tese)
    from activelearning.domain.instances import normalize_label

    seen: set[str] = set()
    unique = []
    for text, label in rows:
        k = normalize_label(text)
        if k not in seen:
            seen.add(k)
            unique.append((text, label))

    from collections import Counter

    counts = Counter(l for _, l in unique)
    frequent = {l for l, n in counts.items() if n >= min_per_class}
    unique = [(t, l) for t, l in unique if l in frequent]
    if len(unique) < 50:
        raise ValueError("dataset pequeno demais após saneamento/dedupe (mínimo 50 linhas)")

    rng = random.Random(seed)
    rng.shuffle(unique)
    if len(unique) > pool_cap:
        unique = unique[:pool_cap]
    n_test = max(10, int(len(unique) * test_fraction))
    test_rows, pool_rows = unique[:n_test], unique[n_test:]

    schema = CategorySchema.from_raw({l for _, l in unique}, include_rare=True)

    def make(rows_, prefix):
        out = []
        for i, (t, l) in enumerate(rows_):
            gold = schema.validate(l) or Label("_rare_")
            out.append(Instance(id=f"{prefix}-{i}", text=t, gold_label=gold))
        return out

    pool = make(pool_rows, "p")
    test = make(test_rows, "t")

    oracle = build_oracle(cfg.get("oracle", {"provider": "simulated", "noise": 0.0}))
    out_dir = artifacts_root / run.id
    out_dir.mkdir(parents=True, exist_ok=True)

    budget = min(int(params.get("budget", max(50, len(pool) // 3))), len(pool))
    result = run_active_learning(
        pool=pool, test=test, schema=schema,
        classifier_factory=PVBinClassifier,
        oracle=oracle,
        strategy=params.get("strategy", "entropy"),
        budget=budget,
        batch_size=int(params.get("batch_size", max(10, budget // 10))),
        initial_size=int(params.get("initial_size", max(10, budget // 10))),
        seed=seed,
        output_path=out_dir / "iterations.jsonl",
    )
    summary = result.summary()
    summary.update({
        "pool_size": len(pool), "test_size": len(test),
        "n_classes": len(schema), "oracle_id": oracle.oracle_id,
    })
    return summary
