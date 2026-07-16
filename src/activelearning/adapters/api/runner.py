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
from activelearning.adapters.persistence.models import Run
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
