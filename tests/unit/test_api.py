"""Testes da API do FlowBuilder: SQLite temporário + oráculo simulado (offline)."""
import json
import time

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from activelearning.adapters.api.app import create_app  # noqa: E402
from activelearning.adapters.api.settings import Settings  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    csv_path = tmp_path / "mini.csv"
    lines = ["nm_item,nm_product"]
    for i in range(30):
        lines.append(f"CERVEJA LATA {i},cerveja")
        lines.append(f"ARROZ TIPO1 {i}KG,arroz")
    csv_path.write_text("\n".join(lines), encoding="utf-8")

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": 42,
                "dataset": {
                    "csv_path": str(csv_path),
                    "text_column": "nm_item",
                    "label_column": "nm_product",
                    "min_samples_per_class": 2,
                },
                "samples": {"random_size": 20, "stratified_per_class": 2},
                "oracles": [],
            }
        ),
        encoding="utf-8",
    )
    settings = Settings(
        root=tmp_path,
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        experiment_config=config_path,
        artifacts_root=tmp_path / "artifacts",
        cors_origins=["http://localhost:5173"],
    )
    return TestClient(create_app(settings))


def test_health(client):
    body = client.get("/api/health").json()
    assert body == {"status": "ok", "database": "sqlite"}


def test_oracles_include_simulated_and_no_secrets(client):
    specs = client.get("/api/oracles").json()
    assert specs[0]["provider"] == "simulated"
    assert "api_key" not in json.dumps(specs).lower()


def test_run_lifecycle_completes_with_artifacts(client):
    created = client.post(
        "/api/runs",
        json={
            "name": "smoke simulado",
            "sample": "rand",
            "limit": 10,
            "oracle": {"provider": "simulated", "noise": 0.0},
        },
    )
    assert created.status_code == 201
    run_id = created.json()["id"]

    for _ in range(50):
        run = client.get(f"/api/runs/{run_id}").json()
        if run["status"] in {"completed", "failed"}:
            break
        time.sleep(0.1)
    assert run["status"] == "completed", run.get("error")
    assert run["report"]["accuracy"] == 1.0
    assert run["report"]["n_total"] == 10
    assert run["artifacts_dir"]

    listed = client.get("/api/runs").json()
    assert any(r["id"] == run_id for r in listed)


def test_unknown_run_is_404(client):
    assert client.get("/api/runs/nao-existe").status_code == 404
