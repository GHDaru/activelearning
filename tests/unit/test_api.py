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


def _upload_csv(client, n_per_class=30, with_noise=True):
    lines = ["descricao,categoria"]
    for i in range(n_per_class):
        lines.append(f"CERVEJA LATA GELADA {i},cerveja")
        lines.append(f"ARROZ TIPO UM {i}KG,arroz")
        lines.append(f"SABAO EM PO ROUPA {i},sabao")
    if with_noise:
        lines.append("UVA PASSA 150G,inativo")        # rótulo operacional
        lines.append("MILHARINA 500G,farinha")        # conflito (2 golds)
        lines.append("MILHARINA 500G,fuba")
        lines.append("CERVEJA LATA GELADA 0,cerveja")  # duplicata exata
        lines.append(",vazio")                          # texto vazio
    return client.post(
        "/api/datasets",
        files={"file": ("meu.csv", "\n".join(lines).encode(), "text/csv")},
        data={"name": "teste upload", "text_column": "descricao",
              "label_column": "categoria", "operational_labels": "inativo"},
    )


def test_dataset_upload_sanitizes_and_reports(client):
    resp = _upload_csv(client)
    assert resp.status_code == 201, resp.text
    ds = resp.json()
    rep = ds["report"]
    assert rep["removed_operational"] == 1
    assert rep["removed_empty"] == 1
    assert rep["n_conflicting_texts"] == 1
    assert rep["n_exact_duplicates"] == 1
    assert rep["n_rows_out"] == rep["n_rows_in"] - 2  # vazio + inativo removidos

    listed = client.get("/api/datasets").json()
    assert any(d["id"] == ds["id"] for d in listed)

    dl = client.get(f"/api/datasets/{ds['id']}/download?which=sanitized")
    assert dl.status_code == 200
    assert b"inativo" not in dl.content
    assert b"MILHARINA" in dl.content  # conflito mantido (decisão da tese)


def test_dataset_upload_bad_columns_is_422(client):
    resp = client.post(
        "/api/datasets",
        files={"file": ("x.csv", b"a,b\n1,2\n", "text/csv")},
        data={"name": "ruim", "text_column": "nao_existe", "label_column": "b"},
    )
    assert resp.status_code == 422


def test_active_learning_run_end_to_end(client):
    ds = _upload_csv(client, n_per_class=40, with_noise=False).json()
    created = client.post("/api/runs", json={
        "name": "al simulado", "kind": "active-learning", "dataset_id": ds["id"],
        "params": {"seed": 7, "budget": 60, "batch_size": 10, "initial_size": 10,
                   "strategy": "entropy", "pool_size": 500},
        "oracle": {"provider": "simulated", "noise": 0.0},
    })
    assert created.status_code == 201, created.text
    run_id = created.json()["id"]
    for _ in range(100):
        run = client.get(f"/api/runs/{run_id}").json()
        if run["status"] in {"completed", "failed"}:
            break
        time.sleep(0.1)
    assert run["status"] == "completed", run.get("error")
    rep = run["report"]
    assert rep["strategy"] == "entropy"
    assert rep["n_labeled"] == 60
    assert rep["final_macro_f1"] > 0.8
    assert len(rep["curve"]) >= 2


def test_active_learning_requires_dataset(client):
    resp = client.post("/api/runs", json={
        "name": "sem dataset", "kind": "active-learning",
        "oracle": {"provider": "simulated"},
    })
    assert resp.status_code == 422
