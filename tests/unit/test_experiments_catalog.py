"""Testes do catálogo de experimentos: status, replay e execução em subprocess."""
import json
import time

import pytest

pytest.importorskip("fastapi")

from activelearning.adapters.api import experiments_catalog as cat  # noqa: E402


def test_catalog_ids_unicos_e_campos_obrigatorios():
    status = cat.catalog_status()
    ids = [e["id"] for e in status]
    assert len(ids) == len(set(ids)), "ids duplicados no catálogo"
    for e in status:
        for campo in ("titulo", "pilar", "pergunta", "descricao",
                      "duracao", "requer_chave", "presets",
                      "artefatos_disponiveis", "n_artefatos"):
            assert campo in e, f"{e['id']} sem campo {campo}"


def test_entradas_legado_tem_estado_de_auditoria():
    for e in cat.catalog_status():
        if e.get("legado"):
            assert e.get("auditoria") in ("verificado", "pendente"), \
                f"{e['id']} legado sem estado de auditoria"


def test_load_results_experimento_desconhecido():
    with pytest.raises(KeyError):
        cat.load_results("nao-existe")


def test_load_results_artefato_ausente_vira_bloco_ausente(monkeypatch, tmp_path):
    monkeypatch.setattr(cat, "EXP", tmp_path)
    monkeypatch.setattr(cat, "LEGACY", tmp_path)
    entry = {
        "id": "fake", "titulo": "Fake", "pilar": "PX", "pergunta": "?",
        "descricao": "", "duracao": "", "requer_chave": False, "presets": {},
        "artefatos": [{"label": "x", "path": "fake/results/nao_existe.json",
                       "kind": "json"}],
    }
    monkeypatch.setattr(cat, "CATALOG", [entry])
    res = cat.load_results("fake")
    assert res["blocks"][0]["kind"] == "ausente"


def test_load_results_json_curve_e_csv(monkeypatch, tmp_path):
    monkeypatch.setattr(cat, "EXP", tmp_path)
    monkeypatch.setattr(cat, "LEGACY", tmp_path)
    d = tmp_path / "fake" / "results"
    d.mkdir(parents=True)
    (d / "resumo.json").write_text(json.dumps({"acc": 0.9}))
    (d / "curva.jsonl").write_text(
        "\n".join(json.dumps({"n": n, "f1": n / 100}) for n in (10, 20, 30)))
    (d / "tabela.csv").write_text("a,b\n1,2\n3,4\n")
    entry = {
        "id": "fake", "titulo": "Fake", "pilar": "PX", "pergunta": "?",
        "descricao": "", "duracao": "", "requer_chave": False, "presets": {},
        "artefatos": [
            {"label": "resumo", "path": "fake/results/resumo.json", "kind": "json"},
            {"label": "curva", "path": "fake/results/curva.jsonl",
             "kind": "curve", "x": "n", "y": "f1"},
            {"label": "tabela", "path_abs": "fake/results/tabela.csv", "kind": "csv"},
        ],
    }
    monkeypatch.setattr(cat, "CATALOG", [entry])
    res = cat.load_results("fake")
    kinds = {b["label"]: b for b in res["blocks"]}
    assert kinds["resumo"]["data"] == {"acc": 0.9}
    assert [p["n"] for p in kinds["curva"]["points"]] == [10, 20, 30]
    assert kinds["tabela"]["rows"] == [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]


def test_execute_roda_subprocess_e_registra_job(monkeypatch, tmp_path):
    monkeypatch.setattr(cat, "JOBS", tmp_path / "jobs")
    monkeypatch.setattr(cat, "ROOT", tmp_path)
    entry = {
        "id": "fake", "titulo": "Fake", "pilar": "PX", "pergunta": "?",
        "descricao": "", "duracao": "", "requer_chave": False,
        "presets": {"eco": ["python", "-c", "print('rodou')"]},
        "artefatos": [],
    }
    monkeypatch.setattr(cat, "CATALOG", [entry])
    job = cat.execute("fake", "eco")
    assert job["status"] == "executando"
    for _ in range(50):  # aguarda o processo curto encerrar
        if cat.job_status("fake")["status"] == "finalizado":
            break
        time.sleep(0.1)
    assert cat.job_status("fake")["status"] == "finalizado"
    assert "rodou" in cat.tail_log("fake")


def test_execute_preset_invalido(monkeypatch, tmp_path):
    monkeypatch.setattr(cat, "JOBS", tmp_path / "jobs")
    entry = {"id": "fake", "titulo": "F", "pilar": "P", "pergunta": "?",
             "descricao": "", "duracao": "", "requer_chave": False,
             "presets": {}, "artefatos": []}
    monkeypatch.setattr(cat, "CATALOG", [entry])
    with pytest.raises(ValueError):
        cat.execute("fake", "inexistente")
