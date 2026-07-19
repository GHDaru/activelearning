"""Catálogo dos experimentos da tese — execução e replay pela interface.

Cada entrada declara: identidade científica (pilar, pergunta), presets de
execução (comandos reais dos runners, todos com retomada por estado) e a
especificação dos artefatos gravados (o "replay": os mesmos arquivos que
sustentam os números da tese, lidos do disco e renderizados na UI).
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / "experiments"
JOBS = EXP / "api_jobs"
# repositório do programa experimental original (P1/P2/AG/oráculo-piloto)
LEGACY = Path(os.environ.get("FALCO_LEGACY_ROOT",
                             str(ROOT.parent / "activetextclassification")))

# kind: json | curve | table | csv (tabela) | text (trecho)
# legado=True: código/artefatos no repositório activetextclassification
CATALOG: list[dict[str, Any]] = [
    {
        "id": "p1-original",
        "titulo": "P1 original — Sensibilidade de L0 (47 tamanhos × 30 repetições)",
        "pilar": "P1", "legado": True, "auditoria": "verificado",
        "pergunta": "O estudo original que a reexecução (P1-replay) valida.",
        "descricao": "Código: examples/L0_experimento.ipynb (repositório "
                     "activetextclassification). Artefatos: "
                     "l0_random_impact_analysis_outputs/ (inclui a tabela do "
                     "apêndice da tese). Divergência da reexecução: ≤0,7 p.p.",
        "duracao": "dias (execução original; use o replay para reproduzir)",
        "requer_chave": False, "presets": {},
        "artefatos": [
            {"label": "Tabela estatística original (apêndice da tese)",
             "path_abs": "examples/l0_random_impact_analysis_outputs/l0_stats_table_analyzed.tex",
             "kind": "text"},
        ],
    },
    {
        "id": "ag-envelope",
        "titulo": "AG original — Envelope evolutivo de L0 (pop. 50 × 100 gerações)",
        "pilar": "P1", "legado": True, "auditoria": "verificado",
        "pergunta": "Quanto a composição pode render além do sorteio? (limites max/min)",
        "descricao": "Código: optimization/genetic_l0_optimizerv4.py — população 50, "
                     "100 gerações, elitismo, 4 cenários (max/min × acurácia/Macro F1), "
                     "checkpoints .pkl por célula. Artefatos: "
                     "examples/ag_optimization_results_L0_*/ + allag.xlsx. "
                     "ATENÇÃO: protocolo original é circular (aptidão = relato); a "
                     "quantificação da inflação (−6,3 p.p.) está no P1-replay.",
        "duracao": "dias (original); mecanismo reproduzível pelo replay do AG",
        "requer_chave": False, "presets": {},
        "artefatos": [
            {"label": "Evolução da aptidão — acurácia máx. (L0=100.000, v2)",
             "path_abs": "examples/ag_optimization_results_L0_100000v2/ag_detailed_fitnessACCURACY_MAXIMIZE.csv",
             "kind": "csv"},
            {"label": "Melhor indivíduo — acurácia máx. (L0=30.000, v1)",
             "path_abs": "examples/ag_optimization_results_L0_30000v1/ag_best_l0_ACCURACY_MAXIMIZE.csv",
             "kind": "csv"},
        ],
    },
    {
        "id": "p2-drisl",
        "titulo": "P2 original — DRI-SL vs. envelope do AG (cold start)",
        "pilar": "P2", "legado": True, "auditoria": "pendente",
        "pergunta": "A heurística sem rótulos supera o otimizador supervisionado?",
        "descricao": "Código: cold_start/dri_cluster.py (DRIClusterColdStart: KMeans "
                     "semântico + relevância TF-IDF + novidade lexical) e "
                     "examples/coldstart_evaluate.ipynb. Os números da tab:drisl-vs-ag "
                     "da tese saem daqui. AUDITORIA PENDENTE: localizar e verificar os "
                     "artefatos numéricos finais da comparação e portar um runner "
                     "reproduzível para esta biblioteca.",
        "duracao": "horas (original)",
        "requer_chave": False, "presets": {},
        "artefatos": [
            {"label": "Notebook de avaliação (fonte dos números)",
             "path_abs": "examples/coldstart_evaluate.ipynb", "kind": "presenca"},
        ],
    },
    {
        "id": "oraculo-piloto",
        "titulo": "Oráculo piloto (legado) — instrumento sem restrição de saída",
        "pilar": "P3", "legado": True, "auditoria": "verificado",
        "pergunta": "A origem do achado RQ4: o instrumento defeituoso replicado no E0.",
        "descricao": "Código: examples/oraculo.ipynb + data_oraculo/ (repositório "
                     "legado). O esquema de saída livre deste piloto gerou os falsos "
                     "erros de fraseado que motivaram o modo enum e o braço free do E0.",
        "duracao": "—", "requer_chave": True, "presets": {},
        "artefatos": [
            {"label": "Notebook do piloto", "path_abs": "examples/oraculo.ipynb",
             "kind": "presenca"},
        ],
    },
    {
        "id": "p1",
        "titulo": "P1 — Sensibilidade do conjunto inicial (replay)",
        "pilar": "P1",
        "pergunta": "Quanto a composição de L0 importa, por tamanho?",
        "descricao": "Reexecução independente do estudo de sensibilidade "
                     "(15 tamanhos × 10 repetições, protocolo com deduplicação). "
                     "Divergência ≤0,7 p.p. do estudo original.",
        "duracao": "~1–2 h (retoma do ponto salvo; instantâneo se completo)",
        "requer_chave": False,
        "presets": {"executar": ["python", "experiments/p1/replay_l0_sensitivity.py"]},
        "artefatos": [
            {"label": "Curva tamanho × acurácia média", "path": "p1/results/replay_l0.jsonl",
             "kind": "curve", "x": "size", "y": "acc_mean"},
            {"label": "Replay do AG (mecanismo + circularidade)",
             "path": "p1/results/replay_ga.jsonl", "kind": "table"},
        ],
    },
    {
        "id": "e0",
        "titulo": "E0 — Avaliação fatorial de oráculos LLM",
        "pilar": "P3",
        "pergunta": "Acurácia, custo e perfil de erro de cada oráculo candidato?",
        "descricao": "6 oráculos × 2 amostras pareadas (S-rand n=1.000; S-strat "
                     "n=1.863), IC de Wilson, McNemar pareado, custo por mil rótulos.",
        "duracao": "horas (limitado pela vazão dos provedores)",
        "requer_chave": True,
        "presets": {},  # execução guiada por config; pela UI apenas replay
        "artefatos": [
            {"label": "Sumário por oráculo/amostra", "path": "e0/results/e0_summary.json", "kind": "json"},
            {"label": "Pareamentos (McNemar)", "path": "e0/results/e0_mcnemar.json", "kind": "json"},
        ],
    },
    {
        "id": "e0p",
        "titulo": "E0-P — Ablação de prompt (faca de dois gumes)",
        "pilar": "P3",
        "pergunta": "Regras de fronteira no prompt ajudam? Onde?",
        "descricao": "Variantes v4a (regras) e v4b (regras+exemplos) pareadas "
                     "contra o prompt base: +4,6 p.p. na S-rand, −10,8 p.p. na S-strat.",
        "duracao": "~1 h (requer chave de API)",
        "requer_chave": True,
        "presets": {},
        "artefatos": [
            {"label": "Análise pareada por variante", "path": "e0p/results/analysis.json", "kind": "json"},
        ],
    },
    {
        "id": "e1e4",
        "titulo": "E1/E4 — Estratégias × ruído de oráculo",
        "pilar": "P4",
        "pergunta": "Qual estratégia vence, e quanto o ruído degrada?",
        "descricao": "Varredura de 5 estratégias × 8 sementes (E1) e ruído "
                     "ε∈{0; 0,1; 0,2; 0,4} (E4). Retenção de 87/74/54% do Macro F1.",
        "duracao": "~horas (retoma; instantâneo se completo)",
        "requer_chave": False,
        "presets": {"executar": ["python", "experiments/e1e4/run_sweeps.py"]},
        "artefatos": [
            {"label": "Análise consolidada (rankings, Wilcoxon)", "path": "e1e4/results/analysis.json", "kind": "json"},
            {"label": "Baseline de supervisão completa", "path": "e1e4/results/baseline.json", "kind": "json"},
        ],
    },
    {
        "id": "e5cycle",
        "titulo": "E5 — Ciclo FALCO real com oráculo gratuito",
        "pilar": "P4",
        "pergunta": "O laço completo funciona ponta a ponta com LLM real, a custo zero?",
        "descricao": "DRI-SL → entropia → oráculo nemotron (NIM) com parada por "
                     "estagnação: encerrou em 6.009 (PVBin) e 4.742 (SGD) de 15.000.",
        "duracao": "smoke ~10 min (oráculo simulado); real requer chave NIM",
        "requer_chave": False,
        "presets": {"smoke (oráculo simulado)":
                    ["python", "experiments/e5cycle/run_cycle.py", "--simulated",
                     "--classifier", "pvbin", "--tag", "_ui"]},
        "artefatos": [
            {"label": "Ciclo real PVBin (b15k)", "path": "e5cycle/results/cycle_pvbin_b15k.json",
             "kind": "cycle"},
            {"label": "Ciclo real SGD (b15k)", "path": "e5cycle/results/cycle_sgd_b15k.json",
             "kind": "cycle"},
        ],
    },
    {
        "id": "e6",
        "titulo": "E6 — Viés de autoavaliação em escala populacional",
        "pilar": "P4",
        "pergunta": "A avaliação nos próprios dados coletados engana? Em que direção?",
        "descricao": "Pool 50k / população 181k reservada; 5 seletores × 2 "
                     "classificadores. Acurácia interna subestima (−14 p.p.); "
                     "Macro F1 interno superestima (+34 p.p.).",
        "duracao": "smoke ~15 min; célula completa ~40 min",
        "requer_chave": False,
        "presets": {"smoke (entropia × SGD)":
                    ["python", "experiments/e6population/run_population_curve.py",
                     "--smoke", "--classifier", "sgd", "--strategy", "entropy"]},
        "artefatos": [
            {"label": "Entropia × SGD — F1 na população",
             "path": "e6population/results/popcurve_sgd_entropy.jsonl",
             "kind": "curve", "x": "n_labels", "y": "f1_ext"},
            {"label": "Aleatório × SGD — F1 na população",
             "path": "e6population/results/popcurve_sgd_random.jsonl",
             "kind": "curve", "x": "n_labels", "y": "f1_ext"},
            {"label": "Análise consolidada", "path": "e6population/results/analysis.json", "kind": "json"},
        ],
    },
    {
        "id": "e3prime",
        "titulo": "E3′ — BERTimbau julga o pipeline (hipótese central)",
        "pilar": "P4",
        "pergunta": "O modelo forte treinado no que o pipeline coletou chega a 95% da régua?",
        "descricao": "5 braços + varredura de orçamento na população reservada. "
                     "Veredito: refutada na configuração executada; 94,1% da régua "
                     "em acurácia com 30% dos rótulos.",
        "duracao": "smoke ~4 min (CPU); braços completos ~horas (GPU recomendada)",
        "requer_chave": False,
        "presets": {"smoke (validação da cadeia)":
                    ["python", "experiments/e2e3/run_smoke_cpu.py",
                     "--classes", "20", "--per-class", "20", "--epochs", "1"]},
        "artefatos": [
            {"label": "Braços A–E + varredura", "path": "e2e3/results/e3prime_*.json",
             "kind": "json-glob"},
        ],
    },
]


def _artifact_path(spec: dict) -> Path:
    if "path_abs" in spec:
        return LEGACY / spec["path_abs"]
    return EXP / spec["path"]


def _artifact_ready(spec: dict) -> bool:
    p = _artifact_path(spec)
    if "path" in spec and "*" in spec.get("path", ""):
        return any(p.parent.glob(p.name))
    return p.exists()


def _job_path(exp_id: str) -> Path:
    return JOBS / f"{exp_id}.json"


def job_status(exp_id: str) -> dict | None:
    jp = _job_path(exp_id)
    if not jp.exists():
        return None
    job = json.loads(jp.read_text())
    try:
        os.kill(job["pid"], 0)
        job["status"] = "executando"
    except (OSError, ProcessLookupError):
        job["status"] = "finalizado"
    return job


def catalog_status() -> list[dict]:
    out = []
    for exp in CATALOG:
        ready = [a["label"] for a in exp["artefatos"] if _artifact_ready(a)]
        out.append({
            **{k: exp[k] for k in ("id", "titulo", "pilar", "pergunta",
                                   "descricao", "duracao", "requer_chave")},
            "legado": exp.get("legado", False),
            "auditoria": exp.get("auditoria"),
            "presets": list(exp["presets"].keys()),
            "artefatos_disponiveis": ready,
            "n_artefatos": len(exp["artefatos"]),
            "job": job_status(exp["id"]),
        })
    return out


def load_results(exp_id: str) -> dict:
    exp = next((e for e in CATALOG if e["id"] == exp_id), None)
    if exp is None:
        raise KeyError(exp_id)
    blocks: list[dict] = []
    for spec in exp["artefatos"]:
        p = _artifact_path(spec)
        if spec["kind"] == "json-glob":
            for f in sorted(p.parent.glob(p.name)):
                if f.name.endswith("_pred.json"):
                    continue
                blocks.append({"label": f.stem, "kind": "json",
                               "data": json.loads(f.read_text())})
            continue
        if not p.exists():
            blocks.append({"label": spec["label"], "kind": "ausente"})
            continue
        if spec["kind"] == "json":
            blocks.append({"label": spec["label"], "kind": "json",
                           "data": json.loads(p.read_text())})
        elif spec["kind"] == "cycle":
            d = json.loads(p.read_text())
            curve = [{"n": n, "y": v} for n, v in d.get("curve_test", [])]
            resumo = {k: d.get(k) for k in
                      ("classifier", "final_macro_f1", "lce_macro_f1",
                       "n_labeled", "invalid_labels", "wall_seconds")}
            blocks.append({"label": spec["label"], "kind": "curve",
                           "points": curve, "resumo": resumo})
        elif spec["kind"] == "curve":
            pts, seen = [], set()
            for line in p.read_text().splitlines():
                if line.strip():
                    d = json.loads(line)
                    x = d.get(spec["x"])
                    if x not in seen:
                        seen.add(x)
                        pts.append({"n": x, "y": d.get(spec["y"])})
            pts.sort(key=lambda q: q["n"])
            blocks.append({"label": spec["label"], "kind": "curve", "points": pts})
        elif spec["kind"] == "text":
            blocks.append({"label": spec["label"], "kind": "text",
                           "text": "\n".join(p.read_text(errors="replace").splitlines()[:50])})
        elif spec["kind"] == "csv":
            import csv as _csv
            with p.open(encoding="utf-8", errors="replace") as fh:
                rows = list(_csv.DictReader(fh))[:20]
            blocks.append({"label": spec["label"], "kind": "table", "rows": rows})
        elif spec["kind"] == "presenca":
            blocks.append({"label": spec["label"], "kind": "text",
                           "text": f"artefato presente no repositório legado: {p}"})
        elif spec["kind"] == "table":
            rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
            blocks.append({"label": spec["label"], "kind": "table", "rows": rows[:30]})
    return {"id": exp_id, "titulo": exp["titulo"], "blocks": blocks}


def execute(exp_id: str, preset: str) -> dict:
    exp = next((e for e in CATALOG if e["id"] == exp_id), None)
    if exp is None:
        raise KeyError(exp_id)
    if preset not in exp["presets"]:
        raise ValueError(f"preset desconhecido: {preset}")
    job = job_status(exp_id)
    if job and job.get("status") == "executando":
        raise RuntimeError("já existe execução em andamento para este experimento")
    JOBS.mkdir(parents=True, exist_ok=True)
    log = JOBS / f"{exp_id}.log"
    with log.open("w") as fh:
        proc = subprocess.Popen(
            exp["presets"][preset], cwd=str(ROOT), stdout=fh, stderr=fh,
            start_new_session=True,
        )
    record = {"pid": proc.pid, "preset": preset, "started": time.time(),
              "log": str(log)}
    _job_path(exp_id).write_text(json.dumps(record))
    return {**record, "status": "executando"}


def stop(exp_id: str) -> dict:
    job = job_status(exp_id)
    if not job or job.get("status") != "executando":
        raise RuntimeError("nenhuma execução em andamento")
    os.killpg(os.getpgid(job["pid"]), signal.SIGTERM)
    return {"status": "interrompido"}


def tail_log(exp_id: str, lines: int = 60) -> str:
    log = JOBS / f"{exp_id}.log"
    if not log.exists():
        return ""
    return "\n".join(log.read_text(errors="replace").splitlines()[-lines:])
