#!/usr/bin/env python3
"""Checagem executável dos invariantes do dataset (item `pipeline-reprodutivel`).

Dono: revisor2 (sessão de revisão da tese — plano de revisão, grupo `dados`).

Prova, a partir do CSV publicado e das funções REAIS do pipeline (nunca
cópias), os números citados na tese e em `data/DICIONARIO.md`:

  1. `retail_csv.load_rows_and_schema`  → 250.221 linhas · schema 621 valores
     (620 classes com ≥5 amostras + `_rare_`);
  2. `run_e3prime.load_base`            → 231.490 textos únicos · 714 classes;
     particionamento fixo (DATA_SEED=42): pool 50.000 (649 classes presentes),
     holdout do ciclo 4.000, população reservada 177.490;
  3. md5 do CSV = versão exata usada (item `md5-versao` do plano).

Uso:
  python3 scripts/check_dataset_invariants.py [caminho/do/dataset.csv]

Sai 0 com tudo verde; sai 1 e nomeia o invariante violado caso contrário
(DoD verificável, princípio IX da constituição da tese: critério = comando).
Sem dependências além da stdlib: os imports pesados de `run_e3prime`
(sklearn, torch via adaptador BERTimbau) são substituídos por stubs — apenas
`load_base` é exercitada, e ela não os usa.
"""
from __future__ import annotations

import hashlib
import sys
import types
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/e2e3"))

# Stubs: run_e3prime importa sklearn e o adaptador BERTimbau no topo do módulo,
# mas load_base não toca em nenhum dos dois. Stub explícito > dependência falsa.
_bert = types.ModuleType("activelearning.adapters.classifiers.bertimbau")
_bert.BertimbauClassifier = None
sys.modules["activelearning.adapters.classifiers.bertimbau"] = _bert
if "sklearn" not in sys.modules:
    _sk = types.ModuleType("sklearn")
    _skm = types.ModuleType("sklearn.metrics")
    _skm.accuracy_score = _skm.f1_score = None
    sys.modules["sklearn"] = _sk
    sys.modules["sklearn.metrics"] = _skm

from activelearning.adapters.datasets.retail_csv import load_rows_and_schema  # noqa: E402
import run_e3prime  # noqa: E402

ESPERADO = {
    "linhas_csv": 250_221,
    "md5_csv": "0682ee5ba077c180fb7a727fb200f154",
    "classes_normalizadas": 794,
    "classes_em_rare": 174,       # 794 - 620
    "schema_total": 621,          # 620 classes + _rare_
    "schema_classes": 620,
    "linhas_pos_filtro_ge2": 250_142,
    "classes_ge2": 715,
    "dedup_textos": 231_490,
    "dedup_classes": 714,
    "pool": 50_000,
    "classes_no_pool": 649,
    "populacao": 177_490,
}

_falhas: list[str] = []


def check(nome: str, obtido, esperado) -> None:
    ok = obtido == esperado
    print(f"[{'OK ' if ok else 'FAIL'}] {nome}: obtido={obtido} esperado={esperado}")
    if not ok:
        _falhas.append(nome)


def main() -> int:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/dataset.csv"
    print(f"dataset: {csv_path}")

    check("md5_csv", hashlib.md5(csv_path.read_bytes()).hexdigest(), ESPERADO["md5_csv"])

    rows, schema = load_rows_and_schema({"dataset": {
        "csv_path": str(csv_path), "text_column": "nm_item", "label_column": "nm_product",
    }})
    check("linhas_csv", len(rows), ESPERADO["linhas_csv"])
    check("schema_total (620 + _rare_)", len(schema), ESPERADO["schema_total"])
    check("schema tem _rare_", "_rare_" in schema.values, True)
    check("schema_classes (sem _rare_)",
          len([v for v in schema.values if v != "_rare_"]), ESPERADO["schema_classes"])

    # load_base lê ROOT/data/dataset.csv fixo; para caminho alternativo,
    # reaponta o _ROOT do módulo (mesma função, outra base — modo de teste).
    # Limitação documentada: o caminho alternativo precisa do layout
    # <raiz>/data/dataset.csv; fora dele, load_base falha com FileNotFoundError
    # (exit != 0 do mesmo jeito, mas sem nomear o invariante).
    if csv_path.resolve() != (ROOT / "data/dataset.csv").resolve():
        run_e3prime._ROOT = csv_path.resolve().parents[1]
    dedup = run_e3prime.load_base()
    classes = {l for _, l in dedup}
    check("dedup_textos", len(dedup), ESPERADO["dedup_textos"])
    check("dedup_classes", len(classes), ESPERADO["dedup_classes"])

    pool = dedup[:run_e3prime.POOL_SIZE]
    populacao = dedup[run_e3prime.POOL_SIZE + run_e3prime.CYCLE_HOLDOUT:]
    check("pool", len(pool), ESPERADO["pool"])
    check("classes_no_pool", len({l for _, l in pool}), ESPERADO["classes_no_pool"])
    check("particoes_somam_dedup (pool + holdout 4k + população)",
          len(pool) + run_e3prime.CYCLE_HOLDOUT + len(populacao), len(dedup))
    check("populacao", len(populacao), ESPERADO["populacao"])

    # coerência interna do dicionário (cadeia completa da prosa):
    from activelearning.domain.instances import normalize_label
    norm = Counter(normalize_label(l) for _, l in rows)
    check("classes_normalizadas", len(norm), ESPERADO["classes_normalizadas"])
    check("classes_norm_ge5 (base do schema)",
          sum(1 for n in norm.values() if n >= 5), ESPERADO["schema_classes"])
    check("classes_em_rare (794 - 620)",
          sum(1 for n in norm.values() if n < 5), ESPERADO["classes_em_rare"])
    check("linhas_pos_filtro_ge2",
          sum(n for n in norm.values() if n >= 2), ESPERADO["linhas_pos_filtro_ge2"])
    check("classes_ge2 (715; o dedup elimina 1 -> 714)",
          sum(1 for n in norm.values() if n >= 2), ESPERADO["classes_ge2"])

    if _falhas:
        print(f"\nRESULTADO: FALHOU ({len(_falhas)} invariante(s)): {', '.join(_falhas)}")
        return 1
    print("\nRESULTADO: OK — todos os invariantes do dataset conferem.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
