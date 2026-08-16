#!/usr/bin/env python3
"""Checagem executável do dataset (DoD do princípio "nenhum número sem artefato").

Prova, a partir do data/dataset.csv publicado e sem dependências de ML, as
contagens citadas na tese e documentadas em data/DICIONARIO.md:

  1. arquivo: 250.221 linhas de dados, colunas nm_item/nm_product, zero
     rótulo operacional 'inativo';
  2. normalização: 794 rótulos crus -> 794 normalizados (nenhuma fusão);
  3. base experimental E5/E6/E3' (receita load_base de
     experiments/e2e3/run_e3prime.py, reimplementada aqui só com stdlib para
     não arrastar sklearn): 231.490 textos deduplicados / 714 classes, com o
     passo 715->714 explicado (classe 'pomada massageadora' eliminada no dedup);
  4. CategorySchema do oráculo (load_rows_and_schema de
     src/activelearning/adapters/datasets/retail_csv.py, importado e executado
     de verdade): 621 valores = 620 rótulos de produto + '_rare_'.

Saída: uma linha PASS/FAIL por checagem; código de saída 0 somente se todas
passarem. Uso:  python scripts/check_dataset.py  [caminho-do-csv]
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from activelearning.adapters.datasets.retail_csv import load_rows_and_schema  # noqa: E402
from activelearning.domain.instances import normalize_label  # noqa: E402

ESPERADO = {
    "linhas": 250_221,
    "rotulos_crus": 794,
    "rotulos_normalizados": 794,
    "classes_pos_filtro_ge2": 715,
    "textos_dedup": 231_490,
    "classes_base": 714,
    "classe_perdida_no_dedup": "pomada massageadora",
    "schema_total": 621,
    "schema_produtos": 620,
}

_falhas: list[str] = []


def check(nome: str, obtido, esperado) -> None:
    ok = obtido == esperado
    print(f"[{'PASS' if ok else 'FAIL'}] {nome}: obtido={obtido!r} esperado={esperado!r}")
    if not ok:
        _falhas.append(nome)


def main() -> int:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _ROOT / "data/dataset.csv"

    # ---- 1. arquivo ----
    with csv_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        check("colunas", reader.fieldnames, ["nm_item", "nm_product"])
        rows = [(r["nm_item"], r["nm_product"]) for r in reader]
    check("linhas de dados", len(rows), ESPERADO["linhas"])
    check("linhas com rótulo 'inativo'",
          sum(1 for _, l in rows if l.strip().lower() == "inativo"), 0)

    # ---- 2. normalização não funde rótulos ----
    crus = {l for _, l in rows}
    normalizados = {normalize_label(l) for l in crus}
    check("rótulos crus distintos", len(crus), ESPERADO["rotulos_crus"])
    check("rótulos normalizados distintos", len(normalizados),
          ESPERADO["rotulos_normalizados"])

    # ---- 3. base experimental (receita load_base, passo a passo) ----
    norm_rows = [(t, normalize_label(l)) for t, l in rows]
    counts = Counter(l for _, l in norm_rows)
    filtradas = [(t, l) for t, l in norm_rows if counts[l] >= 2]
    check("classes após filtro >=2", len({l for _, l in filtradas}),
          ESPERADO["classes_pos_filtro_ge2"])
    seen: set[str] = set()
    dedup = []
    for t, l in filtradas:
        k = t.strip().lower()
        if k not in seen:
            seen.add(k)
            dedup.append((t, l))
    check("textos deduplicados (base E5/E6/E3')", len(dedup),
          ESPERADO["textos_dedup"])
    classes_base = {l for _, l in dedup}
    check("classes na base experimental", len(classes_base),
          ESPERADO["classes_base"])
    perdidas = {l for _, l in filtradas} - classes_base
    check("classe eliminada pelo dedup (explica 715->714)", perdidas,
          {ESPERADO["classe_perdida_no_dedup"]})

    # ---- 4. CategorySchema do oráculo (código de produção de verdade) ----
    cfg = {"dataset": {"csv_path": str(csv_path), "text_column": "nm_item",
                       "label_column": "nm_product", "min_samples_per_class": 5}}
    _, schema = load_rows_and_schema(cfg)
    check("CategorySchema: total de valores", len(schema),
          ESPERADO["schema_total"])
    check("CategorySchema: contém '_rare_'", "_rare_" in schema.values, True)
    check("CategorySchema: rótulos de produto (sem _rare_)",
          len(schema.values) - 1, ESPERADO["schema_produtos"])

    if _falhas:
        print(f"\nFAIL — {len(_falhas)} checagem(ns) falharam: {', '.join(_falhas)}")
        return 1
    print("\nPASS — dataset reproduz todas as contagens documentadas "
          "(231.490/714 e 620+_rare_=621).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
