"""Carga do Retail Product Description-Ptbr (CSV) e amostragem pareada.

Implementação canônica usada pelo runner E0 e pela API do FlowBuilder — a mesma
config JSON (seção ``dataset`` + ``samples``) vale para os dois, mantendo os IDs
``e0-{índice}`` estáveis entre execuções (rastreabilidade, Princípio I).
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

from activelearning.domain.instances import CategorySchema, Instance, Label


def load_rows_and_schema(config: dict) -> tuple[list[tuple[str, str]], CategorySchema]:
    data_cfg = config["dataset"]
    path = Path(data_cfg["csv_path"])
    text_col, label_col = data_cfg["text_column"], data_cfg["label_column"]
    min_per_class = int(data_cfg.get("min_samples_per_class", 5))
    # Rótulos operacionais que vazaram como categoria (auditoria da base:
    # 'inativo' = 144 linhas de status, não tipo de produto).
    excluded = {e.lower() for e in data_cfg.get("exclude_labels", ["inativo"])}

    rows: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            text = (row.get(text_col) or "").strip()
            label = (row.get(label_col) or "").strip()
            if text and label and label.lower() not in excluded:
                rows.append((text, label))

    counts: dict[str, int] = {}
    for _, label in rows:
        counts[label] = counts.get(label, 0) + 1
    frequent = {label for label, n in counts.items() if n >= min_per_class}
    schema = CategorySchema.from_raw(frequent, include_rare=True)
    return rows, schema


def to_instance(rows: list[tuple[str, str]], i: int, schema: CategorySchema) -> Instance:
    text, label = rows[i]
    gold = schema.validate(label) or Label("_rare_")
    return Instance(id=f"e0-{i}", text=text, gold_label=gold)


def build_samples(
    config: dict, rows: list[tuple[str, str]], schema: CategorySchema
) -> dict[str, list[Instance]]:
    """S-rand (aleatória simples) e S-strat (k por classe), ambas com seed fixa."""
    seed = int(config.get("seed", 42))
    samples: dict[str, list[Instance]] = {}

    rand_size = int(config["samples"]["random_size"])
    rng = random.Random(seed)
    rand_indices = rng.sample(range(len(rows)), min(rand_size, len(rows)))
    samples["rand"] = [to_instance(rows, i, schema) for i in rand_indices]

    k = int(config["samples"]["stratified_per_class"])
    by_class: dict[str, list[int]] = {}
    for i, (_, label) in enumerate(rows):
        normalized = schema.validate(label)
        key = normalized.value if normalized else "_rare_"
        by_class.setdefault(key, []).append(i)
    rng = random.Random(seed)
    strat_indices: list[int] = []
    for key in sorted(by_class):
        pool = by_class[key]
        strat_indices.extend(rng.sample(pool, min(k, len(pool))))
    samples["strat"] = [to_instance(rows, i, schema) for i in strat_indices]
    return samples
