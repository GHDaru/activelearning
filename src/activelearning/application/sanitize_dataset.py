"""Caso de uso: saneamento de um conjunto de dados enviado pelo usuário.

Aplica o pipeline de auditoria da tese a um CSV arbitrário (texto, rótulo):

1. remove linhas com texto ou rótulo vazios;
2. remove rótulos OPERACIONAIS (status de cadastro, não categoria — ex.:
   'inativo'), lista configurável;
3. faz o censo de CONFLITOS de gabarito (mesmo texto normalizado com 2+
   rótulos distintos) — mantidos, com relatório (decisão da tese: ruído real
   de domínio; ver análise de sensibilidade);
4. faz o censo de DUPLICATAS exatas (texto+rótulo) — mantidas, com aviso de
   que particionamentos para treino devem deduplicar antes do split.

Devolve o relatório completo e grava o CSV saneado ao lado do original.
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from activelearning.domain.instances import normalize_label

DEFAULT_OPERATIONAL_LABELS = ("inativo",)


@dataclass
class SanitizationReport:
    n_rows_in: int = 0
    n_rows_out: int = 0
    removed_empty: int = 0
    removed_operational: int = 0
    operational_labels: list[str] = field(default_factory=list)
    n_classes: int = 0
    n_conflicting_texts: int = 0
    n_conflicting_rows: int = 0
    conflict_examples: list[dict] = field(default_factory=list)
    n_exact_duplicates: int = 0
    class_histogram_top: list[list] = field(default_factory=list)
    n_rare_classes_lt5: int = 0

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def sanitize_csv(
    input_path: Path,
    output_path: Path,
    text_column: str,
    label_column: str,
    operational_labels: tuple[str, ...] = DEFAULT_OPERATIONAL_LABELS,
    max_conflict_examples: int = 10,
) -> SanitizationReport:
    report = SanitizationReport(operational_labels=list(operational_labels))
    operational = {normalize_label(l) for l in operational_labels}

    kept: list[tuple[str, str]] = []
    with input_path.open(encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or text_column not in reader.fieldnames \
                or label_column not in reader.fieldnames:
            raise ValueError(
                f"colunas {text_column!r}/{label_column!r} não encontradas; "
                f"disponíveis: {reader.fieldnames}"
            )
        for row in reader:
            report.n_rows_in += 1
            text = (row.get(text_column) or "").strip()
            label = (row.get(label_column) or "").strip()
            if not text or not label:
                report.removed_empty += 1
                continue
            if normalize_label(label) in operational:
                report.removed_operational += 1
                continue
            kept.append((text, label))

    report.n_rows_out = len(kept)

    golds_by_text: dict[str, set[str]] = defaultdict(set)
    pair_counter: Counter = Counter()
    class_counter: Counter = Counter()
    for text, label in kept:
        nl = normalize_label(label)
        golds_by_text[normalize_label(text)].add(nl)
        pair_counter[(normalize_label(text), nl)] += 1
        class_counter[nl] += 1

    conflicting = {t: gs for t, gs in golds_by_text.items() if len(gs) > 1}
    report.n_conflicting_texts = len(conflicting)
    report.n_conflicting_rows = sum(
        1 for text, _ in kept if normalize_label(text) in conflicting
    )
    report.conflict_examples = [
        {"texto": t, "rotulos": sorted(gs)}
        for t, gs in sorted(conflicting.items())[:max_conflict_examples]
    ]
    report.n_exact_duplicates = sum(c - 1 for c in pair_counter.values() if c > 1)
    report.n_classes = len(class_counter)
    report.n_rare_classes_lt5 = sum(1 for c in class_counter.values() if c < 5)
    report.class_histogram_top = [
        [label, count] for label, count in class_counter.most_common(15)
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([text_column, label_column])
        writer.writerows(kept)

    return report
