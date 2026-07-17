"""Caso de uso: laço de aprendizado ativo pool-based simulado.

Executa o ciclo L0 → treina → seleciona → rotula (oráculo) → retreina,
produzindo a curva de aprendizado (Macro F1 e acurácia no teste), a LCE e um
registro por iteração (rastreabilidade, Princípio I).

O conjunto de teste NUNCA participa de decisões do laço (apenas medição);
critérios adaptativos usam validação, quando fornecida.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from activelearning.domain.instances import CategorySchema, Instance
from activelearning.domain.metrics import LearningCurve, lce
from activelearning.domain.strategies import (
    entropy_scores,
    least_confidence_scores,
    select_random,
    select_top_k,
    smallest_margin_scores,
)


class TaskClassifier(Protocol):
    classes_: list[str]

    def fit(self, texts: list[str], labels: list[str]) -> "TaskClassifier": ...
    def predict(self, texts: list[str]) -> list[str]: ...
    def predict_proba(self, texts: list[str]) -> np.ndarray: ...


class OraclePort(Protocol):
    oracle_id: str

    def annotate(self, batch: list[Instance], schema: CategorySchema) -> list: ...


UNCERTAINTY = {
    "entropy": entropy_scores,
    "least_confidence": least_confidence_scores,
    "smallest_margin": smallest_margin_scores,
}
STRATEGIES = tuple(UNCERTAINTY) + ("random", "hybrid")


@dataclass
class ALResult:
    strategy: str
    seed: int
    curve_macro_f1: LearningCurve
    curve_accuracy: LearningCurve
    lce_macro_f1: float
    final_macro_f1: float
    final_accuracy: float
    n_labeled: int
    invalid_labels: int
    records: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "strategy": self.strategy,
            "seed": self.seed,
            "lce_macro_f1": round(self.lce_macro_f1, 4),
            "final_macro_f1": round(self.final_macro_f1, 4),
            "final_accuracy": round(self.final_accuracy, 4),
            "n_labeled": self.n_labeled,
            "invalid_labels": self.invalid_labels,
            "curve": [
                {"n": n, "macro_f1": round(f, 4)}
                for n, f in zip(self.curve_macro_f1.l_sizes, self.curve_macro_f1.scores)
            ],
        }


def _macro_f1(pred: list[str], gold: list[str]) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(gold, pred, average="macro", zero_division=0))


def run_active_learning(
    pool: list[Instance],
    test: list[Instance],
    schema: CategorySchema,
    classifier_factory: Callable[[], TaskClassifier],
    oracle: OraclePort,
    strategy: str = "entropy",
    budget: int = 1000,
    batch_size: int = 100,
    initial_size: int = 100,
    seed: int = 42,
    initial_indices: list[int] | None = None,
    baseline_performance: float | None = None,
    output_path: Path | None = None,
    hybrid_random_fraction: float = 0.5,
) -> ALResult:
    """Executa um ciclo completo de AL simulado sobre o pool com golds."""
    if strategy not in STRATEGIES:
        raise ValueError(f"estratégia desconhecida: {strategy!r} (opções: {STRATEGIES})")
    if budget > len(pool):
        raise ValueError("orçamento maior que o pool.")

    rng = np.random.default_rng(seed)
    test_texts = [i.text for i in test]
    test_gold = [i.gold_label.value for i in test]

    unlabeled = list(range(len(pool)))
    # L0: fornecido (ex.: DRI-SL) ou aleatório com semente
    if initial_indices is not None:
        l_indices = list(initial_indices)
    else:
        pick = rng.choice(len(unlabeled), size=min(initial_size, len(unlabeled)), replace=False)
        l_indices = [unlabeled[i] for i in pick]
    unlabeled = [i for i in unlabeled if i not in set(l_indices)]

    # rotulagem do L0 pelo oráculo
    labeled_texts: list[str] = []
    labeled_labels: list[str] = []
    invalid = 0

    def annotate(indices: list[int]) -> None:
        nonlocal invalid
        batch = [pool[i] for i in indices]
        annotations = oracle.annotate(batch, schema)
        for inst, ann in zip(batch, annotations):
            if ann.label is None:
                invalid += 1
                continue  # rótulo inválido: instância consumiu orçamento, não vira treino
            labeled_texts.append(inst.text)
            labeled_labels.append(ann.label.value)

    annotate(l_indices)

    curve_f1, curve_acc = LearningCurve(), LearningCurve()
    records: list[dict] = []
    fh = output_path.open("w", encoding="utf-8") if output_path else None

    def train_and_eval():
        clf = classifier_factory().fit(labeled_texts, labeled_labels)
        pred = clf.predict(test_texts)
        f1 = _macro_f1(pred, test_gold)
        acc = float(np.mean([p == g for p, g in zip(pred, test_gold)]))
        if not curve_f1.l_sizes or len(labeled_texts) > curve_f1.l_sizes[-1]:
            curve_f1.append(len(labeled_texts), f1)
            curve_acc.append(len(labeled_texts), acc)
        return f1, acc, clf

    f1, acc, clf = train_and_eval()
    iteration = 0
    _log(fh, records, iteration, strategy, seed, len(labeled_texts), f1, acc)

    while len(labeled_texts) + batch_size <= budget and unlabeled:
        iteration += 1
        k = min(batch_size, budget - len(labeled_texts), len(unlabeled))
        if strategy == "random":
            chosen_local = select_random(len(unlabeled), k, rng)
        else:
            probs = clf.predict_proba([pool[i].text for i in unlabeled])
            if strategy == "hybrid":
                k_unc = k - int(k * hybrid_random_fraction)
                top = select_top_k(entropy_scores(probs), k_unc)
                rest = np.setdiff1d(np.arange(len(unlabeled)), top)
                rnd = rng.choice(rest, size=min(k - k_unc, len(rest)), replace=False)
                chosen_local = np.concatenate([top, rnd])
            else:
                chosen_local = select_top_k(UNCERTAINTY[strategy](probs), k)
        chosen = [unlabeled[i] for i in chosen_local]
        unlabeled = [i for j, i in enumerate(unlabeled) if j not in set(chosen_local.tolist())]
        annotate(chosen)
        f1, acc, clf = train_and_eval()
        _log(fh, records, iteration, strategy, seed, len(labeled_texts), f1, acc)

    if fh:
        fh.close()

    baseline = baseline_performance if baseline_performance is not None else max(curve_f1.scores)
    value = lce(curve_f1, baseline_performance=baseline) if len(curve_f1) >= 2 else 0.0
    return ALResult(
        strategy=strategy,
        seed=seed,
        curve_macro_f1=curve_f1,
        curve_accuracy=curve_acc,
        lce_macro_f1=value,
        final_macro_f1=curve_f1.scores[-1],
        final_accuracy=curve_acc.scores[-1],
        n_labeled=len(labeled_texts),
        invalid_labels=invalid,
        records=records,
    )


def _log(fh, records, iteration, strategy, seed, n_labeled, f1, acc) -> None:
    record = {
        "iteration": iteration,
        "strategy": strategy,
        "seed": seed,
        "n_labeled": n_labeled,
        "macro_f1": round(f1, 4),
        "accuracy": round(acc, 4),
        "ts": time.time(),
    }
    records.append(record)
    if fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()
