"""Caso de uso: FALCO — aprendizado ativo em fases com oráculo LLM progressivo.

Fase 1 (cold start informado): b0 = ceil(0,01·B) instâncias selecionadas pelo
DRI-SL e rotuladas pelo oráculo Inicial.
Fase 2 (incerteza + oráculo Inicial): lotes por entropia preditiva; transição
quando o Macro F1 no conjunto de VALIDAÇÃO estagna por p iterações (tolerância
eps). O conjunto de teste jamais participa de decisões (correção A1 da R1).
Fase 3 (refinamento): mesma seleção, oráculo Avançado — existe apenas se um
oráculo avançado for configurado (gate do E0).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from activelearning.domain.instances import CategorySchema, Instance
from activelearning.domain.metrics import LearningCurve, lce
from activelearning.domain.strategies import entropy_scores, select_top_k

from .run_active_learning import OraclePort, TaskClassifier, _macro_f1


@dataclass
class FalcoResult:
    """Resultado do runner FALCO completo (cold start → fases → parada).

    Separa a métrica de **relato** (``curve_macro_f1``, no teste) da de
    **decisão** (``curve_val_macro_f1``, na validação — o que dispara a parada);
    ``phase_boundaries`` marca quantos rótulos ao fim de cada fase e
    ``oracle_calls`` conta as chamadas por oráculo (Inicial/Avançado).
    """

    seed: int
    curve_macro_f1: LearningCurve          # medida no TESTE (relato)
    curve_val_macro_f1: list[tuple[int, float]]  # medida na VALIDAÇÃO (decisão)
    phase_boundaries: dict[str, int]       # n_rotulados no fim de cada fase
    lce_macro_f1: float
    final_macro_f1: float
    n_labeled: int
    invalid_labels: int
    oracle_calls: dict[str, int]
    records: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        """Resumo serializável (JSON) do ciclo FALCO."""
        return {
            "seed": self.seed,
            "phase_boundaries": self.phase_boundaries,
            "lce_macro_f1": round(self.lce_macro_f1, 4),
            "final_macro_f1": round(self.final_macro_f1, 4),
            "n_labeled": self.n_labeled,
            "invalid_labels": self.invalid_labels,
            "oracle_calls": self.oracle_calls,
        }


def run_falco(
    pool: list[Instance],
    validation: list[Instance],
    test: list[Instance],
    schema: CategorySchema,
    classifier_factory: Callable[[], TaskClassifier],
    oracle_initial: OraclePort,
    oracle_advanced: OraclePort | None = None,
    drisl_selector: Callable[[Sequence[str], int], list[int]] | None = None,
    budget: int = 1000,
    batch_size: int = 100,
    seed: int = 42,
    stagnation_patience: int = 5,
    stagnation_eps: float = 1e-3,
    baseline_performance: float | None = None,
    output_path: Path | None = None,
) -> FalcoResult:
    """Executa o ciclo FALCO completo sobre o pool (cold start → fases → parada).

    Combina o cold start sem rótulos (``drisl_selector``), a seleção por
    incerteza e um oráculo progressivo (``oracle_initial`` → ``oracle_advanced``),
    parando por estagnação da **validação** (``stagnation_patience``/``_eps``). A
    decisão de parada usa a validação; o relato usa o ``test``. Devolve um
    :class:`FalcoResult`.
    """
    if budget > len(pool):
        raise ValueError("orçamento maior que o pool.")
    rng = np.random.default_rng(seed)

    test_texts = [i.text for i in test]
    test_gold = [i.gold_label.value for i in test]
    val_texts = [i.text for i in validation]
    val_gold = [i.gold_label.value for i in validation]

    unlabeled = list(range(len(pool)))
    labeled_texts: list[str] = []
    labeled_labels: list[str] = []
    invalid = 0
    calls = {"initial": 0, "advanced": 0}

    def annotate(indices: list[int], oracle: OraclePort, tag: str) -> None:
        nonlocal invalid
        batch = [pool[i] for i in indices]
        calls[tag] += len(batch)
        for inst, ann in zip(batch, oracle.annotate(batch, schema)):
            if ann.label is None:
                invalid += 1
                continue
            labeled_texts.append(inst.text)
            labeled_labels.append(ann.label.value)

    # ---------------- Fase 1: DRI-SL + oráculo Inicial -----------------------
    b0 = max(1, math.ceil(0.01 * budget))
    if drisl_selector is not None:
        l0 = list(drisl_selector([pool[i].text for i in unlabeled], b0))
        l0 = [unlabeled[i] for i in l0]
    else:  # fallback: aleatório com semente (ablation "FALCO sem DRI-SL")
        pick = rng.choice(len(unlabeled), size=b0, replace=False)
        l0 = [unlabeled[i] for i in pick]
    unlabeled = [i for i in unlabeled if i not in set(l0)]
    annotate(l0, oracle_initial, "initial")

    curve_test = LearningCurve()
    curve_val: list[tuple[int, float]] = []
    records: list[dict] = []
    fh = output_path.open("w", encoding="utf-8") if output_path else None
    boundaries = {"fase1": len(labeled_texts)}

    def train_eval(phase: str):
        clf = classifier_factory().fit(labeled_texts, labeled_labels)
        test_f1 = _macro_f1(clf.predict(test_texts), test_gold)
        val_f1 = _macro_f1(clf.predict(val_texts), val_gold)
        if not curve_test.l_sizes or len(labeled_texts) > curve_test.l_sizes[-1]:
            curve_test.append(len(labeled_texts), test_f1)
            curve_val.append((len(labeled_texts), val_f1))
        record = {
            "phase": phase, "n_labeled": len(labeled_texts),
            "test_macro_f1": round(test_f1, 4), "val_macro_f1": round(val_f1, 4),
            "ts": time.time(),
        }
        records.append(record)
        if fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
        return clf, val_f1

    clf, val_f1 = train_eval("fase1")

    # ---------------- Fase 2: entropia + oráculo Inicial ---------------------
    best_val, stagnant = val_f1, 0
    while len(labeled_texts) + batch_size <= budget and unlabeled and stagnant < stagnation_patience:
        k = min(batch_size, budget - len(labeled_texts), len(unlabeled))
        probs = clf.predict_proba([pool[i].text for i in unlabeled])
        chosen_local = select_top_k(entropy_scores(probs), k)
        chosen = [unlabeled[i] for i in chosen_local]
        unlabeled = [i for j, i in enumerate(unlabeled) if j not in set(chosen_local.tolist())]
        annotate(chosen, oracle_initial, "initial")
        clf, val_f1 = train_eval("fase2")
        if val_f1 > best_val + stagnation_eps:
            best_val, stagnant = val_f1, 0
        else:
            stagnant += 1
    boundaries["fase2"] = len(labeled_texts)

    # ---------------- Fase 3: entropia + oráculo Avançado --------------------
    if oracle_advanced is not None:
        while len(labeled_texts) + batch_size <= budget and unlabeled:
            k = min(batch_size, budget - len(labeled_texts), len(unlabeled))
            probs = clf.predict_proba([pool[i].text for i in unlabeled])
            chosen_local = select_top_k(entropy_scores(probs), k)
            chosen = [unlabeled[i] for i in chosen_local]
            unlabeled = [i for j, i in enumerate(unlabeled) if j not in set(chosen_local.tolist())]
            annotate(chosen, oracle_advanced, "advanced")
            clf, _ = train_eval("fase3")
    boundaries["fase3"] = len(labeled_texts)

    if fh:
        fh.close()

    baseline = baseline_performance if baseline_performance is not None else max(curve_test.scores)
    value = lce(curve_test, baseline_performance=baseline) if len(curve_test) >= 2 else 0.0
    return FalcoResult(
        seed=seed,
        curve_macro_f1=curve_test,
        curve_val_macro_f1=curve_val,
        phase_boundaries=boundaries,
        lce_macro_f1=value,
        final_macro_f1=curve_test.scores[-1],
        n_labeled=len(labeled_texts),
        invalid_labels=invalid,
        oracle_calls=calls,
        records=records,
    )
