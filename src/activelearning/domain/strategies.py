"""Estratégias de consulta: funções puras sobre distribuições de probabilidade.

Portado de ``activetextclassification/selection.py`` (legado), reescrito como
serviços de domínio sem dependências além de numpy.
"""
from __future__ import annotations

import numpy as np


def _validate(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2:
        raise ValueError("probabilities deve ser matriz (n_instancias, n_classes).")
    return probs


def entropy_scores(probabilities: np.ndarray) -> np.ndarray:
    """Entropia de Shannon por instância (maior = mais incerto)."""
    probs = _validate(probabilities)
    safe = np.clip(probs, 1e-12, 1.0)
    return -np.sum(safe * np.log2(safe), axis=1)


def least_confidence_scores(probabilities: np.ndarray) -> np.ndarray:
    """1 - max(p) por instância (maior = mais incerto)."""
    probs = _validate(probabilities)
    return 1.0 - probs.max(axis=1)


def smallest_margin_scores(probabilities: np.ndarray) -> np.ndarray:
    """Margem negativa entre as duas classes mais prováveis (maior = mais incerto)."""
    probs = _validate(probabilities)
    if probs.shape[1] < 2:
        return np.zeros(probs.shape[0])
    top2 = np.partition(probs, -2, axis=1)[:, -2:]
    return -(top2[:, 1] - top2[:, 0])


def select_top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Índices das k instâncias com maior score, ordenados por score decrescente."""
    if k <= 0:
        raise ValueError("k deve ser positivo.")
    k = min(k, scores.shape[0])
    idx = np.argpartition(scores, -k)[-k:]
    return idx[np.argsort(scores[idx])[::-1]]


def select_random(n_instances: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Seleção aleatória sem reposição (baseline RS)."""
    k = min(k, n_instances)
    return rng.choice(n_instances, size=k, replace=False)


def select_hybrid(
    scores: np.ndarray, k: int, entropy_fraction: float, rng: np.random.Generator
) -> np.ndarray:
    """Mistura: fração por incerteza (top score), restante aleatório do que sobrou."""
    if not 0.0 <= entropy_fraction <= 1.0:
        raise ValueError("entropy_fraction deve estar em [0, 1].")
    k = min(k, scores.shape[0])
    k_ent = round(k * entropy_fraction)
    chosen = list(select_top_k(scores, k_ent)) if k_ent else []
    remaining = np.setdiff1d(np.arange(scores.shape[0]), chosen)
    k_rnd = k - len(chosen)
    if k_rnd > 0 and remaining.size:
        chosen.extend(rng.choice(remaining, size=min(k_rnd, remaining.size), replace=False))
    return np.asarray(chosen, dtype=int)
