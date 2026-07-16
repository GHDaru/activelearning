"""Métricas de domínio: curva de aprendizado e LCE (definição única, v5 da tese)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LearningCurve:
    """Série (|L_t|, métrica) de um ActiveLearningRun."""

    l_sizes: list[int] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)

    def append(self, l_size: int, score: float) -> None:
        if self.l_sizes and l_size <= self.l_sizes[-1]:
            raise ValueError("l_size deve ser estritamente crescente.")
        self.l_sizes.append(l_size)
        self.scores.append(score)

    def __len__(self) -> int:
        return len(self.l_sizes)


def lce(
    curve: LearningCurve,
    baseline_performance: float,
    l_start_ideal: float = 0.0,
) -> float:
    """Learning Curve Efficiency = AUC_real / AUC_ideal.

    AUC_real: integral da curva pelos pontos medidos (Regra de Simpson; degenera
    para trapézio com apenas 2 pontos). AUC_ideal: baseline_performance mantido
    constante de ``l_start_ideal`` (default 0) até o último ponto medido.
    Definição única da tese (apêndice LCE, aproximação parabólica).
    """
    if baseline_performance <= 0:
        raise ValueError("baseline_performance deve ser > 0.")
    if len(curve) < 2:
        raise ValueError("LCE requer ao menos 2 pontos na curva.")

    x = np.asarray(curve.l_sizes, dtype=float)
    y = np.asarray(curve.scores, dtype=float)
    if np.any(~np.isfinite(y)):
        raise ValueError("Curva contém valores não finitos.")

    delta_ideal = float(x[-1]) - l_start_ideal
    if delta_ideal <= 0:
        raise ValueError("Intervalo ideal inválido (l_final <= l_start_ideal).")

    if len(curve) == 2:
        auc_real = float(np.trapezoid(y, x))
    else:
        from scipy.integrate import simpson

        auc_real = float(simpson(y=y, x=x))

    return auc_real / (baseline_performance * delta_ideal)
