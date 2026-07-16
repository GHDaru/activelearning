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


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Intervalo de confiança de Wilson para uma proporção (acurácia)."""
    if total <= 0:
        raise ValueError("total deve ser positivo.")
    if not 0 <= successes <= total:
        raise ValueError("successes deve estar em [0, total].")
    from scipy.stats import norm

    z = float(norm.ppf(1 - (1 - confidence) / 2))
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_test(b: int, c: int) -> float:
    """Teste de McNemar para amostras pareadas (χ² com correção de continuidade).

    ``b`` = casos em que A acertou e B errou; ``c`` = A errou e B acertou.
    Com poucos discordantes (b+c < 25) usa o teste binomial exato.
    Retorna o p-valor bicaudal.
    """
    if b < 0 or c < 0:
        raise ValueError("b e c devem ser não negativos.")
    n = b + c
    if n == 0:
        return 1.0
    from scipy.stats import binomtest, chi2

    if n < 25:
        return float(binomtest(min(b, c), n=n, p=0.5).pvalue)
    stat = (abs(b - c) - 1) ** 2 / n
    return float(chi2.sf(stat, df=1))
