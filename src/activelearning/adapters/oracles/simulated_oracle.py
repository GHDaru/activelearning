"""Oráculo simulado: rótulos-ouro com ruído paramétrico ε (E1: ε=0; E4: ε>0)."""
from __future__ import annotations

import numpy as np

from ...domain.annotation import Annotation, OracleUsage
from ...domain.instances import CategorySchema, Instance


class SimulatedOracle:
    """Devolve o gold_label com probabilidade 1-ε; com ε, um rótulo errado uniforme."""

    def __init__(self, noise: float = 0.0, seed: int = 42) -> None:
        if not 0.0 <= noise < 1.0:
            raise ValueError("noise deve estar em [0, 1).")
        self._noise = noise
        self._rng = np.random.default_rng(seed)
        self.oracle_id = f"simulated:noise={noise}"
        self.prompt_version = "n/a"

    def annotate(self, batch: list[Instance], schema: CategorySchema) -> list[Annotation]:
        annotations: list[Annotation] = []
        values = schema.values
        for instance in batch:
            if instance.gold_label is None:
                raise ValueError(
                    f"SimulatedOracle requer gold_label (instance {instance.id})."
                )
            label = instance.gold_label
            if self._noise > 0 and self._rng.random() < self._noise:
                wrong = [v for v in values if v != label.value]
                label = schema.validate(str(self._rng.choice(wrong)))
            annotations.append(
                Annotation(
                    instance_id=instance.id,
                    label=label,
                    oracle_id=self.oracle_id,
                    prompt_version=self.prompt_version,
                    usage=OracleUsage(),
                )
            )
        return annotations
