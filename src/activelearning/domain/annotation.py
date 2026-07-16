"""Anotações produzidas por oráculos, com proveniência, validade e custo."""
from __future__ import annotations

from dataclasses import dataclass

from .instances import Label


@dataclass(frozen=True, slots=True)
class OracleUsage:
    """Observabilidade de uma chamada de oráculo (Princípio IV da constituição).

    ``cached_input_tokens`` é o subconjunto de ``input_tokens`` servido do cache de
    prompt do provedor (custo reduzido) — no nosso desenho, o prefixo estático
    (system prompt + schema com enum) é idêntico entre chamadas, então quase todo
    o input é cacheável a partir da segunda chamada.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    cost_usd: float = 0.0
    cached_input_tokens: int = 0

    @property
    def cache_hit_rate(self) -> float:
        return self.cached_input_tokens / self.input_tokens if self.input_tokens else 0.0

    def __add__(self, other: "OracleUsage") -> "OracleUsage":
        return OracleUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            latency_seconds=self.latency_seconds + other.latency_seconds,
            cost_usd=self.cost_usd + other.cost_usd,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
        )


@dataclass(frozen=True, slots=True)
class Annotation:
    """Rótulo atribuído a uma Instance por um oráculo.

    ``label`` é None quando o oráculo respondeu fora do CategorySchema
    (``raw_response`` preserva o que veio); ``is_valid_label`` distingue
    explicitamente esse caso — ele conta como erro de oráculo, nunca some.
    """

    instance_id: str
    label: Label | None
    oracle_id: str
    prompt_version: str
    raw_response: str = ""
    rationale: str = ""
    usage: OracleUsage = OracleUsage()

    @property
    def is_valid_label(self) -> bool:
        return self.label is not None

    def is_correct(self, gold: Label | None) -> bool | None:
        """Correção vs. rótulo-ouro. None quando não há ouro (produção)."""
        if gold is None:
            return None
        return self.label is not None and self.label == gold
