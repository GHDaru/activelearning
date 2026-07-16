"""OraclePort: contrato para fontes de rótulos (LLMs, simulado, humano)."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..domain.annotation import Annotation
from ..domain.instances import CategorySchema, Instance


@runtime_checkable
class OraclePort(Protocol):
    """Dado um batch de instâncias e o schema fechado de categorias, anota.

    Contratos (Princípios III e IV da constituição):
    - a saída do LLM DEVE ser restrita ao ``schema`` (enum em saída estruturada
      quando o provedor suporta; validação exata caso contrário);
    - resposta fora do schema vira ``Annotation(label=None, raw_response=...)``;
    - cada Annotation DEVE carregar ``OracleUsage`` preenchido.
    """

    oracle_id: str
    prompt_version: str

    def annotate(self, batch: list[Instance], schema: CategorySchema) -> list[Annotation]:
        """Anota o batch, preservando a ordem e a correspondência 1:1."""
        ...
