"""Instâncias, rótulos e o schema fechado de categorias (fonte única do enum)."""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field


def normalize_label(raw: str) -> str:
    """Normalização canônica de rótulos: minúsculas, sem acentos, espaços colapsados."""
    text = unicodedata.normalize("NFKD", raw)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(text.lower().split())


@dataclass(frozen=True, slots=True)
class Label:
    """Rótulo de classe já normalizado."""

    value: str

    def __post_init__(self) -> None:
        normalized = normalize_label(self.value)
        if not normalized:
            raise ValueError("Label não pode ser vazio.")
        object.__setattr__(self, "value", normalized)

    def __str__(self) -> str:  # pragma: no cover - conveniência
        return self.value


RARE_LABEL = Label("_rare_")


@dataclass(frozen=True, slots=True)
class Instance:
    """Um texto candidato a rotulagem; em simulação carrega o rótulo-ouro."""

    id: str
    text: str
    gold_label: Label | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Instance.id é obrigatório.")
        if not self.text or not self.text.strip():
            raise ValueError("Instance.text não pode ser vazio.")


@dataclass(frozen=True)
class CategorySchema:
    """Conjunto FECHADO de rótulos válidos da tarefa.

    É a fonte única do ``enum`` enviado aos oráculos LLM (Princípio III da
    constituição). Qualquer resposta fora deste conjunto é uma anotação inválida,
    contabilizada explicitamente — nunca aceita nem descartada em silêncio.
    """

    labels: frozenset[Label]
    include_rare: bool = True
    _sorted: tuple[str, ...] = field(init=False, repr=False, compare=False, default=())

    def __post_init__(self) -> None:
        if not self.labels:
            raise ValueError("CategorySchema requer ao menos um rótulo.")
        values = {label.value for label in self.labels}
        if self.include_rare:
            values.add(RARE_LABEL.value)
        object.__setattr__(self, "_sorted", tuple(sorted(values)))

    @classmethod
    def from_raw(cls, raw_labels: list[str] | set[str], include_rare: bool = True) -> CategorySchema:
        return cls(labels=frozenset(Label(v) for v in raw_labels), include_rare=include_rare)

    @property
    def values(self) -> tuple[str, ...]:
        return self._sorted

    def __len__(self) -> int:
        return len(self._sorted)

    def __contains__(self, item: object) -> bool:
        if isinstance(item, Label):
            return item.value in self._sorted
        if isinstance(item, str):
            return normalize_label(item) in self._sorted
        return False

    def validate(self, raw: str) -> Label | None:
        """Retorna o Label normalizado se pertencer ao schema; None caso contrário."""
        normalized = normalize_label(raw)
        return Label(normalized) if normalized in self._sorted else None

    def to_json_schema(self, constrained: bool = True) -> dict:
        """Schema JSON de saída estruturada.

        Com ``constrained=True`` (padrão e único modo permitido em produção pela
        constituição, Princípio III), o rótulo é restrito por ``enum`` — correção do
        defeito do legado ``activetextclassification/prompts.py``, cujo
        ``predicted_category`` era string livre e contaminava a medição de acurácia
        do oráculo com variações de fraseado.

        ``constrained=False`` existe EXCLUSIVAMENTE para o sub-experimento RQ4 do E0
        (efeito do instrumento): reproduz o instrumento do legado para quantificar a
        deflação de acurácia causada pela ausência do enum.
        """
        category_property: dict = {
            "type": "string",
            "description": "A categoria escolhida, exatamente uma da lista.",
        }
        if constrained:
            category_property["enum"] = list(self._sorted)
        # strict=True (OpenAI structured outputs) IMPÕE o schema na decodificação;
        # sem ele a API trata o schema como orientação e o modelo pode desviar do
        # enum (observado no piloto: ~4% de rótulos fora do schema em lote).
        # O modo strict exige todas as propriedades em `required`.
        return {
            "name": "oracle_classification",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "expanded_description": {
                        "type": "string",
                        "description": (
                            "Descrição com abreviações expandidas "
                            "(ex.: 'cv br lt 350' -> 'cerveja brahma lata 350 ml')."
                        ),
                    },
                    "predicted_category": category_property,
                    "rationale": {
                        "type": "string",
                        "description": "Justificativa breve da escolha.",
                    },
                },
                "required": ["expanded_description", "predicted_category", "rationale"],
                "additionalProperties": False,
            },
        }
