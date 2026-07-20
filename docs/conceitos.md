# Conceitos

O domínio da biblioteca é **puro** (sem I/O) e pequeno. Entender estes tipos é
suficiente para usar todo o resto.

## Instance

Um texto candidato a rotulagem. Em simulação, carrega o rótulo-ouro
(`gold_label`) para permitir medir sem anotação humana.

```python
Instance(id="1", text="CERV BRAHMA LT 350ML", gold_label=Label("cerveja"))
```

`id` e `text` são obrigatórios; `text` não pode ser vazio.

## Label

Rótulo de classe **normalizado** (minúsculas, sem acento, sem espaços extras). A
normalização acontece na construção — dois rótulos "equivalentes" viram o mesmo
`Label`.

## CategorySchema

O conjunto **fechado** de rótulos válidos da tarefa. É a fonte única do `enum`
enviado aos oráculos LLM: qualquer resposta fora dele é uma anotação inválida,
contabilizada explicitamente.

```python
schema = CategorySchema.from_raw(["cerveja", "sabao", "refrigerante"])
schema.values        # ('_rare_', 'cerveja', 'refrigerante', 'sabao')
schema.validate("Cerveja")   # Label('cerveja')  (normaliza)
schema.validate("vinho")     # None              (fora do esquema)
```

Por padrão o esquema inclui um rótulo `_rare_` para a cauda longa
(`include_rare=True`).

## Annotation

O resultado de um oráculo para uma `Instance`: um `Label` válido **ou** `None`
(resposta inválida/fora do esquema). Nunca se descarta uma resposta em silêncio —
a invalidade é medida.

## Estratégias de seleção

Quais instâncias vale a pena rotular. Disponíveis via o parâmetro `strategy`:

| Estratégia | Ideia |
|------------|-------|
| `entropy` | maior entropia da distribuição de classes (incerteza) — o padrão validado |
| `least_confident` | menor probabilidade máxima |
| `smallest_margin` | menor diferença entre as duas classes mais prováveis |
| `random` | amostragem aleatória (linha de base) |
| `hybrid` | mistura incerteza + aleatório (`hybrid_random_fraction`) |

## Métrica LCE

**Label-Cost Efficiency** — resume a *curva de aprendizado* (desempenho × rótulos)
num único número: quanto desempenho cada rótulo comprou. É o instrumento central
para comparar estratégias com orçamentos diferentes.

## Ports e Adapters

O domínio conversa com o mundo por **contratos** (`ports`), implementados por
**adapters** intercambiáveis:

- `OraclePort.annotate(batch, schema)` → simulado, OpenAI, Gemini, Ollama, …
- `ClassifierPort` → PVBin (protótipo TF-IDF), SGD, BERTimbau.
- `EmbedderPort` → para o DRI-SL (cold start).

Trocar um LLM por outro, ou o classificador leve pelo forte, é trocar o adapter —
o laço de AA não muda. Veja a [arquitetura](architecture.md).
