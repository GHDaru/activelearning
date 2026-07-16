# activelearning

Biblioteca de **Aprendizado Ativo para classificação de texto com oráculos LLM** —
motor experimental da tese de doutorado (framework **FALCO**) e futuro núcleo do
FlowBuilder.

Arquitetura **DDD + Hexagonal**: domínio puro, capacidades externas via ports/adapters.
Governança por spec-kit: veja [.specify/memory/constitution.md](.specify/memory/constitution.md)
e [docs/architecture.md](docs/architecture.md).

## Estrutura

```
src/activelearning/
├── domain/        # puro: Instance, Label, CategorySchema, Annotation, Budget,
│                  # strategies (entropia/margem/...), metrics (LCE)
├── ports/         # OraclePort, ClassifierPort, EmbedderPort, ...
├── adapters/
│   └── oracles/   # OpenAI, Gemini, Ollama, Simulated — TODOS com saída
│                  # estruturada restrita por enum (CategorySchema)
└── application/   # use cases: EvaluateOracle (E0), RunActiveLearning, RunFalco
experiments/
└── e0/            # E0: avaliação de oráculos (runner + config)
specs/             # especificações spec-kit por feature
```

## Setup

```bash
uv sync --all-extras          # ambiente completo
uv run pytest                 # testes de domínio (sem rede/GPU/chaves)
```

## Experimentos

| ID | O quê | Como rodar |
|----|-------|------------|
| E0 | Avaliação de oráculos LLM (schema com enum) | `uv run python experiments/e0/run_e0.py --config experiments/e0/config.json` |
| E1 | Estratégias × batch size (PVBin + oráculo simulado) | (próximo) |
| E2 | Épocas de fine-tuning do BERTimbau | (próximo) |
| E3 | FALCO vs RS vs US (BERTimbau + oráculo LLM) | (alvo da defesa) |
| E4 | Robustez a ruído do oráculo | (condicional a E0) |

E0 requer: `OPENAI_API_KEY` / `GEMINI_API_KEY` no ambiente (ou `.env`) e/ou Ollama
local com os modelos baixados; e o dataset em `data/dataset.csv` (colunas `nm_item`,
`nm_product` — o mesmo do repositório legado `activetextclassification`).

## Regra de ouro

**Nenhum número entra na tese sem um artefato rastreável aqui** (config + git SHA +
seeds + JSONL de execução). Legado (`activetextclassification`, `FlowBuilder`) é
somente leitura.
