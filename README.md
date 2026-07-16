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

### Credenciais (`.env`)

Crie um arquivo `.env` na raiz do projeto (já ignorado pelo git — **nunca** commitá-lo):

```
OPENAI_API_KEY=sk-proj-...
GEMINI_API_KEY=...
```

O runner do E0 carrega o `.env` automaticamente. Providers sem chave são pulados.
O dataset já está versionado em `data/dataset.csv` (250.365 instâncias).

### Windows / PowerShell

```powershell
git clone https://github.com/GHDaru/activelearning
cd activelearning
uv sync --all-extras
notepad .env        # cole as chaves no formato acima e salve
uv run python experiments/e0/run_e0.py --config experiments/e0/config.json
```

(Para Ollama local: instale o Ollama para Windows, `ollama pull gemma3` e
`ollama pull qwen2.5` antes de rodar.)

### Prompt caching (OpenAI)

O adapter OpenAI é desenhado para o prompt caching automático da API: o prefixo
estático (system prompt + schema com enum de 622 categorias, ~2,9k tokens) é idêntico
em todas as chamadas e fica cacheado; só a descrição do produto varia, ao final.
Um `prompt_cache_key` estável (hash do prefixo) melhora o roteamento. Tokens cacheados
custam 50% do preço de input e são reportados no relatório (`cached_input_tokens`,
`cache_hit_rate`). Medição real (gpt-4o-mini, N=8): ~94% do input de cada chamada
servido do cache a partir da 2ª chamada; custo ≈ US$ 0,29 por 1.000 rótulos.

## Regra de ouro

**Nenhum número entra na tese sem um artefato rastreável aqui** (config + git SHA +
seeds + JSONL de execução). Legado (`activetextclassification`, `FlowBuilder`) é
somente leitura.
