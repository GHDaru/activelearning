# activelearning

Biblioteca de **Aprendizado Ativo para classificação de texto com oráculos LLM** —
motor experimental da tese de doutorado (framework **FALCO**) e futuro núcleo do
FlowBuilder.

> **Instalação (PyPI):** `pip install falco-active-learning`
> (o nome de importação continua `import activelearning`). Extras opcionais:
> `pip install "falco-active-learning[classifiers,oracles,api]"`.
> 📖 **Documentação:** <https://ghdaru.github.io/activelearning/> ·
> Deploy e publicações: [DEPLOY_AND_PUBLISH.md](DEPLOY_AND_PUBLISH.md).

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

## Documentação

| Guia | Conteúdo |
|------|----------|
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | **reprodutibilidade para outros pesquisadores**: ambiente, dados, comando exato de cada experimento, artefatos, custos e roteiro de publicação (Zenodo/Papers with Code) |
| [docs/biblioteca.md](docs/biblioteca.md) | uso da biblioteca: classificadores (PVBin, BERTimbau), DRI-SL, oráculos, laço de AL, runner FALCO, saneamento |
| [docs/flowbuilder.md](docs/flowbuilder.md) | frontend + backend: subir os serviços, fluxo pela UI, referência da API (curl) |
| [docs/experimentos.md](docs/experimentos.md) | como executar e parametrizar E0/E0-P/E1/E4/P1/E2-E3; artefatos e reprodutibilidade |
| [docs/avaliacao-e-graficos.md](docs/avaliacao-e-graficos.md) | rotinas de análise estatística e geração de figuras (paleta validada) |
| [docs/bertimbau-local-passo-a-passo.md](docs/bertimbau-local-passo-a-passo.md) | BERTimbau na RTX 3090 local, passo a passo |
| [docs/architecture.md](docs/architecture.md) | arquitetura DDD + Hexagonal |

## Experimentos (estado)

| ID | O quê | Estado |
|----|-------|--------|
| E0 | Avaliação de oráculos LLM (enum/json-prompt/free) | completo (pagos) + free via NVIDIA NIM em execução |
| E0-P | Ablação de prompt no modelo fraco | completo |
| E1/E1b | Estratégias × lote (PVBin + oráculo simulado, 8 sementes) | completo |
| E4 | Robustez a ruído do oráculo (ε ∈ {0,1; 0,2; 0,4}) | completo (PVBin) |
| P1/P2 | Replays de sensibilidade de L0 e do AG (anticircularidade) | completo |
| E2/E3 | BERTimbau: épocas×|L| e FALCO integrado | instrumental pronto; execução na GPU (bloco H) |

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
