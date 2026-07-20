# Instalação

## Requisitos

- Python **≥ 3.11**
- O núcleo depende apenas de `numpy` e `scipy` (sem rede, GPU ou chaves).

## Núcleo

```bash
pip install falco-active-learning
```

O nome de **importação** continua `activelearning`:

```python
import activelearning
from activelearning.domain.instances import Instance, Label, CategorySchema
```

## Extras opcionais

Instale só o que precisar:

```bash
pip install "falco-active-learning[classifiers]"   # scikit-learn, torch, transformers, sentence-transformers
pip install "falco-active-learning[oracles]"       # openai, google-genai, ollama
pip install "falco-active-learning[api]"           # FastAPI + backend do FlowBuilder
pip install "falco-active-learning[classifiers,oracles,api]"   # tudo
```

| Extra | Habilita |
|-------|----------|
| `classifiers` | classificadores PVBin/SGD/BERTimbau e *embeddings* |
| `oracles` | oráculos LLM reais (OpenAI, Gemini, Ollama, compatíveis) |
| `api` | o backend FastAPI (FlowBuilder) e a geração de fichamento |
| `dev` | pytest, ruff, httpx |
| `docs` | mkdocs-material + mkdocstrings (esta documentação) |

!!! note "O oráculo simulado é offline"
    Você pode rodar todo o laço de AA **sem chaves** usando `SimulatedOracle`
    (rótulo-ouro com ruído ε controlado). Ideal para calibrar orçamento e lote
    antes de gastar com um LLM real.

## Desenvolvimento (a partir do código)

```bash
git clone https://github.com/GHDaru/activelearning
cd activelearning
uv sync --all-extras     # ou: pip install -e ".[classifiers,oracles,api,dev,docs]"
uv run pytest            # testes de domínio (sem rede/GPU/chaves)
```
