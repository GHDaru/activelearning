# FlowBuilder — guia de uso (backend + frontend)

Interface web sobre a biblioteca: subir um CSV → base saneada persistida →
execuções de AL parametrizadas → curva de aprendizado na tela.

## Subir o backend (FastAPI)

```bash
cd activelearning
source .venv/bin/activate
pip install fastapi "uvicorn[standard]" sqlmodel python-multipart httpx
uvicorn activelearning.adapters.api.app:create_app --factory --reload --port 8000
```

Variáveis de ambiente (todas opcionais):

| Variável | Padrão | Efeito |
|---|---|---|
| `DATABASE_URL` | (vazio → SQLite local) | Postgres/Neon para Runs e Datasets |
| `FLOWBUILDER_CONFIG` | `experiments/e0/config.json` | config de amostras/oráculos usada pelos runs |
| `FLOWBUILDER_ARTIFACTS` | `experiments/api_runs` | onde os artefatos de execução são gravados |

Uploads ficam em `<artifacts>/../uploads/<dataset_id>/{original,sanitized}.csv`.

## Subir o frontend (React + Vite)

```bash
cd apps/web
npm install
npm run dev          # http://localhost:5173 (proxy para a API em :8000)
npm run build        # build de produção (tsc + vite)
```

## Fluxo pela UI

1. **Upload**: formulário com arquivo CSV, nome, coluna de texto, coluna de
   rótulo e rótulos operacionais (padrão `inativo`). Ao enviar, o saneamento
   roda na hora e o painel mostra o relatório (linhas mantidas, operacionais
   removidos, censo de conflitos/duplicatas). A base saneada pode ser baixada.
2. **Nova execução**: escolha o dataset, o tipo (`active-learning`), e os
   parâmetros — semente, orçamento, lote, tamanho do L0, estratégia, tamanho do
   pool, fração de teste — e o oráculo (simulado com ruído configurável, ou
   qualquer provedor LLM do formato E0).
3. **Acompanhar**: a lista de runs mostra estado (`queued/running/done/failed`);
   ao concluir, a curva de aprendizado (SVG) aparece com Macro-F1 por rótulos
   adquiridos e a LCE final.

## A mesma coisa via API (curl)

```bash
# 1) upload + saneamento
curl -F "file=@dataset.csv" -F "name=minha-base" \
     -F "text_column=nm_item" -F "label_column=nm_product" \
     -F "operational_labels=inativo" \
     http://localhost:8000/api/datasets
# → {"id": "ab12cd34ef56", ..., "report": {...}}

# 2) baixar a base saneada (ou a original)
curl -O "http://localhost:8000/api/datasets/ab12cd34ef56/download?which=sanitized"

# 3) disparar uma execução de AL parametrizada
curl -X POST http://localhost:8000/api/runs -H 'Content-Type: application/json' -d '{
  "name": "al-entropia-s42",
  "kind": "active-learning",
  "dataset_id": "ab12cd34ef56",
  "params": {"seed": 42, "budget": 1000, "batch_size": 50, "initial_size": 100,
              "strategy": "entropy", "pool_size": 5000, "test_fraction": 0.2,
              "min_per_class": 2},
  "oracle": {"provider": "simulated", "noise": 0.1}
}'

# 4) acompanhar
curl http://localhost:8000/api/runs            # lista
curl http://localhost:8000/api/runs/<run_id>   # estado + curva + LCE
```

Referência completa dos endpoints:

| Método/rota | O que faz |
|---|---|
| `GET /api/health` | ping |
| `GET /api/oracles` | provedores/modelos disponíveis (conforme chaves no ambiente) |
| `GET /api/samples` | amostras S-rand/S-strat da config ativa |
| `POST /api/datasets` | upload multipart + saneamento imediato |
| `GET /api/datasets` · `GET /api/datasets/{id}` | lista/detalhe (com relatório) |
| `GET /api/datasets/{id}/download?which=original\|sanitized` | download |
| `POST /api/runs` | cria e agenda execução (`oracle-eval` ou `active-learning`) |
| `GET /api/runs` · `GET /api/runs/{id}` | lista/estado/resultados |

Notas de validação: `active-learning` exige `dataset_id` (422 sem ele; 404 se o
dataset não existe); estratégia deve ser uma de
`entropy|least_confidence|smallest_margin|random|hybrid`.

## Oráculo LLM gratuito no FlowBuilder

Para o teste ponta a ponta com oráculo real de custo zero, use o provedor
`nvidia` (chave `NVIDIA_API_KEY` no `.env`; ver `docs/biblioteca.md`) ou
`openrouter` com um modelo `:free` — atenção à cota de 50 requisições/dia do
OpenRouter sem créditos (decisões D-005/D-006 no repositório da tese).
