# Spec 002 — FlowBuilder: interface web para fluxos de AL

## Objetivo
Automatizar via interface a execução e o acompanhamento dos fluxos experimentais
da tese (v0: avaliação de oráculo estilo E0), sem abrir mão da rastreabilidade
(Princípio I): todo run gera artefatos JSONL retomáveis em disco; o banco guarda
apenas o índice e o report.

## Arquitetura
- **Backend** (`src/activelearning/adapters/api/`): FastAPI como *driving adapter*
  do hexágono — importa o domínio e o caso de uso `EvaluateOracle` diretamente;
  nenhuma regra de negócio na API.
- **Persistência** (`src/activelearning/adapters/persistence/`): SQLAlchemy 2.
  `DATABASE_URL` ausente → SQLite local (`flowbuilder.db`); Neon/Postgres em
  produção (`postgresql://…` no `.env`, nunca commitado, chaves rotacionáveis).
- **Frontend** (`apps/web/`): React + Vite + TypeScript (pnpm). Dev server proxya
  `/api` → `localhost:8000`.
- **Dados**: mesma config JSON do E0 (`experiments/e0/config.json`) — IDs
  `e0-{índice}` estáveis entre CLI e API (implementação única em
  `adapters/datasets/retail_csv.py`; fábrica de oráculos em
  `adapters/oracles/factory.py`).

## Endpoints v0
| Método | Rota | Descrição |
|---|---|---|
| GET | /api/health | status + backend de banco |
| GET | /api/oracles | specs disponíveis (simulado + config E0; sem segredos) |
| GET | /api/samples | amostras (rand/strat) e tamanhos |
| POST | /api/runs | cria run {name, sample, limit, oracle} e executa em background |
| GET | /api/runs | lista runs |
| GET | /api/runs/{id} | detalhe com report + caminho dos artefatos |

## Execução
```bash
# backend (raiz do repo)
uv sync --extra api && uv run uvicorn --factory activelearning.adapters.api.app:create_app --reload --port 8000
# frontend
cd apps/web && pnpm install && pnpm dev   # http://localhost:5173
```

## Critérios de aceite v0 (cobertos em tests/unit/test_api.py)
1. Health responde com o backend de banco correto.
2. /api/oracles inclui o simulado e não vaza chaves.
3. Run com oráculo simulado (ε=0) completa com accuracy 1.0 e artifacts_dir preenchido.
4. Run inexistente → 404.

## Evolução prevista
- v1: fluxos de AL completos (RunActiveLearning: estratégia × iterações × LCE),
  curvas de aprendizado no frontend.
- v2: comparação de runs (McNemar/Wilson server-side), export para a tese.
- Auth simples (token) antes de qualquer deploy fora de localhost.
