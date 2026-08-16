# Plan 004 — Site da tese: fundação (backend hospedável + front configurável)

Lane: **infra** (spec completa + reversibilidade — toca deploy e uma rota que
escreve fora deste repositório).

## Constitution Check (Maestro Principles I–VIII)

| # | Princípio | Veredito | Por quê |
|---|---|---|---|
| I | Spec-driven | ✅ | Este plano deriva de `specs/004-site-da-tese/spec.md`, escrita antes do código, com as três decisões do autor (16/08/2026) registradas como origem. |
| II | Human-governed orchestration | ✅ | O autor decidiu as três perguntas de arquitetura (onde mora o site, alcance da execução, hospedagem) via `AskUserQuestion`; eu executo dentro dessas balizas. Nenhuma decisão de escopo foi tomada sem ele. |
| III | Reversibility / risk-proportional gates | ⚠️→rework | Publicar um backend real na internet pública é ação de blast radius alto (escreve no repositório `tesedaru` via `/api/fichamentos`, executa `experiments/*/run_*.py` sob demanda, consome as chaves de oráculo do autor). Rework aplicado: (a) gate `X-Admin-Token` na rota que escreve fora deste repositório — fica fechada por padrão assim que o host define a variável; (b) catálogo público usa oráculo `simulated` (sem custo/chave) — habilitar oráculo real fica para R2, depois de existir teto por visitante; (c) nada neste ciclo dispara deploy sozinho — `render.yaml` e `Dockerfile` são material para o autor aplicar manualmente (marcado **[VOCÊ]** em `DEPLOY_AND_PUBLISH.md`), preservando o gate humano antes do primeiro deploy público. |
| IV | Test-first / verifiable DoD | ✅ | `tests/unit/test_api.py` cobre a API hoje; este ciclo mantém a suíte verde e adiciona verificação executável do gate de admin (ver DoD abaixo) em vez de alegar a proteção em prosa. |
| V | Context economy / cutting by boundary | ✅ | Corte pela fronteira "alcançar o backend de fora" — não inclui navegação do texto da tese nem limite de custo por visitante (ficam explicitamente fora de escopo na spec, R2). Fatia estreita: 4 arquivos de infraestrutura + 3 arquivos de código. |
| VI | Living artifacts | ✅ | `DEPLOY_AND_PUBLISH.md` (o guia que o autor já usa) é atualizado no mesmo ciclo que o `Dockerfile`/`render.yaml` que ele descreve — não fica um artefato descrevendo o outro que não existe. |
| VII | Light governance / YAGNI | ✅ | Nenhum framework novo: reusa o FastAPI e o React já existentes, um `Dockerfile` simples (sem orquestrador), um blueprint declarativo (`render.yaml`) em vez de passos manuais não versionados. |
| VIII | Intelligible communication | ✅ | Termos abertos na 1ª ocorrência nesta spec e neste plano: PDF (Portable Document Format), CORS (Cross-Origin Resource Sharing), DNS (Domain Name System), API (Application Programming Interface). |

**Veredito:** uma tensão real (III), resolvida por rework nesta mesma iteração — não foi para Complexity Tracking porque o rework a fechou por completo (a rota perigosa não fica exposta por padrão).

## Design

Ver `spec.md` §Arquitetura. Resumo executável:

1. `apps/web/src/api.ts` — `API_BASE` lido de `VITE_API_BASE` (padrão vazio =
   comportamento atual, relativo). Toda chamada de rede e toda URL usada em
   `href`/`src` passa por `apiUrl()`.
2. `Dockerfile` (raiz) — imagem CPU só com `[api,oracles]`; distinta da imagem
   dedicada do E2/E3 (`experiments/e2e3/Dockerfile`, que é GPU/torch).
3. `render.yaml` — blueprint declarativo do host; segredos (`DATABASE_URL`,
   chaves de oráculo, `FLOWBUILDER_ADMIN_TOKEN`) com `sync: false` — nunca
   entram no repositório.
4. `settings.py` + `app.py` — `FLOWBUILDER_ADMIN_TOKEN` opcional; quando
   definido, `POST /api/fichamentos` exige o header `X-Admin-Token` idêntico
   (comparação de tempo constante via `secrets.compare_digest`).
5. `DEPLOY_AND_PUBLISH.md` §1 — reescrito com o caminho concreto decidido
   (Render + Vercel, subdomínio) no lugar do rascunho anterior ("me avise se
   quiser esse caminho").

## Tasks

Ver `tasks.md`.

## DoD (verificável, princípio IV)

```bash
# 1. Suíte segue verde
python -m pytest tests/ -q

# 2. Front builda com e sem VITE_API_BASE
cd apps/web && npm run build && VITE_API_BASE=https://example.test npm run build

# 3 e 4. apiUrl() concatena a base em runtime (template literal), não em string
#    fundida no bundle — checar a CONSTANTE gerada, não uma URL já concatenada
#    (uma tentativa inicial de grep por 'https://.../api' deu falso-negativo
#    por isso mesmo: o bundle nunca contém a URL completa como literal único).
cd apps/web && npm run build && \
  ! grep -qo '"https://[a-zA-Z0-9.-]\+"' dist/assets/*.js   # vazia: nenhuma string-URL no bundle
cd apps/web && VITE_API_BASE=https://example.test npm run build && \
  grep -qo '"https://example.test"' dist/assets/*.js        # com host: a constante existe

# 5. O gate de admin é uma checagem, não uma promessa em prosa
cd /home/user/activelearning && python -c "
from fastapi.testclient import TestClient
from activelearning.adapters.api.app import create_app
from activelearning.adapters.api.settings import Settings
import os, tempfile
os.environ['FLOWBUILDER_ADMIN_TOKEN'] = 'segredo-de-teste'
s = Settings.from_env()
assert s.admin_token == 'segredo-de-teste'
app = create_app(s)
c = TestClient(app)
r = c.post('/api/fichamentos', files={'file': ('x.pdf', b'%PDF-1.4', 'application/pdf')})
assert r.status_code == 401, r.status_code
print('gate ok: sem token -> 401')
"

# 6. Dockerfile builda (evidência de que a receita não está quebrada)
docker build -t falco-flowbuilder-api-check . && echo "docker build ok"
```

**Item 6 — evidência parcial.** Este ambiente de execução não tem daemon Docker
(`docker build` falha com "no such file or directory" no socket) — o build
completo da imagem não foi verificado aqui. O que FOI verificado, reproduzindo
os mesmos passos do `Dockerfile` fora do container: `pip install ".[api,oracles]"`
em venv limpo succeeds; `uvicorn --factory
activelearning.adapters.api.app:create_app --host 0.0.0.0 --port $PORT`
(o `CMD` exato) sobe e responde `GET /api/health` com `{"status":"ok"}`. Falta
confirmar em ambiente com Docker: a camada de imagem em si (base `python:3.12-slim`,
`COPY`, permissões). Pendente para quem aplicar `docker build` de fato antes do
primeiro deploy (`DEPLOY_AND_PUBLISH.md` §1.4).

## TAIL

- `TAIL:security` — a rota `/api/fichamentos` mudou de superfície de risco
  (escreve fora deste repositório, agora hospedável publicamente); o gate por
  segredo compartilhado é a mitigação deste ciclo. Ficam registrados como
  dívida explícita para R2: teto de custo/rate-limit em `/api/runs` e
  `/api/experiments/{id}/execute` quando o oráculo real for habilitado.
- `TAIL:gate` — nenhum passo deste ciclo dispara deploy público sozinho;
  aplicar `render.yaml` no Render e ligar o subdomínio no Vercel continuam
  **[VOCÊ]** em `DEPLOY_AND_PUBLISH.md`. O merge desta branch é o gate humano
  do código; o primeiro deploy público é um segundo gate, do autor, fora
  deste repositório.
