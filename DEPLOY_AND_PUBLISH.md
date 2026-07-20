# Deploy e publicações — FALCO

Guia do que **você** precisa configurar. Tudo que dá para versionar já está no
repositório (configs, workflows, metadados). O que exige suas contas/segredos
está marcado com **[VOCÊ]**.

Índice: [1) Vercel](#1-vercel-front--backend-read-only) ·
[2) PyPI](#2-pypi-biblioteca) · [3) Zenodo/DOI](#3-zenodo-doi-citável) ·
[4) Papers with Code](#4-papers-with-code) · [5) Segredos](#5-segredos-a-rotacionar)

---

## 1) Vercel (front + backend read-only)

**O que sobe:** o front (React/Vite) + uma função serverless **somente leitura**
que serve a base de conhecimento (grafo). Rodar experimentos, subir datasets e
servir PDFs **não** cabe no serverless do Vercel (processos longos, disco,
direitos autorais) — isso continua local ou num host de contêiner (ver 1.4).

Arquivos já prontos: `vercel.json`, `api/index.py`, `api/requirements.txt`,
`api/data/kg.{json,html}`, `.vercelignore`.

### 1.1 Passos **[VOCÊ]**
1. Suba o repositório `activelearning` no GitHub (se ainda não estiver).
2. Em <https://vercel.com> → **Add New Project** → importe `GHDaru/activelearning`.
3. **Framework Preset:** Other (o `vercel.json` já define build e rotas).
4. **Environment Variables** → adicione:
   - `VITE_HOSTED = 1`  → ativa o modo demo (esconde experimentos/upload/PDF).
5. **Deploy**. Ao final você recebe uma URL `https://<projeto>.vercel.app`.

### 1.2 Como funciona
- `vercel.json` builda `apps/web` (saída `apps/web/dist`) e roteia `/api/*` para
  `api/index.py` (FastAPI read-only). O front chama os mesmos `/api/kg/*`.
- O grafo é um snapshot em `api/data/`. Para atualizar depois de mexer nos
  fichamentos: `FALCO_THESIS_ROOT=../tesedaru python scripts/sync_public_kg.py`
  e faça commit — o próximo deploy publica o grafo novo.

### 1.3 Domínio próprio **[VOCÊ]** (opcional)
Vercel → Project → **Settings → Domains** → adicione `falco.seudominio.com` e
aponte o DNS (CNAME para `cname.vercel-dns.com`) conforme as instruções da tela.

### 1.4 Backend completo (opcional, quando quiser rodar experimentos online)
O FastAPI inteiro (`activelearning.adapters.api.app:create_app`) roda bem num
host de contêiner. Sugestão (Render):
- New **Web Service** → repo `activelearning`.
- Build: `uv sync --extra api` · Start:
  `uv run uvicorn --factory activelearning.adapters.api.app:create_app --host 0.0.0.0 --port $PORT`
- Env: `DATABASE_URL` (Neon/Postgres), `FALCO_THESIS_ROOT`, chaves de oráculo.
- No front, troque as chamadas para essa URL (hoje o front usa `/api` relativo;
  para um back externo, sirva o front atrás do mesmo domínio ou adicione um
  `VITE_API_BASE`). Me avise se quiser esse caminho — preparo o `VITE_API_BASE`.

---

## 2) PyPI (biblioteca)

Nome de distribuição: **`falco-active-learning`** (o import continua
`import activelearning`). Já configurado em `pyproject.toml` (+ `LICENSE`,
classifiers, URLs). O build gera `dist/*.whl` e `dist/*.tar.gz` com `uv build`.

### 2.1 Trusted Publishing (recomendado — sem token) **[VOCÊ]**
Workflow pronto: `.github/workflows/publish-pypi.yml` (publica ao criar um
Release no GitHub).
1. Crie conta em <https://pypi.org> e verifique o e-mail.
2. <https://pypi.org/manage/account/publishing/> → **Add a pending publisher**:
   - PyPI Project Name: `falco-active-learning`
   - Owner: `GHDaru` · Repository: `activelearning`
   - Workflow name: `publish-pypi.yml` · Environment: `pypi`
3. No GitHub → **Releases → Draft a new release** → tag `v0.1.0` → **Publish**.
   O Action builda e publica. Confira em `https://pypi.org/project/falco-active-learning/`.
4. Teste: `pip install falco-active-learning` (núcleo) ou
   `pip install "falco-active-learning[classifiers,oracles,api]"`.

### 2.2 Alternativa por token
Se preferir token: gere em PyPI → Account → API tokens, salve como secret
`PYPI_API_TOKEN` no GitHub e troque o passo de publish por `twine upload` com o
token. (O Trusted Publishing é mais seguro; prefira 2.1.)

---

## 3) Zenodo (DOI citável)

Gera um DOI para o **software** e, separadamente, para a **tese** — cada repo
tem seu `.zenodo.json` com os metadados prontos.

### 3.1 Passos **[VOCÊ]**
1. Entre em <https://zenodo.org> com o GitHub (login social).
2. <https://zenodo.org/account/settings/github/> → ligue o interruptor dos
   repositórios `GHDaru/activelearning` e `GHDaru/tesedaru`.
3. Crie um **Release** no GitHub em cada repo (ex.: `v0.1.0`). O Zenodo captura
   o release e emite o DOI automaticamente (usa o `.zenodo.json`).
4. Copie o **badge DOI** que o Zenodo mostra e cole no topo do `README.md`.
   (O DOI "concept" agrega todas as versões; o DOI de versão aponta a release.)

> Ordem prática: faça o release do PyPI (2.1) e do Zenodo juntos — a mesma tag
> `v0.1.0` serve para os dois.

---

## 4) Papers with Code

O Papers with Code lista **papers** com código associado. É uma submissão
**manual** no site (não dá para automatizar), e ele exige uma **URL de paper**
pública. Fluxo recomendado:

### 4.1 Pré-requisito: URL do paper **[VOCÊ]**
- Opção A (rápida): use o **DOI da tese no Zenodo** (passo 3) como URL do paper.
- Opção B (mais visibilidade): publique um dos artigos derivados (A1–A5) ou a
  tese no **arXiv** e use a URL do arXiv. O arXiv é o formato que o PwC melhor
  reconhece (importa título/abstract automaticamente).

### 4.2 Submissão **[VOCÊ]**
1. Conta em <https://paperswithcode.com> (login GitHub).
2. **Add Paper** → cole a URL (arXiv/Zenodo). Confira título, autores, abstract.
3. **Add Code** → link do repositório `github.com/GHDaru/activelearning`
   (marque como *official implementation*).
4. **Add Dataset** → busque/crie "Classificação de Produtos — Varejo CPG PR/BR"
   com o DOI `10.34740/kaggle/dsv/4265348`.
5. **Tasks/Methods:** marque `Active Learning`, `Text Classification`,
   `Short Text Classification`. (SOTA/leaderboard é opcional.)

### 4.3 Material pronto para colar
Um rascunho de abstract, tarefas e links está em
[`docs/paperswithcode.md`](docs/paperswithcode.md) — copie de lá.

---

## 5) Segredos a rotacionar **[VOCÊ]**

Nada de segredo vai para o Vercel/PyPI/Zenodo neste plano (o demo é read-only e
o publish usa OIDC). Ainda assim, como higiene:
- Rotacione as 5 chaves de API dos oráculos (OpenAI, OpenRouter, MaaS/Huawei,
  NVIDIA, Neon). Elas só existem no `.env` local (gitignored) — confirme que
  **nenhuma** foi commitada: `git log -p | grep -i "api_key" | head`.
- Se for usar o backend completo (1.4), ponha as chaves nas *Environment
  Variables* do host de contêiner, nunca no repositório.
