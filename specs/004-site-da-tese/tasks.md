# Tasks 004 — Site da tese: fundação

## T1 — Front: base da API configurável
- [x] `apps/web/src/vite-env.d.ts` — declara `VITE_API_BASE`.
- [x] `apps/web/src/api.ts` — `API_BASE` + `apiUrl()`; toda chamada de rede e
      todo `href`/`src` (download, grafo) passam por `apiUrl()`.
- [x] Verificado: `npm run build` sem a variável não emprega nenhum host
      absoluto no bundle; com a variável, a constante aparece.

## T2 — Backend: empacotamento para host de contêiner
- [x] `Dockerfile` (raiz) — imagem CPU, `[api,oracles]`, sem `--reload`.
- [x] `render.yaml` — blueprint declarativo, segredos com `sync: false`,
      disco persistente para `FLOWBUILDER_ARTIFACTS`.
- [x] Verificado sem daemon Docker (ambiente sem acesso): `pip install
      ".[api,oracles]"` + `uvicorn --factory ... create_app` reproduzem o
      `Dockerfile` fora do container e respondem `/api/health`. Build real da
      imagem fica pendente para quem tiver Docker (nota em `plan.md`).

## T3 — Segurança: fechar `/api/fichamentos` para host público
- [x] `settings.py` — `admin_token` via `FLOWBUILDER_ADMIN_TOKEN`.
- [x] `app.py` — gate `X-Admin-Token` (comparação de tempo constante) na
      rota que escreve fora deste repositório.
- [x] `tests/unit/test_api.py` — dois testes novos: 401 sem token/com token
      errado quando configurado; comportamento antigo preservado quando a
      variável não está definida.

## T4 — Documentação viva (Princípio VI)
- [x] `DEPLOY_AND_PUBLISH.md` §1.4 reescrito com o caminho concreto (Render +
      Neon + Vercel), substituindo o rascunho "me avise se quiser esse
      caminho".

## T5 — DoD e evidência
- [x] `python -m pytest tests/ -q` → 88 passed (86 prévios + 2 novos de T3).
- [x] `npm run build` (com e sem `VITE_API_BASE`) → ambos verdes; checagem
      corrigida em `plan.md` depois que a primeira tentativa deu
      falso-negativo (grep assumia string fundida; `apiUrl()` concatena em
      runtime).
- [ ] `docker build` real — não executável neste ambiente; ver nota em
      `plan.md` §DoD item 6.

## Fora deste ciclo (R2 — registrado em `spec.md` §Fora do escopo)
- Navegação do texto da tese dentro do site.
- Página por experimento com receita de reprodução ao lado do resultado.
- Teto de custo/rate-limit por visitante quando o oráculo real for habilitado.
