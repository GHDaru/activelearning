# Tasks 003 — Dicionário de dados + invariantes executáveis do dataset

## Verification first

- [x] **T0** — Computar os números reais do CSV com as funções do pipeline
  ANTES de fixar expectativas (contagens de linhas/classes/schema/partições;
  md5). Evidência no `qa-report.md`.

## Implementation

- [x] **T1** — `data/DICIONARIO.md`: colunas, `normalize_label`, derivação
  620+`_rare_`=621, cadeia 250.221→231.490/714, partições, notas 715→714 e
  649-no-pool. (FR1)
- [x] **T2** — `scripts/check_dataset_invariants.py` com as funções reais
  (`load_rows_and_schema` + `run_e3prime.load_base` com stubs comentados),
  exit 0/1. (FR2, FR3)
- [x] **T3** — Provar a checagem em vermelho: CSV truncado ⇒ exit 1 nomeando
  invariantes. (DoD, segunda lei)
- [x] **T4** — `.gitignore`: exceção `!data/DICIONARIO.md` (whitelist de
  `data/` já usada pelos vizinhos).

## Living documentation (same pull request)

- [x] **T5** — O dicionário É a documentação viva; `DICIONARIO.md` aponta o
  script e vice-versa; plano de revisão da tese atualizado NA MAIN do tesedaru
  (v9, commit 'Plano v9') com links — a cópia do plano nesta branch é anterior
  (v8) e converge no merge.

## Closing tail — MANDATORY, one line each, never delete

- [x] `TAIL:review` — revisão independente em contexto fresco antes do gate (evidência no qa-report).
- [x] `TAIL:security` — varredura de segredos/injeção no diff (evidência no qa-report).
- [ ] `TAIL:gate` — gate humano do autor no merge da branch (aguardando).
