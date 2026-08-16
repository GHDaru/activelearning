# Spec 003 — Dicionário de dados + invariantes executáveis do dataset

- **Status**: Em gate · **Lane**: full · **Date**: 2026-08-16
- **Origin**: grupo `dados` do plano de revisão da tese
  (`tesedaru/docs/records/plano-revisao.json`, itens `dicionario-dados` e
  `pipeline-reprodutivel`), derivado do parecer ARS R6 (achado Persp-C1 e
  TODOs herdados do artigo a5). Solicitação do autor: toda branch segue o
  fluxo Maestro completo (spec → plan → tasks → implement).

## What and why

Os números estruturais do dataset citados na tese — 250.221 linhas, 231.490
textos únicos, 714 classes, esquema de categorias (CategorySchema) com 620
classes + `_rare_` = 621, pool de 50.000 com 649 classes, população reservada
de 177.490 — existiam apenas como afirmações de prosa e contagens avulsas em
documentos. O princípio V da constituição da tese ("nenhum número sem artefato
rastreável") e o princípio IX ("DoD de texto verificável") exigem que cada um
resolva para um artefato executável. Este ciclo entrega (1) o dicionário de
dados junto ao dataset e (2) a checagem executável que prova os números a
partir do CSV publicado usando as funções reais do pipeline.

## Functional requirements

- **FR1**: dicionário de dados em `data/DICIONARIO.md` documentando colunas,
  normalização de rótulos, derivação do esquema 620+`_rare_`=621 e a cadeia de
  contagens 250.221 → 250.142 → 231.490/714 → partições 50k/4k/177.490.
- **FR2**: script `scripts/check_dataset_invariants.py` que prova cada número
  com as funções REAIS (`retail_csv.load_rows_and_schema` e
  `run_e3prime.load_base`), nunca cópias da lógica; exit 0 verde / exit 1 com
  o invariante nomeado.
- **FR3**: o md5 da versão exata do CSV registrado e verificado (insumo do
  item `md5-versao`, cuja prosa no Capítulo 3 fica com o agente principal).

## Out of scope

- Editar o texto da tese (Capítulo 3) — prosa é superfície do principal.
- Publicação no Kaggle e licença (itens `kaggle-licenca`, dono: autor).
- Corrigir os números desatualizados de `docs/plano-mestre.md` (795 brutas /
  622) — achado reportado ao autor, correção é ciclo próprio.

## Acceptance criteria (DoD)

- `python3 scripts/check_dataset_invariants.py` → exit 0, 17 linhas `[OK ]`.
- WHEN o CSV é alterado (uma linha removida) THE SYSTEM SHALL sair com exit 1
  nomeando os invariantes violados (segunda lei do verifiable-dod: checagem
  provada em vermelho).
- `grep -c "_rare_" data/DICIONARIO.md` → ≥ 1 (esquema documentado).
- `git check-ignore data/DICIONARIO.md` → exit 1 (arquivo versionável).

## Clarify

- 715 classes têm ≥2 ocorrências antes do dedup, mas o dedup elimina todas as
  instâncias de uma classe → 714 no conjunto experimental (documentado).
- 649 = classes presentes no pool de 50k (o "649 raw classes" do artigo a5).
