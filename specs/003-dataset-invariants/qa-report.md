# QA report 003 — Dicionário de dados + invariantes executáveis do dataset

- **Date**: 2026-08-16 · **Lane**: full · **Verdict**: ✅ COMPLIANT (aguardando gate humano)

## Fitness functions (DoD)

| Check | Expected | Result |
|---|---|---|
| `python3 scripts/check_dataset_invariants.py` | exit 0, 16 linhas `[OK ]` | exit 0, `grep -c "[OK ]"` = 16 ✅ |
| mesmo script com CSV truncado (1 linha removida) | exit 1 nomeando invariantes | exit 1 — "FALHOU (6 invariante(s)): md5_csv, linhas_csv, dedup_textos, classes_no_pool, populacao, linhas_pos_filtro_ge2" ✅ |
| `grep -c "_rare_" data/DICIONARIO.md` | ≥ 1 | 6 ✅ |
| `git check-ignore data/DICIONARIO.md` | exit 1 (versionável) | exit 1 ✅ |
| `md5sum data/dataset.csv` | `0682ee5ba077c180fb7a727fb200f154` (= script = dicionário) | idêntico nos três lugares ✅ |

## Closing tail — the evidence

- **TAIL:review** — revisão independente em contexto fresco (agente `review`,
  sessão separada, 2026-08-16). Veredito: **aprovado com ressalvas**; 5
  achados (A–E), nenhum de corretude. Disposição: A (cobertura do dicionário
  além do que o script checava + código morto `cnt`) → CORRIGIDO: script
  passou de 12 para 16 invariantes, cobrindo toda contagem da página
  (794/174/250.142/715) e o código morto foi removido; B (T5 citava plano v9
  que está na main, não na branch) → CORRIGIDO: T5 reescrito explicitando
  main vs branch; C (qa-report ausente) → este arquivo; D (checagem
  tautológica do holdout) → CORRIGIDO: substituída por checagem aritmética
  pool + 4.000 + população = dedup; E (limitação do modo de teste) →
  CORRIGIDO: limitação documentada em comentário. O revisor recomputou
  independentemente os números não cobertos e reproduziu os caminhos verde e
  vermelho.
- **TAIL:security** — varredura de segredos no diff da branch
  (`git diff origin/main...HEAD | grep -niE "api[_-]?key|secret|token|..."`):
  nenhum segredo; únicos matches são falsos positivos de prosa. Diff 100%
  aditivo, sem código executado em import, sem rede, sem escrita fora do repo.
- **TAIL:gate** — aguardando gate humano do autor no merge da branch
  `claude/tesedaru-activelearning-maestro-bf56y7`. ATENÇÃO ao arbitrar:
  existe entrega equivalente do revisor1 na branch
  `claude/maestro-cycles-statistical-analysis-fwla6a` (`data/DICIONARIO.md` +
  `scripts/check_dataset.py`); os números conferem entre as duas (verificação
  cruzada §6) — escolher UMA para merge e usar a outra como registro de
  verificação.

## Requirement coverage

- **FR1**: entregue — `data/DICIONARIO.md` (toda contagem agora coberta pelo script).
- **FR2**: entregue — `scripts/check_dataset_invariants.py` com as funções reais; stubs auditados pela revisão (falham ruidosamente se usados).
- **FR3**: entregue — md5 conferido em script, dicionário e execução.
