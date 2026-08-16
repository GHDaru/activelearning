# Plan 003 — Dicionário de dados + invariantes executáveis do dataset

- **Spec**: `spec.md` · **Lane**: full · **Date**: 2026-08-16

## Constitution Check (governance/principles.md do método Maestro)

| Principle | Compliance |
|---|---|
| I. Spec-driven | ✅ nasce do grupo `dados` do plano de revisão (artefato versionado aprovado pelo autor) e desta spec; nenhum escopo além dela. |
| II. Human-governed orchestration | ✅ agente executa na branch; merge somente com gate humano do autor (regra `gate` do plano). |
| III. Reversibility / risk gates | ✅ só criação de arquivos novos + 1 linha aditiva no `.gitignore`, em branch — reversível por descarte da branch. |
| IV. Test-first / verifiable DoD | ✅ o entregável central É a checagem executável; falha provada de propósito (CSV truncado ⇒ exit 1) antes de confiar no verde. |
| V. Context economy / boundary | ✅ recorte por fronteira: só `data/` + `scripts/`; zero contato com domínio, adaptadores ou experimentos. |
| VI. Living artifacts | ✅ dicionário e script nascem no mesmo pull request e referenciam-se mutuamente; `CORRECTIONS.md` existente citado, não duplicado. |
| VII. Light governance / YAGNI | ✅ nenhum framework de validação novo: stdlib + funções já existentes do pipeline. |
| VIII. Intelligible communication | ✅ siglas abertas no dicionário (CSV, md5, CategorySchema explicado); tabelas com fonte de cada número. |

**No violations.**

## Artifacts of this cycle

| Artifact | Declaration | Why |
|---|---|---|
| `research.md` | `ART:research=no` | nenhuma incógnita técnica: as funções do pipeline já existem; o trabalho é prová-las. |
| `data-model.md` | `ART:data-model=no` | o "modelo de dados" é o próprio dicionário entregue (`data/DICIONARIO.md`). |
| `contracts/` | `ART:contracts=no` | nenhuma interface nova: o script consome funções públicas existentes. |
| `checklist.md` | `ART:checklist=no` | o DoD da spec já é executável; checklist duplicaria função (princípio VI). |
| `ux-design.md` | `ART:ux-design=no` | não toca tela. |

## How

- Gabarito dos números OBTIDO ANTES de escrever qualquer expectativa
  (diagnose-before-fix): contagens computadas do CSV com as funções reais e
  só então congeladas como `ESPERADO` no script.
- O script importa `run_e3prime.load_base` de verdade; os imports pesados do
  módulo (sklearn, adaptador BERTimbau/torch) são substituídos por stubs
  explícitos e comentados — `load_base` não os usa; dependência falsa seria
  pior que o stub.
- Divergência conhecida documentada, não corrigida em silêncio:
  `docs/plano-mestre.md` traz 795/622 (versão antiga do CSV, 250.365); o CSV
  vigente dá 794/621 — reportado ao autor.

## Verification (DoD)

Ver `qa-report.md`: cada critério da spec com comando, esperado e resultado real.
