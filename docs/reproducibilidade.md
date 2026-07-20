# Reprodutibilidade

O guia completo de reprodutibilidade — ambiente, dados, o comando exato de cada
experimento, artefatos, custos e o roteiro de publicação (Zenodo / Papers with
Code) — vive na raiz do repositório:

- **[REPRODUCIBILITY.md](https://github.com/GHDaru/activelearning/blob/main/REPRODUCIBILITY.md)**

Princípio que rege o projeto:

!!! quote "Regra de ouro"
    **Nenhum número entra na tese sem um artefato rastreável** (config + git SHA +
    seeds + JSONL de execução). O repositório legado
    (`activetextclassification`) é somente leitura e preserva a origem de P1/P2.

## Proveniência dos experimentos

| Origem | O quê |
|--------|-------|
| `activetextclassification` (congelado) | P1 (sensibilidade de L0), P2 (DRI-SL, algoritmo genético), pilotos de oráculo — os **originais** citados na tese |
| `activelearning` (este repo) | E0, E0-P, E1, E4, E5, E6, E3′ e as **reexecuções** de verificação de P1/P2 |

Os números portados do legado foram verificados por igualdade de escores,
reexecução independente ou conferência direta de artefato — o registro dessa
auditoria acompanha o código (`docs/auditoria-experimentos-legado.md`).
