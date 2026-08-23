# Estatística pareada — varredura 100% homogênea (pós-clipping)

McNemar + bootstrap (10k réplicas, `stats_e3prime.py --pairs A-B,B-C,E35-D`)
recomputados sobre os 27/27 braços do E3′ (3 sementes × 9 braços) já **todos**
treinados com `bertimbau.py` corrigido (`activelearning@1dabdbb`, gradient
clipping) — tarefa tesedaru 2015 / `dec-regerar-25-bracos-aprovado`.

25 dos 27 braços foram retreinados nesta rodada (sufixo `_bs16v2` em
`experiments/e2e3/results/`); os 2 restantes (E25/s42, E/s123) já tinham sido
regenerados com clipping numa correção anterior (sufixo `_bs16`, preservados
como referência de estabilidade — não retreinados de novo). As predições
usadas aqui são cópias temporárias com nome plano
(`e3prime_<braço>_s<semente>_pred.json`, exigido pelo `stats_e3prime.py`),
apagadas depois de gerar estes dois artefatos — a fonte permanente de cada
braço é o `_bs16v2`/`_bs16` correspondente na pasta de cima.

Compare com `mcnemar_s<semente>_bs16.json` / `bootstrap_f1_s<semente>_bs16.json`
(a varredura mista de 2026-08-18, antes desta regeração) para ver os deltas.

## O que mudou vs a varredura mista

| | mista (2026-08-18) | homogênea (esta rodada) |
|---|---|---|
| Hipótese central F1(A)≥0,95×F1(D) | NÃO sustentada | **NÃO sustentada** (sem mudança; gap ainda maior: A médio 0,2972 vs 0,95×D médio 0,4365) |
| E35 > D | sustentada, significativa nas 3 sementes | **sustentada em 42 e 123; INVERTE em s7** (E35 < D, ΔF1 bootstrap [-0,0084;-0,0017], McNemar acc não sig., p=0,67) |
| B > C (seleção vs aleatório) | consistente nas 3 sementes | **continua consistente** nas 3 (bootstrap ΔF1 positivo, IC longe de zero, nas 3) |
| Piso de orçamento (F1 cruza 0,95×D) | E25 (50% do pool) | **E30 (60% do pool)** — E25 ficou estável (~0,4324) enquanto D subiu (0,4508→0,4594), então E25 deixou de cruzar |
| Piso de orçamento (acurácia cruza 0,95×D) | E20 (40% do pool) | **E20 (40% do pool)** — sem mudança |

Detalhe do porquê o piso de F1 mudou: D médio subiu de 0,4508 para 0,4594
(a maioria dos D foi regenerada com clipping e ganhou); E25 médio ficou em
0,4324 nas duas rodadas (E25/s42 não foi retreinado — é a referência fixa —
e os outros dois moveram pouco). 0,95×D subiu de 0,4283 para 0,4365, e
0,4324 passou a ficar abaixo desse novo limiar.
