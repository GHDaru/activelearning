# E0 — Análise consolidada dos PILOTOS (16/07/2026)

Pilotos executados nesta sessão (não substituem o E0 completo, que roda na máquina
do autor com `config.json` integral). Amostras: **S-rand** n=100 (subamostra da
S-rand oficial, seed 42) e **S-strat** n=621 (1 instância/classe; a config atual
gera 3/classe=1.863 — os IDs `e0-{índice}` permanecem pareáveis via gold da base
inteira). Prompt v3; lote b=25 (MaaS) / b=10 (demais); T=0.

## Tabela (acurácia com IC95% de Wilson)

| Amostra | Oráculo (modo) | n | Acc | IC95% | Macro-F1 | Inválidos | US$/1k |
|---|---|---|---|---|---|---|---|
| rand | gpt-4o (enum) | 100 | 0,780 | [0,689; 0,850] | 0,739 | 0,0% | 0,881 |
| rand | nemotron-3-ultra :free (json-prompt) | 100 | 0,780 | [0,689; 0,850] | 0,730 | 2,0% | 0,000 |
| rand | deepseek-v4-pro (json-prompt) | 100 | 0,770 | [0,678; 0,842] | 0,728 | 0,0% | 0,412 |
| rand | glm-5.2 (json-prompt) | 100 | 0,770 | [0,678; 0,842] | 0,719 | 0,0% | 0,248 |
| rand | deepseek-v4-flash (json-prompt) | 100 | 0,730 | [0,636; 0,807] | 0,679 | 1,0% | 0,035 |
| rand | gpt-4o-mini (enum) | 100 | 0,650 | [0,552; 0,736] | 0,603 | 4,0% | 0,037 |
| rand | gpt-4o-mini (free, RQ4) | 100 | 0,580 | [0,482; 0,672] | 0,539 | 7,0% | 0,046 |
| strat | deepseek-v4-pro (json-prompt) | 621 | 0,844 | [0,813; 0,870] | 0,804 | 0,0% | 0,405 |
| strat | gpt-4o (enum) | 621 | 0,821 | [0,789; 0,849] | 0,782 | 0,5% | 0,837 |
| strat | deepseek-v4-flash (json-prompt) | 621 | 0,800 | [0,767; 0,830] | 0,757 | 0,8% | 0,034 |
| strat | gpt-4o-mini (enum) | 621 | 0,464 | [0,425; 0,503] | 0,429 | 3,5% | 0,037 |

## McNemar pareado — leituras principais

**S-rand (n=100)**: os cinco melhores (gpt-4o, nemotron, v4-pro, glm-5.2,
v4-flash) são estatisticamente indistinguíveis entre si; TODOS os pares
significativos envolvem gpt-4o-mini (enum ou free) — o mini é eliminado
(p≤0,0075 contra v4-pro, glm-5.2, gpt-4o, nemotron).

**S-strat (n=621)**:
- v4-pro > v4-flash: b=43 vs c=16, p=0,00071 — na amostra estratificada (que
  expõe classes raras) o modelo maior É significativamente melhor.
- v4-pro vs gpt-4o: b=31 vs c=17, p=0,061 — vantagem do v4-pro não atinge 5%.
- v4-flash vs gpt-4o: p=0,124 — empate estatístico a 1/25 do custo.
- gpt-4o-mini perde de todos com p≈0 (colapsa em classes raras: 46,4%).

**RQ4 (efeito do instrumento, mini enum vs free)**: 65% vs 58%, p=0,065 —
direção consistente com a tese do instrumento; confirmar com n=1000 no E0 cheio.

## Leitura do gate (critérios da spec 001)

- Nenhum modelo atinge acc ≥ 85% pontual em nenhuma amostra; o melhor é
  v4-pro na strat (84,4%, IC inclui 85%). **E4 (robustez a ruído) tende a ser
  obrigatório** — decisão final só com o E0 completo (n=1000 rand + 3/classe).
- Candidato custo-ótimo: **deepseek-v4-flash** (US$0,034/1k; 80% strat; empate
  estatístico com gpt-4o na strat).
- Candidato acurácia: **deepseek-v4-pro** (84,4% strat; 0% inválidos).
- Teto de acurácia mensurável ≈ 99,3% (conflitos de gold, ver
  `data/CORRECTIONS.md`); parte dos erros observados são golds questionáveis.

## Artefatos

- `e0_table.json` / `e0_mcnemar.json`: gerados por `analyze_e0.py --results
  experiments/e0/results_pilot`.
- `rand/`, `strat/`: anotações JSONL (retomáveis) + relatórios por oráculo.
- `../results_pilot_strict/`: reexecução pós-correção `strict:true` (validação
  do schema; ver histórico).
