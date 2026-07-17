# Avaliação de resultados e geração de gráficos

Rotinas prontas para transformar os artefatos dos experimentos em números de
tese (com estatística) e figuras (PDF para o LaTeX, PNG para docs).

## Rotinas de avaliação (números)

| Rotina | Entrada | Saída | O que calcula |
|---|---|---|---|
| `experiments/e0/analyze_e0.py` | `e0/results/**/annotations_*.jsonl` | stdout + resumo | IC de Wilson por oráculo/amostra, McNemar pareado (binomial exato <25 discordâncias), custo por 1k, decisão do gate |
| `experiments/e0/analyze_noise_impact.py` | idem + gabarito | `e0/results/noise_impact.json` | sensibilidade ao ruído do gabarito (excluir conflitantes × multi-gold) |
| `experiments/e0p/analyze_e0p.py` | `e0p/results/**` + anotações v3 do E0 | `e0p/results/analysis.json` | acurácia v3/v4a/v4b + McNemar exato pareado (mesmos 500 itens) |
| `experiments/e1e4/analyze_e1e4.py` | `e1e4/results/sweeps.jsonl` | `e1e4/results/analysis.json` | média±dp de LCE e Macro-F1 final por célula, Wilcoxon pareado por semente, ablação de lote, retenção sob ruído |

Todas rodam offline (sem chaves) e são idempotentes — pode reexecutar após
qualquer run novo para reconsolidar.

Convenções estatísticas do projeto:

- **IC de acurácia**: Wilson 95% (nunca normal-aproximado em n≤2k).
- **Comparação pareada de oráculos**: McNemar sobre os discordantes; binomial
  exato quando os discordantes somam <25.
- **Comparação de estratégias/sementes**: Wilcoxon pareado por semente; com 8
  sementes o menor p atingível é 2/2⁸ = 0,0078 — reporte esse teto junto.
- **LCE**: AUC da curva real (Simpson) normalizada pelo teto supervisionado —
  ver Cap. 3 da tese e `domain/metrics`.

## Figuras

```bash
pip install matplotlib
python experiments/plots/make_figures.py            # todas
python experiments/plots/make_figures.py --only e1  # uma família
```

Saída em `experiments/plots/figures/`:

| Figura | Conteúdo | Fonte |
|---|---|---|
| `fig_e1_curvas` | curvas de aprendizado (Macro-F1 × \|L\|), 5 estratégias, média de 8 sementes | `sweeps.jsonl` |
| `fig_e4_ruido` | Macro-F1 final × ε com ±dp, entropia vs aleatória, faixa dos LLMs reais destacada | `sweeps.jsonl` |
| `fig_e0_custo_acuracia` | custo US$/1k (simlog) × acurácia S-rand por oráculo (modo de produção; execuções completas) | `e0/results/rand/report_*.json` |

Para usar na tese: copie os `.pdf` para `tesedaru/5-resultados-falco/imagens/` e
inclua com `\includegraphics`.

### Padrões visuais (não negociáveis)

- **Paleta categórica fixa e validada** (colorblind-safe): azul `#2a78d6`,
  verde `#008300`, magenta `#e87ba4`, amarelo `#eda100`, aqua `#1baf7a` — nesta
  ordem, nunca reciclada. Validada com o verificador de paleta (CVD ΔE 9,1;
  piso de visão normal 19,6; todas as checagens passam).
- Identidade **nunca só por cor**: cada série tem estilo de linha e marcador
  próprios + rótulo direto no fim da linha (isso também cobre impressão P&B).
- **Um eixo só** — nunca eixo duplo; grade recessiva; sem moldura.
- Novas figuras: siga `make_figures.py` como referência de estilo e valide
  qualquer paleta nova antes de usar.

## Receita para um experimento novo

1. Runner grava JSONL/JSON em `experiments/<exp>/results/` (retomável, com
   semente e proveniência).
2. `analyze_<exp>.py` agrega e TESTA (IC/McNemar/Wilcoxon conforme o caso) →
   `results/analysis.json`.
3. Figura em `make_figures.py` (nova função `fig_<exp>_...`).
4. Commit dos três; a tese cita o `analysis.json` como artefato.
