# Auditoria dos experimentos originais (repositório legado)

Data: 19/07/2026. Motivada pela pergunta do autor: "estão faltando
resultados? o AG por exemplo está ali?". Resposta: **sim, faltavam** — o
catálogo cobria apenas o repositório novo; os experimentos ORIGINAIS de
P1/P2 vivem no repositório `GHDaru/activetextclassification` (público).
Esta auditoria mapeia número da tese → código → artefato → estado.

## Inventário

| Número na tese | Código (repo legado) | Artefato | Estado |
|---|---|---|---|
| P1: sensibilidade 47×30 (amplitude 6,4 p.p. em I=100; tab. do apêndice) | `examples/L0_experimento.ipynb` | `examples/l0_random_impact_analysis_outputs/l0_stats_table_analyzed.tex` (a própria tabela do apêndice) + planilhas | **VERIFICADO** — arquivo localizado; validado indiretamente pela reexecução (≤0,7 p.p., `experiments/p1/replay_l0.jsonl`) |
| AG: envelope evolutivo (tab:ag-evolucao) | `activetextclassification/optimization/genetic_l0_optimizerv4.py` | `examples/ag_optimization_results_L0_*/` (CSV de aptidão detalhada + best_l0 + checkpoints .pkl) e `examples/allag.xlsx` | **VERIFICADO** — configuração real confirmada NO CÓDIGO: população **50**, **100 gerações**, elitismo; 4 cenários. Sufixos `old/oldold/v1/v2` indicam re-execuções — a consolidação de qual versão alimentou a tabela da tese está em `allag.xlsx` (verificação fina pendente de abertura da planilha) |
| P2: DRI-SL vs AG (tab:drisl-vs-ag) | `activetextclassification/cold_start/dri_cluster.py` (DRIClusterColdStart) + `examples/coldstart_evaluate.ipynb` | notebook localizado; **artefato numérico final da comparação NÃO localizado ainda** | **AUDITORIA PENDENTE** — ver ações abaixo |
| Oráculo piloto (origem do RQ4: saída livre → falsos erros) | `examples/oraculo.ipynb` + `data_oraculo/` | dados do piloto presentes | **VERIFICADO** (como registro histórico; o E0 braço *free* é a réplica controlada) |

## Correções derivadas da auditoria

1. **A3 (artigo)**: a configuração do AG agora é escrita com o valor
   verificado (população 50, 100 gerações) — substituindo a remissão
   genérica que ficou após remover os detalhes não-verificados.
2. **Catálogo do FALCO**: 4 entradas novas com chip `legado` e estado de
   auditoria; replay lê artefatos do repositório legado quando presente
   (`FALCO_LEGACY_ROOT`, padrão `../activetextclassification`).

## Ações pendentes (marcadas para verificação e auditoria)

- [ ] **P2**: abrir `coldstart_evaluate.ipynb` e localizar o artefato
      numérico (CSV/planilha) que gera a tab:drisl-vs-ag; registrar o
      caminho no catálogo; conferir os 10 números da tabela da tese
      contra o artefato.
- [ ] **P2**: portar um runner reproduzível `experiments/p2/` para a
      biblioteca nova (DRI-SL da biblioteca × melhores indivíduos do AG
      carregados dos checkpoints legados), com reavaliação em teste
      intocado (protocolo anticircularidade).
- [ ] **AG**: abrir `allag.xlsx` e confirmar qual versão (v1/v2/old)
      alimentou cada linha da tab:ag-evolucao.
- [ ] Documentar no Cap. 3 (ou apêndice) a proveniência dupla:
      "resultados originais no repositório activetextclassification;
      reexecuções na biblioteca activelearning" — já dito no Cap. 4,
      conferir se basta.
