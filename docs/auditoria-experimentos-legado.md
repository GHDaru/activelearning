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
| P2: DRI-SL vs AG (tab:drisl-vs-ag) | `activetextclassification/cold_start/dri_cluster.py` (DRIClusterColdStart) + `examples/coldstart_evaluate.ipynb` | `data_splits_cache/dri_vs_random_final_log_results.csv` | **VERIFICADO 19/07** — cruzamento automático: os 10 números DRI-SL da tabela da tese (acc e Macro F1 em I=100/500/1.000/2.500/5.000) conferem EXATAMENTE (diferença < 0,05 p.p.) com o CSV |
| Oráculo piloto (origem do RQ4: saída livre → falsos erros) | `examples/oraculo.ipynb` + `data_oraculo/` | dados do piloto presentes | **VERIFICADO** (como registro histórico; o E0 braço *free* é a réplica controlada) |

## Correções derivadas da auditoria

1. **A3 (artigo)**: a configuração do AG agora é escrita com o valor
   verificado (população 50, 100 gerações) — substituindo a remissão
   genérica que ficou após remover os detalhes não-verificados.
2. **Catálogo do FALCO**: 4 entradas novas com chip `legado` e estado de
   auditoria; replay lê artefatos do repositório legado quando presente
   (`FALCO_LEGACY_ROOT`, padrão `../activetextclassification`).

## Ações pendentes (marcadas para verificação e auditoria)

- [x] **P2**: artefato localizado (`data_splits_cache/dri_vs_random_final_log_results.csv`)
      e os 10 números da tabela conferidos automaticamente — TODOS batem
      (19/07/2026).
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

## Porte para a biblioteca nova: estado honesto (auditado 19/07)

| Componente legado | Porte na biblioteca nova | Verificação |
|---|---|---|
| PVBin (classificador) | `adapters/classifiers/pvbin.py` | **numérica**: igualdade exata das matrizes de escore entre implementações (declarada na tese, Cap. 4) |
| DRI-SL (`cold_start/dri_cluster.py`) | `adapters/strategies/drisl.py` | **conceitual, NÃO numérica**: mesmo algoritmo (cluster semântico + relevância + novidade lexical), reimplementado com encoder próprio (TF-IDF+SVD). Usado APENAS nos experimentos novos (E5/E6/E3'); os números de P2 na tese vêm do artefato legado verificado. Equivalência numérica não é requerida nem alegada — registrar se a banca perguntar |
| AG (`optimization/genetic_l0_optimizerv4.py`) | NÃO portado; `experiments/p1/replay_ga.py` reimplementa o MECANISMO em escala reduzida (Npop=30, 40 ger.) com protocolo anticircularidade | mecanismo reproduzido (+5,2 p.p. em I=50); envelope completo permanece o legado |
| Laço de AL / oráculos | `application/run_falco.py`, `adapters/oracles/*` | testes de unidade + execuções ponta a ponta (E5 real) |

**Arquitetura (DDD/hexagonal)**: os portes seguem a constituição — domínio puro
em `domain/`, casos de uso em `application/`, tudo que toca IO/modelos/API em
`adapters/`. O catálogo de experimentos vive em `adapters/api/` (correto:
orquestra subprocessos e filesystem). Dívida menor conhecida: o cálculo de
estatísticas de dataset está inline no adapter da API (funcional e testado;
extraível para `application/` se crescer).

**Testes (19/07)**: suíte com 80 casos, 100% verdes — incluindo os novos
`test_experiments_catalog.py` (status/replay/subprocess/presets) e
`test_dataset_stats`. A escrita dos testes do catálogo encontrou e corrigiu
um bug real: filhos de execução viravam zumbis e o status ficava
"executando" para sempre (corrigido com waitpid + inspeção de /proc).
