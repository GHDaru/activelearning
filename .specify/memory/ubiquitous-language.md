# Ubiquitous Language — activelearning

Termos do domínio de Aprendizado Ativo com oráculo LLM, como usados na tese (FALCO) e
neste código. Código, specs e texto da tese DEVEM usar estes nomes.

## Aggregates & Entities

| Termo | Definição | Código |
|---|---|---|
| **Instance** | Um texto candidato a rotulagem, com id estável e, em simulação, um rótulo-ouro (`gold_label`). | `domain.instances.Instance` |
| **Pool (U)** | Conjunto de Instances não rotuladas disponível para consulta. Diminui a cada iteração. | `domain.pool.Pool` |
| **LabeledSet (L)** | Conjunto de Instances já anotadas (com `Annotation`). `L0` é o estado inicial (cold start). | `domain.pool.LabeledSet` |
| **ActiveLearningRun** | Uma execução completa do ciclo de AA: configuração + sequência de iterações + curva de aprendizado. Aggregate root do experimento. | `application` (use case) + `ExperimentLogPort` |

## Value Objects

| Termo | Definição | Código |
|---|---|---|
| **Label** | Rótulo de classe, normalizado (minúsculas, sem acento). | `domain.instances.Label` |
| **CategorySchema** | O conjunto FECHADO de Labels válidos da tarefa (+ sentinela `_RARE_`). Fonte única do `enum` dos oráculos LLM. | `domain.instances.CategorySchema` |
| **Annotation** | Rótulo atribuído a uma Instance por um oráculo, com proveniência (modelo, prompt, custo) e validade (`is_valid_label`). | `domain.annotation.Annotation` |
| **OracleUsage** | Tokens de entrada/saída, latência e custo monetário estimado de uma chamada de oráculo. | `domain.annotation.OracleUsage` |
| **Budget (B)** | Orçamento total de consultas ao oráculo e seu consumo corrente. | `domain.budget.Budget` |
| **QueryBatch (Q_t)** | O lote de Instances selecionado na iteração t para envio ao oráculo (tamanho b). | `domain.pool.QueryBatch` |
| **LearningCurve** | Série (|L_t|, métrica) registrada ao longo de um Run. | `domain.metrics.LearningCurve` |
| **LCE** | Learning Curve Efficiency: AUC_real/AUC_ideal (Simpson; trapézio com 2 pontos). | `domain.metrics.lce` |

## Domain Services (puros)

| Termo | Definição | Código |
|---|---|---|
| **QueryStrategy** | Função pura que ordena instâncias por informatividade a partir de distribuições de probabilidade: `entropy`, `least_confidence`, `smallest_margin`, `random`, `hybrid`. | `domain.strategies` |
| **PhasePolicy** | Política de transição de fases do FALCO (1→2→3): estabilização do macro-F1 por p iterações, piso de orçamento para a Fase 3. | `domain.phases.PhasePolicy` |
| **DRI-SL** | Algoritmo de cold start: clusterização semântica + seleção intra-cluster por variedade lexical. Nome único (o legado "DRI-Cluster" foi absorvido). | `domain.coldstart` (lógica) + `EmbedderPort` |

## Ports (interfaces; implementadas por adapters)

| Port | Capacidade | Adapters previstos |
|---|---|---|
| **OraclePort** | Dado um batch de textos + CategorySchema, retorna Annotations. | `OpenAIOracle`, `GeminiOracle`, `OllamaOracle`, `AnthropicOracle`, `SimulatedOracle(noise=ε)` |
| **ClassifierPort** | fit(L) / predict_proba(textos). | `BertimbauClassifier`, `BinaryVectorizerClassifier` (PVBin) |
| **EmbedderPort** | Textos → vetores densos (para DRI-SL). | `SentenceTransformerEmbedder` |
| **ExperimentLogPort** | Persistência de config, iterações, artefatos. | `JsonlExperimentLog` |
| **DatasetPort** | Carrega o dataset rotulado e produz partições T/V/U estratificadas. | `CsvDataset` |

## Convenções FALCO

- **Fase 1**: cold start DRI-SL + oráculo LLM Inicial → produz L0.
- **Fase 2**: QueryStrategy de incerteza + oráculo LLM Inicial.
- **Fase 3**: mesma estratégia + oráculo LLM Avançado (pós-estabilização).
- **RS/US**: baselines com mesma infraestrutura, trocando apenas a estratégia.
- **Experimentos numerados**: E0 (avaliação do oráculo), E1 (estratégias/batch com PVBin),
  E2 (épocas do BERTimbau), E3 (FALCO vs RS vs US), E4 (robustez a ruído do oráculo).
