# Arquitetura — activelearning

DDD + Hexagonal (Ports & Adapters). O domínio é puro; tudo que toca o mundo externo
(LLMs, HuggingFace, sklearn, disco, rede) é adapter.

```
            ┌────────────────────────── driving adapters ──────────────────────────┐
            │  CLI de experimentos (experiments/e*/run_*.py)                       │
            │  FlowBuilder backend (FastAPI)  ← futuro, mesmo core                 │
            └───────────────┬───────────────────────────────────────────────────────┘
                            │ chama
            ┌───────────────▼───────────────┐
            │        application/           │   use cases: EvaluateOracle (E0),
            │  (orquestração, transações)   │   RunActiveLearning (E1/E3), RunFalco
            └───────────────┬───────────────┘
                            │ usa
            ┌───────────────▼───────────────┐
            │           domain/             │   Instance, Label, CategorySchema,
            │  entidades, VOs, serviços     │   Annotation, Budget, QueryBatch,
            │  puros (sem I/O, sem libs ML) │   strategies, PhasePolicy, LCE, DRI-SL
            └───────────────┬───────────────┘
                            │ define contratos
            ┌───────────────▼───────────────┐
            │            ports/             │   OraclePort, ClassifierPort,
            │     (Protocols/ABCs)          │   EmbedderPort, DatasetPort,
            └───────────────┬───────────────┘   ExperimentLogPort
                            │ implementados por
            ┌───────────────▼───────────────────────────────────────────────┐
            │                        adapters/                              │
            │  oracles/: OpenAI · Gemini · Ollama · Anthropic · Simulated   │
            │  classifiers/: Bertimbau (HF) · BinaryVectorizer (PVBin)      │
            │  embedders/: SentenceTransformer                              │
            │  storage/: JsonlExperimentLog · CsvDataset                    │
            └───────────────────────────────────────────────────────────────┘
```

## Decisões de projeto

1. **Estratégias de consulta são domínio, não port.** Entropia/margem/confiança são
   funções puras sobre distribuições de probabilidade (`numpy` permitido como exceção
   pragmática documentada — é a única dependência numérica do domínio).
2. **`CategorySchema` é a fonte única do `enum`.** O schema JSON enviado a qualquer LLM
   é gerado a partir dele (`CategorySchema.to_json_schema()`), garantindo o Princípio
   III da constituição. Rótulos fora do schema são registrados como `invalid_label`,
   nunca silenciosamente aceitos ou descartados.
3. **Oráculo simulado com ruído paramétrico** (`SimulatedOracle(noise=ε)`) é um adapter
   de primeira classe: viabiliza E1 (ε=0) e E4 (robustez) sem custo de API.
4. **FALCO é um use case, não uma classe-deus.** `RunFalco` compõe: DRI-SL (domínio) +
   PhasePolicy (domínio) + dois OraclePort (Inicial/Avançado) + um ClassifierPort.
   RS/US são o mesmo use case `RunActiveLearning` com estratégia diferente — o que
   garante comparação justa por construção.
5. **Logs de experimento são JSONL append-only** com config + git SHA + seeds no
   cabeçalho. Análise/figuras leem só desses artefatos (nunca de estado em memória).

## Mapa de portes do legado

| Legado (`activetextclassification`) | Destino | Estratégia |
|---|---|---|
| `GeneticL0Optimizer` (v4) | `application/optimize_l0.py` + domínio | Portar depois de E3 (resultados legados continuam válidos) |
| `cold_start` DRI-SL | `domain/coldstart.py` | Portar com testes (E2/E3 dependem) |
| `selection.py` | `domain/strategies.py` | Portado (funções puras) |
| `prompts.py` (schema SEM enum) | `domain/instances.CategorySchema.to_json_schema()` | **Corrigido** — origem do bug de medição do oráculo |
| `oracle/*` | `adapters/oracles/*` | Reescrever com saída estruturada restrita |
| `utils.calculate_lce` | `domain/metrics.py` | Portado (Simpson, def. única v5 da tese) |
| `ActiveLearner` | `application/run_active_learning.py` | Reescrever enxuto |

## Roadmap de experimentos

| Exp | Use case | Adapters necessários | Status |
|---|---|---|---|
| E0 | `EvaluateOracle` | oracles reais + CsvDataset + JsonlLog | **nesta entrega** (runner pronto; execução requer chaves de API) |
| E1 | `RunActiveLearning` | SimulatedOracle + BinaryVectorizer | próximo |
| E2 | script dedicado | Bertimbau | próximo |
| E3 | `RunFalco` + `RunActiveLearning` | Bertimbau + 2 oracles reais | alvo da defesa |
| E4 | `RunFalco` | SimulatedOracle(ε) + Bertimbau | condicional ao resultado de E0 |
