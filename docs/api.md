# Referência da API

Documentação gerada automaticamente a partir das *docstrings* dos módulos
públicos. Para o uso guiado, veja o [início rápido](quickstart.md) e o
[guia da biblioteca](biblioteca.md).

## Domínio

### Instâncias e esquema

::: activelearning.domain.instances

### Anotações

::: activelearning.domain.annotation

### Métricas

::: activelearning.domain.metrics

## Ports (contratos)

### OraclePort

::: activelearning.ports.oracle

## Adapters — classificadores

### PVBin (protótipo TF-IDF)

::: activelearning.adapters.classifiers.pvbin

### SGD (TF-IDF + regressão logística)

::: activelearning.adapters.classifiers.sgd_text

### BERTimbau (transformer, classificador forte)

::: activelearning.adapters.classifiers.bertimbau

## Adapters — oráculos

### Oráculo simulado (offline)

::: activelearning.adapters.oracles.simulated_oracle

### Fábrica de oráculos

::: activelearning.adapters.oracles.factory

### Oráculo com cache

::: activelearning.adapters.oracles.cached

## Adapters — estratégias

### DRI-SL (cold start sem rótulos)

::: activelearning.adapters.strategies.drisl

## Aplicação (casos de uso)

### Laço de aprendizado ativo

::: activelearning.application.run_active_learning

### Runner FALCO (ciclo completo)

::: activelearning.application.run_falco

### Avaliação de oráculo (E0)

::: activelearning.application.evaluate_oracle

### Saneamento de dataset

::: activelearning.application.sanitize_dataset
