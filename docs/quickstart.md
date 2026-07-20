# Início rápido

## 1. Domínio + oráculo simulado (sem chaves)

Só o núcleo (`pip install falco-active-learning`). Cria um esquema fechado de
categorias e rotula um lote com o oráculo simulado:

```python
from activelearning.domain.instances import Instance, Label, CategorySchema
from activelearning.adapters.oracles.simulated_oracle import SimulatedOracle

# Conjunto FECHADO de rótulos válidos (a fonte do enum enviado aos LLMs)
schema = CategorySchema.from_raw(["cerveja", "sabao", "refrigerante"])

batch = [
    Instance(id="1", text="CERV BRAHMA LT 350ML", gold_label=Label("cerveja")),
    Instance(id="2", text="SAB OMO 1KG",          gold_label=Label("sabao")),
]

oracle = SimulatedOracle(noise=0.0)            # 0 ≤ ε < 1 (taxa de erro)
for inst, ann in zip(batch, oracle.annotate(batch, schema)):
    print(inst.text, "->", ann.label.value if ann.label else "INVÁLIDO")
```

Saída:

```
CERV BRAHMA LT 350ML -> cerveja
SAB OMO 1KG -> sabao
```

!!! tip "Respostas fora do esquema não são silenciadas"
    Se um oráculo devolver um rótulo fora do `CategorySchema`, a anotação vem com
    `label = None` (inválida) e é **contabilizada** — nunca aceita nem descartada
    em silêncio. É o Princípio III da constituição do projeto.

## 2. Um laço de AL completo (com classificador)

Requer o extra de classificadores: `pip install "falco-active-learning[classifiers]"`.
O laço parte de um `L0`, treina, seleciona os mais incertos, o oráculo rotula e
re-treina — até o orçamento:

```python
from activelearning.domain.instances import Instance, Label, CategorySchema
from activelearning.adapters.oracles.simulated_oracle import SimulatedOracle
from activelearning.adapters.classifiers.pvbin import PVBinClassifier
from activelearning.application.run_active_learning import run_active_learning

schema = CategorySchema.from_raw(["cerveja", "sabao", "refrigerante"])

# pool e teste são listas de Instance com gold_label (simulação)
pool  = [...]   # ex.: milhares de Instance
test  = [...]   # conjunto de avaliação

result = run_active_learning(
    pool=pool,
    test=test,
    schema=schema,
    classifier_factory=lambda: PVBinClassifier(),  # protótipo TF-IDF binário
    oracle=SimulatedOracle(noise=0.1),             # ε = 10% de ruído
    strategy="entropy",          # entropy | least_confident | smallest_margin | random | hybrid
    budget=1000,                 # nº de rótulos que o oráculo pode fornecer
    batch_size=100,
    initial_size=100,
    seed=42,
)

print("Macro F1 final:", result.final_macro_f1)
print("LCE (área sob a curva):", result.lce_macro_f1)
print("rótulos usados:", result.n_labeled)
```

O `ALResult` traz as curvas de aprendizado (`curve_macro_f1`, `curve_accuracy`),
a métrica **LCE**, o desempenho final e a contagem de rótulos — os mesmos
instrumentos usados nos experimentos da tese.

## 3. Trocar o oráculo simulado por um LLM real

Requer o extra `oracles` e uma chave no ambiente. O contrato é o mesmo
(`OraclePort.annotate(batch, schema)`), então o laço acima não muda — só o objeto
`oracle`:

```python
from activelearning.adapters.oracles.factory import build_oracle

oracle = build_oracle({"provider": "openai", "model": "gpt-4o-mini", "mode": "enum"})
# ... passe esse `oracle` para run_active_learning(...)
```

!!! warning "Custo e cache"
    Oráculos LLM cobram por token. O adapter OpenAI é desenhado para *prompt
    caching* (o prefixo com o enum de categorias é estável); o relatório reporta
    `cached_input_tokens` e `cache_hit_rate`. Comece sempre pelo `SimulatedOracle`
    para calibrar orçamento e lote.

Próximo: o [guia da biblioteca](biblioteca.md) detalha classificadores, DRI-SL
(cold start sem rótulos), oráculos e o runner FALCO.
