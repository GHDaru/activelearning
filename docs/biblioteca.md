# Guia da biblioteca `activelearning`

Como usar a biblioteca em código próprio: classificadores, estratégias, oráculos,
laço de AL, runner FALCO e saneamento de dados. Todos os exemplos são executáveis
a partir da raiz do repositório com `PYTHONPATH=src` (ou `pip install -e .`).

## Instalação

```bash
git clone https://github.com/GHDaru/activelearning.git
cd activelearning
python3 -m venv .venv && source .venv/bin/activate
pip install -e .            # ou: uv sync --all-extras
# extras conforme o uso:
pip install scikit-learn scipy              # núcleo experimental
pip install openai                          # oráculos LLM (OpenAI/MaaS/OpenRouter/NIM)
pip install sentence-transformers           # encoder SBERT do DRI-SL
pip install torch transformers              # classificador BERTimbau
pip install fastapi uvicorn sqlmodel python-multipart httpx  # FlowBuilder API
```

Credenciais: crie um `.env` na raiz (NUNCA commitado — está no `.gitignore`):

```bash
OPENAI_API_KEY=...
OPENROUTER_API_KEY=...
MAAS_API_KEY=...          # + MAAS_BASE_URL se diferente do padrão
NVIDIA_API_KEY=...
DATABASE_URL=...          # opcional; sem ela o FlowBuilder usa SQLite local
```

## Conceitos do domínio (`activelearning.domain`)

- `Instance(id, text, gold_label)` — um texto candidato a rotulagem; em simulação
  carrega o rótulo-ouro.
- `Label(value)` — rótulo **normalizado** (minúsculas, sem acentos, espaços
  colapsados) via `normalize_label`.
- `CategorySchema` — o espaço FECHADO de categorias; fonte única do enum dos
  oráculos.
- `Annotation` — rótulo atribuído por um oráculo, com proveniência
  (`oracle_id`, `prompt_version`, usage/custo).

## Classificadores (porta `TaskClassifier`)

API comum: `fit(texts, labels)`, `predict(texts)`, `predict_proba(texts)`
(colunas na ordem de `classes_`).

### PVBin — leve, determinístico, segundos de treino

```python
from activelearning.adapters.classifiers.pvbin import PVBinClassifier

clf = PVBinClassifier()            # binário, (1,2)-gramas, sem norma (config da dissertação)
clf.fit(["arroz branco 5kg", "feijao preto 1kg"], ["arroz", "feijao"])
print(clf.predict(["arroz agulhinha"]))          # ['arroz']
print(clf.predict_proba(["arroz agulhinha"]))    # softmax sobre escores protótipo·consulta
```

### BERTimbau — forte, requer GPU para escala (bloco H)

```python
from activelearning.adapters.classifiers.bertimbau import BertimbauClassifier

clf = BertimbauClassifier(epochs=3, batch_size=64, max_length=32, seed=42)
clf.fit(train_texts, train_labels)   # fine-tuning do checkpoint neuralmind
pred = clf.predict(test_texts)
```

Guia completo de execução local/Colab: `docs/bertimbau-local-passo-a-passo.md`
e `experiments/e2e3/bertimbau_colab_tpu.ipynb`.

## Cold start sem rótulos: DRI-SL

```python
from activelearning.adapters.strategies.drisl import drisl_select, SbertEncoder, TfidfSvdEncoder

encoder = SbertEncoder()             # paraphrase-multilingual-MiniLM-L12-v2 (CPU ok)
# encoder = TfidfSvdEncoder()        # alternativa leve p/ testes, sem download
res = drisl_select(texts, target_size=500, encoder=encoder, seed=42)
l0_indices = res.indices             # índices do pool escolhidos p/ rotular primeiro
```

Alocação proporcional por cluster k-means (densidade semântica) + novidade
lexical intra-cluster (termos novos no perfil TF-IDF). Determinístico por semente.

## Oráculos (porta `OraclePort`)

Todos os provedores saem da mesma factory usada nos experimentos:

```python
from activelearning.adapters.oracles.factory import build_oracle

oracle = build_oracle({"provider": "simulated", "noise": 0.1, "seed": 42})  # offline
oracle = build_oracle({"provider": "openai", "model": "gpt-4o-mini",
                       "mode": "enum", "items_per_call": 10})
oracle = build_oracle({"provider": "huawei-maas", "model": "deepseek-v4-flash",
                       "mode": "json-prompt", "items_per_call": 25,
                       "requests_per_minute": 3})
oracle = build_oracle({"provider": "nvidia", "model": "nvidia/nemotron-3-ultra-550b-a55b",
                       "mode": "json-prompt", "items_per_call": 10})
annotations = oracle.annotate(instances, schema)
```

Modos de instrumento (RQ4 do E0): `enum` (saída estruturada restrita ao schema —
modo de produção), `json-prompt` (JSON sem restrição de enum; p/ provedores sem
structured output), `free` (texto livre; só para medir o efeito do instrumento).
Variantes de prompt do E0-P: `prompt_variant="v4a"|"v4b"` nos oráculos OpenAI.

## Laço de AL completo

```python
from activelearning.application.run_active_learning import run_active_learning

result = run_active_learning(
    pool=pool_instances, test=test_instances, schema=schema,
    classifier_factory=PVBinClassifier,
    oracle=oracle,
    strategy="entropy",        # entropy | least_confidence | smallest_margin | random | hybrid
    budget=3000, batch_size=100, initial_size=100, seed=42,
    initial_indices=l0_indices,          # injete o L0 do DRI-SL aqui (senão: aleatório)
    baseline_performance=0.54,           # teto p/ normalizar a LCE (opcional)
    output_path=Path("out/curva.jsonl"), # curva persistida ponto a ponto
)
print(result.lce, result.final_macro_f1, result.curve[:3])
```

Regras: o conjunto de teste NUNCA entra nas decisões do laço; L0 vazio com
oráculo que não rotula nada levanta `RuntimeError` acionável.

## Runner FALCO (fases + troca de oráculo)

```python
from activelearning.application.run_falco import run_falco

result = run_falco(
    pool=pool, validation=val, test=test, schema=schema,
    classifier_factory=PVBinClassifier,
    oracle_initial=build_oracle({"provider": "huawei-maas", "model": "deepseek-v4-flash",
                                 "mode": "json-prompt", "items_per_call": 25}),
    oracle_advanced=build_oracle({"provider": "huawei-maas", "model": "deepseek-v4-pro",
                                  "mode": "json-prompt", "items_per_call": 25}),
    drisl_selector=lambda texts, k: drisl_select(texts, k, encoder).indices,
    budget=3000, batch_size=100,
    stagnation_patience=5, stagnation_eps=1e-3,   # transição por estagnação NA VALIDAÇÃO
)
```

Fases: (1) DRI-SL rotula `b0 = 1%·B` com o LLM Inicial; (2) seleção por
incerteza até o Macro-F1 **em validação** estagnar (paciência 5, ε=1e-3);
(3) troca para o LLM Avançado até o orçamento. O teste fica intocado até a
avaliação final.

## Saneamento de dados

```python
from activelearning.application.sanitize_dataset import sanitize_csv

report = sanitize_csv(Path("original.csv"), Path("sanitized.csv"),
                      text_column="nm_item", label_column="nm_product",
                      operational_labels=("inativo",))
print(report.to_dict())   # linhas mantidas/removidas, censo de conflitos e duplicatas
```

Política (a mesma da tese): rótulos operacionais são REMOVIDOS; conflitos e
duplicatas são CENSADOS e mantidos (representam o ruído real do domínio) — o
relatório permite ao usuário decidir o contrário.

## Testes

```bash
python -m pytest tests/ -q     # 67 testes; suíte roda offline (sem chaves)
```
