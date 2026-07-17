# Guia dos experimentos (E0, E0-P, E1/E4, P1/P2, E2/E3)

Como executar, parametrizar e reproduzir cada experimento da tese. Regra de
ouro (constituição do projeto): **nenhum número sem artefato rastreável** — todo
resultado citado na tese aponta para um JSON/JSONL versionado neste repositório.

Pré-requisitos comuns: `data/dataset.csv` na raiz (colunas `nm_item`,
`nm_product`) e `.env` com as chaves dos provedores usados (ver
`docs/biblioteca.md`). Execute sempre da RAIZ do repositório.

## Mapa geral

| Exp. | Pergunta | Runner | Config | Artefatos |
|---|---|---|---|---|
| E0 | Quais LLMs servem de oráculo? (acurácia, custo, erro, instrumento) | `experiments/e0/run_e0.py` | `experiments/e0/config*.json` | `experiments/e0/results/**` |
| E0-P | Prompt melhora o modelo fraco? | `experiments/e0p/run_e0p.py` | usa config do E0 | `experiments/e0p/results/**` |
| E1 | Qual estratégia de seleção? (oráculo perfeito) | `experiments/e1e4/run_sweeps.py` | flags no script | `experiments/e1e4/results/sweeps.jsonl` |
| E4 | Quanto o AL resiste a oráculo ruidoso? | idem (células e4) | idem | idem |
| P1/P2 | Replays de sensibilidade de L0 e do AG | `experiments/p1/replay_l0_sensitivity.py`, `replay_ga.py` | no script | `experiments/p1/results/*.jsonl` |
| E2/E3 | BERTimbau: épocas×\|L\| e FALCO integrado (GPU) | `experiments/e2e3/*` | flags | `experiments/e2e3/results/*.json` |

## E0 — avaliação de oráculos

```bash
python experiments/e0/run_e0.py --config experiments/e0/config.json
```

A config declara: semente, amostras (S-rand n=1000; S-strat 3/classe), lote de
gravação e a lista de oráculos. Cada oráculo é um spec:

```json
{"provider": "huawei-maas", "model": "deepseek-v4-pro", "mode": "json-prompt",
 "items_per_call": 25, "requests_per_minute": 3,
 "pricing_usd_per_mtok": [0.5, 0.5, 1.5], "samples": ["rand", "strat"]}
```

Parâmetros que importam: `mode` (`enum` = produção; `json-prompt` p/ provedores
sem structured output; `free` só p/ RQ4), `items_per_call` (lote de rotulagem —
amortiza o prefixo; calibrado no `config_calibration.json`),
`requests_per_minute` (respeite o provedor), `pricing_usd_per_mtok`
(entrada/entrada-cacheada/saída — para custo real).

Execução é **retomável**: anotações já gravadas em
`results/<amostra>/annotations_<oracle_id>.jsonl` são puladas. Providers sem
chave no ambiente são pulados com aviso.

Configs versionadas: `config.json` (OpenAI oficial), `config_full_paid_maas.json`,
`config_full_paid_openai.json`, `config_calibration.json` (lote 1/10/25),
`config_full_nvidia.json` (braço free via NVIDIA NIM — D-006),
`config_full_lowcost*.json` (histórico OpenRouter — D-005).

**Análises**: `python experiments/e0/analyze_e0.py` (IC de Wilson, McNemar
pareado com binomial exato <25 discordâncias, custo/1k, gate pré-registrado) e
`python experiments/e0/analyze_noise_impact.py` (sensibilidade ao ruído do
gabarito: excluir conflitantes × multi-gold). Anatomia de erros:
`experiments/e0/results/ANALISE_ERROS.md`.

## E0-P — ablação de prompt no modelo fraco

```bash
python experiments/e0p/run_e0p.py       # v4a e v4b × {rand, strat}, n=500 pareado
python experiments/e0p/analyze_e0p.py   # McNemar exato v3×v4a×v4b → results/analysis.json
```

Variantes em `src/activelearning/adapters/oracles/prompt.py`: `v4a` (10 regras
de fronteira derivadas da anatomia de erros) e `v4b` (v4a + 10 exemplos
INVENTADOS — desenho anti-vazamento D-004). O v3 reutiliza as anotações oficiais
do E0 (mesmo instrumento, mesmos itens).

## E1/E4 — estratégias e robustez a ruído (PVBin + oráculo simulado)

```bash
python experiments/e1e4/run_sweeps.py        # 104 células: e1 + e1b + e4
python experiments/e1e4/analyze_e1e4.py      # agregação + Wilcoxon → results/analysis.json
```

Desenho: pool deduplicado de 20k, orçamento 3.000, lote 100, 8 sementes/célula.
E1 = 5 estratégias com ruído 0; E1b = ablação de lote (entropia, b∈{50,200});
E4 = {entropia, aleatória} × ε∈{0.1,0.2,0.4}. O teto supervisionado
(`results/baseline.json`) treina com o pool inteiro e normaliza a LCE.
Wilcoxon pareado por semente — com n=8, o menor p possível é 0,0078.

## P1/P2 — replays de validação independente

```bash
python experiments/p1/replay_l0_sensitivity.py   # 15 tamanhos × 10 reps (D-002)
python experiments/p1/replay_ga.py               # AG 2×2, pop 30, 40 ger., anticircularidade
```

Convergência com os originais e o achado de circularidade: ver
`tesedaru/docs/convergencia-replays.md` e Cap. 4 da tese.

## E2/E3 — BERTimbau (bloco H; GPU)

```bash
python experiments/e2e3/run_smoke_cpu.py                 # valida a cadeia (CPU ok)
python experiments/e2e3/train_full.py --epochs 3         # teto supervisionado (GPU)
# Colab/TPU: experiments/e2e3/bertimbau_colab_tpu.ipynb (instruções no notebook)
```

Roteiro completo (grade épocas×|L|×sementes, estimativas por hardware):
`docs/bertimbau-local-passo-a-passo.md`.

## Reprodutibilidade

- Sementes explícitas em toda célula; 8 sementes nas comparações inferenciais.
- Partições imutáveis; teste NUNCA participa de decisão de laço.
- Anotações LLM com proveniência completa (oracle_id, prompt_version, tokens,
  custo, latência) em JSONL retomável.
- Decisões de desenho numeradas (D-001…D-006) em `tesedaru/docs/decisoes.md`.
