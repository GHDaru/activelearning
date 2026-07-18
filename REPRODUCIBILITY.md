# Guia de Reprodutibilidade / Reproducibility Guide

> **EN summary**: this repository contains ALL code, data (11 MB CSV, versioned
> in-repo) and result artifacts (~13 MB of JSON/JSONL) needed to reproduce every
> number reported in the PhD thesis "Aprendizado Ativo com Modelos de Linguagem
> para Textos Curtos em Português" (FALCO framework, PPGMNE/UFPR). Each
> experiment below lists the exact command, seeds, expected artifacts, runtime
> and API cost. Portuguese follows.

Princípio da tese (constituição do projeto): **nenhum número sem artefato
rastreável**. Toda tabela e figura da tese aponta para um JSON/JSONL versionado
aqui, produzido por um comando único e parametrizado listado abaixo.

## 1. Ambiente

```bash
git clone https://github.com/GHDaru/activelearning.git
cd activelearning
python3 -m venv .venv && source .venv/bin/activate
pip install -e . scikit-learn scipy matplotlib
pip install openai                       # experimentos com oráculo LLM real
pip install sentence-transformers        # encoder SBERT do DRI-SL (opcional)
pip install torch transformers           # BERTimbau (E2/E3; GPU recomendada)
python -m pytest tests/ -q               # 70 testes; roda offline, sem chaves
```

Testado com Python 3.11. A suíte de testes NÃO requer rede, GPU nem chaves.

## 2. Dados

- `data/dataset.csv` (11 MB, versionado): 250.365 descrições de produtos de
  varejo em português, 621 categorias (após filtro de suporte mínimo),
  colunas `nm_item` (texto) e `nm_product` (rótulo). Proveniência e correções:
  `data/README.md` e `data/CORRECTIONS.md`; original preservado em
  `data/data_old/`.
- Auditoria de ruído do gabarito (conflitos censados, análise multi-gold):
  Cap. 3 da tese + `experiments/e0/analyze_noise_impact.py`.
- Partições: derivadas por semente DENTRO de cada runner (nunca à mão);
  deduplicação por texto antes de qualquer divisão pool/validação/teste.

## 3. Credenciais (apenas p/ oráculos LLM reais)

`.env` na raiz (gitignored — NUNCA commitar): `OPENAI_API_KEY`,
`OPENROUTER_API_KEY`, `MAAS_API_KEY`(+`MAAS_BASE_URL`), `NVIDIA_API_KEY`.
Providers sem chave são pulados com aviso. Experimentos com `SimulatedOracle`
reproduzem-se 100% offline.

**Aviso de determinismo**: toda medição de oráculo LLM é uma fotografia da
tripla *(modelo, provedor, data)* — temperatura 0 não garante estabilidade
entre versões/provedores (o E0 demonstra o MESMO modelo divergindo entre dois
serviços de serving, p<0,001). As anotações originais estão versionadas em
JSONL com proveniência completa justamente para permitir re-análise sem
re-consulta.

## 4. Mapa de reprodução (experimento → comando → artefato)

| Exp. | Comando (da raiz) | Semente(s) | Artefatos | Duração / custo |
|---|---|---|---|---|
| E0 oráculos | `python experiments/e0/run_e0.py --config experiments/e0/config.json` (+ configs `_full_paid_maas`, `_full_paid_openai`, `_full_nvidia`) | 42 | `experiments/e0/results/{rand,strat}/annotations_*.jsonl`, `report_*.json`, `e0_summary.json` | horas (rate limits); ~US$ 4 total |
| E0 análise | `python experiments/e0/analyze_e0.py` | — | `e0_mcnemar.json`, `e0_table.json` | s |
| E0 ruído gabarito | `python experiments/e0/analyze_noise_impact.py` | — | `noise_impact.json` | s |
| E0-P prompts | `python experiments/e0p/run_e0p.py` depois `analyze_e0p.py` | 42 | `experiments/e0p/results/**`, `analysis.json` | ~2h; ~US$ 0,10 |
| E1/E1b/E4 | `python experiments/e1e4/run_sweeps.py` depois `analyze_e1e4.py` | 0–7 | `sweeps.jsonl` (104 células), `baseline.json`, `analysis.json` | ~9h CPU; US$ 0 |
| P1/P2 replays | `python experiments/p1/replay_l0_sensitivity.py` e `replay_ga.py` | múltiplas | `experiments/p1/results/*.jsonl` | ~2h CPU |
| Ciclo E2E real | `python experiments/e5cycle/run_cycle.py --classifier both --budget 15000 --pool-size 50000 --val-size 2000 --test-size 2000 --items-per-call 50 --cache experiments/e5cycle/results/annotation_cache_nemotron.jsonl --tag _b15k` | 42 | `cycle_{pvbin,sgd}_b15k.json` + records + cache | ~1h; US$ 0 (NIM) |
| Calibração de lote | `python experiments/e5cycle/calibrate_batch.py` | 42 | `calibration_b20_b50.json` | ~5 min; US$ 0 |
| E6 interna×externa | `python experiments/e6population/run_population_curve.py --classifier both --strategy {entropy,random,drisl}` | 42 | `popcurve_{clf}_{estr}.jsonl` + summaries | ~2-6h CPU/braço; US$ 0 |
| E2/E3 BERTimbau | `python experiments/e2e3/run_smoke_cpu.py`; `train_full.py`; notebook `bertimbau_colab_tpu.ipynb` | 42 (0–7 no E2) | `experiments/e2e3/results/*.json` | GPU: min–h |
| Figuras | `python experiments/plots/make_figures.py` | — | `experiments/plots/figures/*.{pdf,png}` | s |

Todos os runners com oráculo real são **retomáveis** (JSONL append por
instância) e os de laço longo têm **retomada por arquivo de estado**. O
`CachedOracle` garante que nenhuma instância é paga duas vezes.

## 5. Onde cada número da tese mora

- Cap. 4 (P1/P2): `experiments/p1/results/` + tabelas originais do draft
  (repositório `Tese-Vers-o-Draft`, read-only).
- Cap. 5 (E0/E0-P/E1/E4): `experiments/e0/results/`, `e0p/results/analysis.json`,
  `e1e4/results/analysis.json`.
- Apêndice A7 / ciclo e curvas populacionais: `e5cycle/results/`,
  `e6population/results/`.
- Decisões de desenho numeradas (D-001…D-007) e diário de execução:
  repositório da tese (`tesedaru/docs/decisoes.md`, `diario.md`).

## 6. Publicação e citação (roteiro)

1. **Zenodo (DOI)**: ativar a integração GitHub→Zenodo e criar um *release*
   `v1.0-tese` deste repositório — o Zenodo arquiva o snapshot (código + dados
   + artefatos, ~40 MB) e emite DOI citável. Repetir para o repositório da
   tese se desejado.
2. **Papers with Code**: após publicar a tese/artigo, criar a entrada em
   paperswithcode.com vinculando este repositório (o site aceita teses e
   preprints; a ligação é feita pela URL do PDF/arXiv + URL do GitHub).
   Categorias sugeridas: *Active Learning*, *Text Classification*.
3. **Hugging Face Datasets** (opcional): espelhar `data/dataset.csv` como
   dataset público com o mesmo README de proveniência — aumenta a
   descobribilidade e dá visualizador de dados gratuito.
4. Arquivo `CITATION.cff` na raiz (a criar no release) com a referência da
   tese, para o botão "Cite this repository" do GitHub.

## 7. O que NÃO é reproduzível bit a bit

- Respostas de oráculos LLM comerciais (ver aviso da Seção 3) — por isso as
  anotações originais são artefatos de primeira classe.
- Custos em US$ (preços de API datados de jul/2026; as razões entre modelos
  são mais estáveis que os valores absolutos).
- Tempos de parede (dependem de hardware e rate limits vigentes).
