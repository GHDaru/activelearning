# Dicionário de dados — Retail Product Description-Ptbr (`dataset.csv`)

Item `dicionario-dados` do plano de revisão da tese
(`tesedaru/docs/records/plano-revisao.json`). Complementa `README.md`
(apresentação) e `CORRECTIONS.md` (auditoria e changelog 250.365 → 250.221).
Toda contagem abaixo é reproduzível por `scripts/check_dataset_invariants.py`
— rode-o antes de citar qualquer número desta página.

## Identidade do arquivo

| Campo | Valor |
|---|---|
| Arquivo | `data/dataset.csv` |
| md5 | `0682ee5ba077c180fb7a727fb200f154` |
| Linhas de dados (sem cabeçalho) | 250.221 |
| Versão | corrigida (remoção do rótulo operacional `inativo`, 144 linhas — ver `CORRECTIONS.md`); original com 250.365 em `data_old/` |
| Codificação | UTF-8, separador vírgula, aspas padrão `csv` |
| Origem | repositório legado `GHDaru/activetextclassification` (mesmo arquivo); publicação Kaggle pendente de licença (item `kaggle-licenca` do plano) |

## Colunas

| Coluna | Tipo | Descrição |
|---|---|---|
| `nm_item` | texto | Descrição curta do produto como digitada pelo varejista (~20–40 caracteres, abreviações, caixa alta frequente). É o texto classificado. |
| `nm_product` | texto | Categoria de produto atribuída pelo varejista — o rótulo-ouro da tarefa. |

## Normalização de rótulos (`normalize_label`)

Fonte: `src/activelearning/domain/instances.py`. Aplicada a TODO rótulo que
entra no domínio (classe `Label`): minúsculas, remoção de acentos (NFKD sem
combining) e colapso de espaços. No CSV corrigido a normalização não funde
classes: 794 valores distintos de `nm_product` antes e depois.

## CategorySchema — o enum fechado do oráculo (620 + `_rare_` = 621)

Fonte: `src/activelearning/adapters/datasets/retail_csv.py`
(`load_rows_and_schema`, `min_samples_per_class = 5`) +
`CategorySchema` (`include_rare = True`).

```
794 classes normalizadas
  → 620 classes com ≥ 5 ocorrências        (as demais 174 caem em _rare_)
  → + rótulo reservado `_rare_`
  = 621 valores no enum enviado aos oráculos LLM
```

`_rare_` é rótulo reservado do domínio (`RARE_LABEL`): não existe no CSV;
representa "classe rara/fora do esquema" na saída restrita do oráculo.

## Pipeline experimental E5/E6/E3′ (`load_base`) — 231.490 / 714

Fonte: `experiments/e2e3/run_e3prime.py` (`load_base`, `min_per_class = 2`,
`DATA_SEED = 42`). Cadeia de contagens, nesta ordem:

```
250.221 linhas do CSV
  → filtro: classe (normalizada) com ≥ 2 ocorrências  → 250.142 linhas (715 classes)
  → dedup por texto (strip + lower, 1ª ocorrência)    → 231.490 textos únicos
      (714 classes — uma classe perde todas as instâncias no dedup,
       pois seus textos duplicam textos anteriores de outra classe)
  → shuffle determinístico (semente de dados 42)
  → pool           = dedup[      :50.000]  (50.000 itens · 649 classes presentes)
  → holdout ciclo  = dedup[50.000:54.000]  (val 2k + teste 2k do ciclo real)
  → população      = dedup[54.000:      ]  (177.490 itens — avaliação reservada)
```

Notas de leitura:
- **714 vs 621**: são recortes diferentes e AMBOS corretos. 714 = classes da
  base experimental deduplicada (filtro ≥ 2); 621 = enum do oráculo
  (620 classes com ≥ 5 amostras + `_rare_`).
- **649** = classes efetivamente presentes no pool de 50k (é a contagem a que
  o artigo a5 se referia como "649 raw classes"; a correção do a5 é usar 714
  para a base e 649 para o pool).
- O filtro ≥ 2 é aplicado ANTES do dedup: no conjunto deduplicado a menor
  classe pode ter 1 ocorrência.
- O particionamento é fixado por `DATA_SEED = 42` independentemente da semente
  de treino (`--seed`), para manter os braços pareáveis entre sementes.

## Invariantes executáveis

`python3 scripts/check_dataset_invariants.py` verifica, com as funções REAIS
do pipeline (não cópias), TODA contagem citada nesta página: md5 · 250.221
linhas · 794 classes normalizadas · schema 621 com `_rare_` (620 + 174 em
`_rare_`) · filtro ≥2 (250.142 linhas, 715 classes) · dedup 231.490 · 714
classes · pool 50.000 (649 classes) · partições que somam o dedup · população
177.490. Sai com código 0 (tudo verde) ou 1 (violação), imprimindo cada
invariante.
