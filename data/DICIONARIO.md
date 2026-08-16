# Dicionário de dados — dataset.csv (Retail Product Description-Ptbr)

> Companheiro do `README.md` (visão geral) e do `CORRECTIONS.md` (auditoria e
> correções). Este arquivo documenta as colunas, a normalização de rótulos e as
> duas visões derivadas usadas pelos experimentos. Toda contagem daqui é
> verificável por `python scripts/check_dataset.py` (sem dependências de ML).

## Identidade do arquivo

| Campo | Valor |
|---|---|
| Caminho | `data/dataset.csv` |
| Linhas de dados (sem cabeçalho) | **250.221** |
| md5 | `0682ee5ba077c180fb7a727fb200f154` |
| Origem | repositório legado `GHDaru/activetextclassification` (mesmo arquivo), corrigido em 16/07/2026 |
| Versão anterior | `data_old/dataset_original.csv` (250.365 linhas); diff exato em `data_old/removed_inativo.csv` (144 linhas do rótulo operacional `inativo` — ver `CORRECTIONS.md`) |

## Colunas

| Coluna | Tipo | Descrição | Características |
|---|---|---|---|
| `nm_item` | texto | Descrição curta do produto como cadastrada pelo varejista | 4–50 caracteres (mediana 32); abreviações agressivas (`CV BR LT 350`), maiúsculas, unidades coladas |
| `nm_product` | texto | Categoria do produto (rótulo-ouro) | 794 valores distintos crus; forte desbalanceamento (`biscoito` = 14.292 linhas; 79 classes com 1 linha) |

## Normalização de rótulos

Função canônica: `normalize_label` em
`src/activelearning/domain/instances.py` — minúsculas, remoção de acentos
(NFKD sem combinantes), espaços colapsados. Sobre esta versão do CSV a
normalização **não funde nenhum par de rótulos**: 794 crus → 794 normalizados
(a checagem executável falha se isso mudar).

## Visões derivadas (as duas contagens citadas na tese)

### 1. Base experimental E5/E6/E3' — 231.490 textos / 714 classes

Receita (`load_base` em `experiments/e2e3/run_e3prime.py`, idêntica no E5/E6):

1. normalizar `nm_product` com `normalize_label`;
2. manter classes com **≥ 2** ocorrências → 250.142 linhas, 715 classes;
3. deduplicar por texto (`strip().lower()`, primeira ocorrência vence)
   → **231.490 textos**;
4. embaralhar com `random.Random(42)` (DATA_SEED fixo).

Classes na base final: **714**, não 715 — a classe `pomada massageadora`
tem 2 ocorrências, mas ambos os textos são duplicatas de linhas anteriores
do CSV, e o passo 3 a elimina por inteiro. O particionamento derivado é
pool = 50.000 primeiros, holdout do ciclo = 4.000 seguintes (val 2k + teste
2k), população reservada = 177.490 restantes.

### 2. CategorySchema do oráculo — 620 + `_rare_` = 621

Receita (`load_rows_and_schema` em
`src/activelearning/adapters/datasets/retail_csv.py`, usada pelo E0 e pela
API):

1. descartar linhas com texto/rótulo vazio e o rótulo operacional `inativo`
   (`exclude_labels`, hoje inócuo — a base corrigida já não o contém);
2. contar ocorrências do rótulo **cru** e manter os com
   `min_samples_per_class` **≥ 5** → **620** rótulos;
3. `CategorySchema.from_raw(..., include_rare=True)` normaliza cada rótulo e
   acrescenta `_rare_` → **621 valores** no `enum` enviado ao oráculo LLM
   (saída estruturada com `strict=True`).

Rótulos fora do schema no gold viram `_rare_` na instância
(`schema.validate(label) or Label("_rare_")`). Observação: o corte ≥ 5 do
schema conta o rótulo cru e o corte ≥ 2 da base experimental conta o rótulo
normalizado; nesta versão do CSV a distinção não altera contagens (não há
fusões na normalização), mas as receitas são independentes por construção.

## Invariantes verificáveis

`python scripts/check_dataset.py` prova, a partir do CSV publicado:

- 250.221 linhas de dados, colunas `nm_item,nm_product`, zero `inativo`;
- 794 rótulos crus = 794 normalizados (nenhuma fusão);
- visão 1 reproduz 231.490 textos / 714 classes (e o porquê do 715→714);
- visão 2 reproduz o schema de 621 = 620 + `_rare_`.
