# data/

## dataset.csv — Retail Product Description-Ptbr

Dataset público de descrições de produtos de e-commerce em português brasileiro,
coletado de 18 grandes varejistas (Darú, 2022; Darú, 2024).

- **250.221 instâncias** (versão corrigida; original com 250.365 em `data_old/` — ver `CORRECTIONS.md`)
- Colunas: `nm_item` (descrição curta do produto), `nm_product` (categoria/rótulo-ouro)
- Características: textos curtos (~20–40 caracteres), abreviações, forte
  desbalanceamento de classes
- Origem: repositório legado `GHDaru/activetextclassification` (mesmo arquivo)

É o dataset único de todos os experimentos (E0–E4). Partições T/V/U e amostras
derivadas são geradas pelos runners a partir dele (com semente fixa) e **não**
são versionadas.
