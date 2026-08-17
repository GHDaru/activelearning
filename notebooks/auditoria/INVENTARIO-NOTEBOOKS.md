# Etapa 1 — Inventário de todos os notebooks

> `executor01`, 2026-08-17. Tarefa
> `20260816-2205_principal_executor01_tarefa_rastreabilidade-numeros-e-notebooks`.
> Nada é apagado aqui: o que estiver morto ou duplicado é **listado** e vai ao
> `principal` para decisão.

## 1. Kernels no Kaggle (conta `ghdaru`)

| Kernel | Experimento | Papel | Estado | Ação proposta |
|---|---|---|---|---|
| `falco-e3prime-s42` | classificador-forte (E3′) | execução s42 canônica | COMPLETE | renomear p/ padrão |
| `falco-e3prime-s123` | classificador-forte | execução s123 canônica | COMPLETE | renomear |
| `falco-e3-semente-7` | classificador-forte | execução s7 canônica | COMPLETE | renomear |
| `falco-e3prime-semente-123` | classificador-forte | **versão anterior** da s123 | COMPLETE | **candidato a remoção** |
| `falco-auditoria-escala-populacional` | escala-populacional (E6) | auditoria | COMPLETE | já no padrão |
| `falco-auditoria-classificador-forte` | classificador-forte | auditoria | COMPLETE | já no padrão |
| `exercise-functions-and-getting-help` | — | tutorial Kaggle, 2021 | — | alheio à tese |
| `exercise-syntax-variables-and-numbers` | — | tutorial Kaggle, 2020 | — | alheio à tese |
| `eda-exploratory-data-analysis` | — | exploração, 2024 | — | alheio; verificar se é da tese |

**Três padrões de nome para o MESMO experimento**: `falco-e3-semente-7`,
`falco-e3prime-s42`, `falco-e3prime-semente-123`. É a bagunça que a tarefa
manda resolver.

**Padrão proposto** (a partir da sugestão do `principal`, com os nomes
legíveis de `NOMES.md`):

```
falco-<experimento>[-s<semente>]          execução
falco-auditoria-<experimento>             auditoria (reproduz número publicado)
```

De-para dos kernels de execução:

| Hoje | Proposto |
|---|---|
| `falco-e3-semente-7` | `falco-classificador-forte-s7` |
| `falco-e3prime-s42` | `falco-classificador-forte-s42` |
| `falco-e3prime-s123` | `falco-classificador-forte-s123` |
| `falco-e3prime-semente-123` | remover (duplicata anterior da s123) |

> Renomear kernel no Kaggle **cria um novo e deixa o antigo órfão** — o slug vem
> do título. Proposta: criar com o nome novo, conferir que roda, e só então
> pedir ao autor que apague o antigo. Nada some sem esse passo.

## 2. Notebooks em `activelearning`

| Arquivo | Experimento | Papel | Estado |
|---|---|---|---|
| `notebooks/auditoria/escala-populacional.ipynb` | E6 | auditoria | ativo, roda no Kaggle |
| `notebooks/auditoria/classificador-forte.ipynb` | E3′ | auditoria | ativo, roda no Kaggle |
| `experiments/e2e3/e3prime_kaggle.ipynb` | E3′ | execução (Kaggle) | ativo |
| `experiments/e2e3/e3prime_colab.ipynb` | E3′ | execução (Colab) | **superado** pelo do Kaggle |
| `experiments/e2e3/bertimbau_colab_tpu.ipynb` | E2/E3 | execução em TPU | **verificar se ainda vale** |

## 3. Notebooks em `tesedaru`

Nenhum. O repositório da tese não tem notebook — correto, é LaTeX.

## 4. O legado `GHDaru/activetextclassification` — 22 notebooks

Repositório **público, somente leitura**, anexado nesta sessão. É a "edição do
ano passado" a que o autor se referiu, e **é aqui que mora a evidência do
Capítulo 4** que eu havia classificado como inexistente.

### Correção de um achado meu

Eu havia registrado que o Cap. 4 "não é reproduzível a partir do repositório",
porque `activelearning/experiments/p1/results/` não existe. Está incompleto: a
evidência existe, **em outro repositório e sem referência cruzada nenhuma**.

| Onde | O quê |
|---|---|
| `examples/L0_experimento.ipynb` | célula 4 tem **`0.891` na saída gravada** — a acurácia de saturação do Cap. 4 — e gera a tabela LaTeX das estatísticas |
| `examples/data/sensibilidade/l0_random_impact_metrics_PVBin.xlsx` | **o artefato** por trás da Tabela de sensibilidade |
| `examples/data/sensibilidade/l0_random_impact_stats_table_PVBin.tex` | a tabela LaTeX gerada |
| `examples/L0_graficos_l0.ipynb` | as figuras do Cap. 4, lendo o xlsx acima |
| `examples/ag_*.ipynb` (5 notebooks) | o AG do pilar P2 |
| `examples/coldstart_*.ipynb` (2) | DRI-SL como *cold start* |
| `examples/oraculo*.ipynb` (3) | precursores do E0 |

Status correto do Cap. 4, portanto, não é `sem-evidencia` e sim
**`rastreado-em-repositorio-legado`**: existe, é recuperável, e **não está
citada em lugar nenhum** — nem no `REPRODUCIBILITY.md`, que aponta para um
`Tese-Vers-o-Draft` que não é este repositório.

### Ressalvas que o `principal` precisa levar ao autor

1. O legado é **somente leitura** nesta sessão (proxy anônimo). Para citá-lo
   como fonte de evidência da tese, ele precisa de um commit fixo referenciado.
2. Os notebooks do legado guardam saída **embutida no `.ipynb`**, não em
   artefato versionado à parte. É evidência, mas frágil: reexecutar sobrescreve.
3. Vários têm nome de rascunho — `oraculo copy.ipynb`, `temptestoraculo.ipynb`,
   `notebook2py.ipynb`. Não proponho tocar: repositório alheio à tarefa e
   read-only.

## Próximo passo desta etapa

Criar os kernels com o nome novo e conferir que rodam, antes de pedir a remoção
de qualquer coisa. Em paralelo, a Etapa 2 (varredura dos números) começa pelo
Cap. 5, que é o mais exposto e onde metade da varredura já está feita.
