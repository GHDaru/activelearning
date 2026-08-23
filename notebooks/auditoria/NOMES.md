# Convenção de nomes — experimentos e artefatos

> Proposta do `executor01` ao `principal`, a pedido do autor (2026-08-17).
> Duas dores distintas, resolvidas juntas porque a segunda quase apagou um
> número da tese.

## Dor 1 — `E0`, `E4`, `E3′` não dizem nada

Os códigos são úteis para citar no texto (`\ref{sec:res-e6}`) e péssimos para
navegar num repositório. Quem abre `experiments/e6population/` não sabe o que
vai encontrar; quem procura "onde está o experimento do prompt" não tem por
onde começar.

**Regra**: o nome legível é o principal; o código continua existindo como
apelido, porque a tese inteira cita por ele e quebrar essa ponte perderia a
rastreabilidade.

| Código | Nome legível | O que o experimento responde |
|---|---|---|
| E0 | **escolha-do-oraculo** | qual LLM anota melhor, a que custo e com que perfil de erro |
| E0-P | **efeito-do-prompt** | quanto do resultado é do modelo e quanto é da forma de perguntar |
| E1 | **estrategias-de-selecao** | qual estratégia de seleção vence com oráculo perfeito |
| E4 | **robustez-ao-ruido** | o laço aguenta um oráculo que erra? e que tipo de erro é benigno? |
| E5 | **ciclo-completo** | o pipeline ponta a ponta, com oráculo real, funciona? |
| E6 | **escala-populacional** | os seletores na base inteira, e o quanto a autoavaliação engana |
| E3′ | **classificador-forte** | o BERTimbau, treinado fora do laço, valida o que o pipeline entrega |
| P1/P2 | **conjunto-inicial** | composição e tamanho do L₀ importam? qual o teto? |

Aplicado **já** aos notebooks de auditoria (superfície minha):

```
notebooks/auditoria/escala-populacional.ipynb     (era e6-populacao)
notebooks/auditoria/classificador-forte.ipynb     (era e3prime-validacao)
```

As pastas `experiments/e6population/` etc. **não** foram renomeadas: são
superfície compartilhada, referenciadas por scripts, pela tese e pelos dois
outros agentes. A renomeação delas é decisão do `principal`, e se acontecer
tem de vir com um mapa de-para e um passe nos caminhos codificados.

## Dor 2 — o nome do artefato não diz o REGIME, e isso apaga número

Achado de 2026-08-17, e é grave. O padrão atual é:

```
e3prime_<braço>_s<semente>.json
```

Ele codifica a semente e mais nada. Só que o E3′ roda em **dois regimes**:

| Regime | Lote | Avaliação | Origem |
|---|---|---|---|
| **pareado** | 16 | amostra estratificada de 20.092 | é o que está publicado na tese |
| **canonico** | 128 | população inteira, 177.490 | é o comando da tarefa das sementes |

Os dois produzem `e3prime_D_s42.json`. No repositório esse arquivo é o do
regime pareado, com Macro F1 **0,4509** — o número citado no Cap. 5. A execução
canônica da semente 42 gera um arquivo de mesmo nome com **0,3691**. Copiar a
saída por cima **destrói o número da tese sem conflito, sem aviso e sem
rastro**.

**Regra**: o regime entra no nome.

```
<experimento>_<braço>_s<semente>_<regime>.json

classificador-forte_E35_s42_pareado.json
classificador-forte_E35_s42_canonico.json
```

**Convenção adotada (do `executor02`, mergeada na main em 17/08)**: o regime
publicado foi movido para `experiments/e2e3/results/legacy_s42_bs16_eval20k/` e
a raiz passou a guardar o canônico. Eu havia proposto o inverso — isolar o
canônico em `canonico/` — e converge para a dele: uma convenção só vale se for
uma. A minha pasta foi removida.

Isso resolve a perda de dado, mas **não** a ambiguidade: o nome
`e3prime_D_s42.json` continua sem dizer o regime, e quem o lê fora de contexto
não sabe qual dos dois tem na mão. A solução de verdade é o `run_e3prime.py`
escrever o regime no nome — mudança de superfície compartilhada, que precisa
preservar a retomada (hoje ela procura o nome antigo) e é decisão do
`principal`.

## Por que não renomear os arquivos publicados agora

Porque o Cap. 5 e os artefatos de análise (`mcnemar_s42.json`,
`bootstrap_f1_s42.json`) apontam para os nomes atuais. Renomear sem um ciclo
dedicado troca um problema silencioso por outro. A proposta é: o runner passa a
escrever com regime, os novos nascem certos, e uma migração dos antigos entra
como ciclo próprio com de-para registrado em ADR.

## Rebatismo E3′ → E3 (tarefa 20260822-1200) — mapeamento, não renome

O `principal` decidiu (aviso 2026-08-22T18:30, a partir de medição do
`revisor2`: 135 arquivos com `e3prime` no nome — 130 em
`experiments/e2e3/results/` — e 144 ocorrências em código/config em 27
arquivos) que o rebatismo **E3′ → E3** é só no texto visível da tese.
Identificadores internos são estáveis por desenho e **não mudam**:
renomeá-los quebraria os caminhos que o `revisor2` já citou ao verificar
números publicados — perda de rastreabilidade, não ganho.

**Mapa de/para (o único artefato desta tarefa):**

| No texto da tese (Cap. 3 em diante, por `revisor1`) | Nos artefatos/código deste repositório |
|---|---|
| **E3** (o experimento) | `e2e3` (diretório), `e3prime` (prefixo de arquivo/script: `run_e3prime.py`, `stats_e3prime.py`, `e3prime_<braço>_s<seed>.json`, `e3prime_<braço>_s<seed>_pred.json`) |
| braços A–E, régua D, E35 | mantidos idênticos nos dois lados — não são parte do rebatismo |

Nada em `experiments/e2e3/`, em `notebooks/auditoria/build_classificador-forte.py`
ou em `docs/records/rastreabilidade.json` foi renomeado por esta tarefa. Esta
tabela é a nota de equivalência completa.
