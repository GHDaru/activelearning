# Papers with Code — material para submissão

Copie/cole nos campos do formulário em <https://paperswithcode.com>. Ver o fluxo
em [`../DEPLOY_AND_PUBLISH.md`](https://github.com/GHDaru/activelearning/blob/main/DEPLOY_AND_PUBLISH.md) (seção 4).

## Título
FALCO: Aprendizado Ativo com Oráculos LLM para Classificação de Texto Curto em Português

## Abstract (rascunho — ajuste ao paper/tese final)
Classificar texto curto de varejo (descrições de cupom fiscal, 4–50 caracteres,
caixa alta, abreviação agressiva) em centenas de categorias de cauda longa é
caro pelo custo de rotulagem, não pela capacidade do classificador. Propomos o
FALCO (Framework de Aprendizado Ativo com LLM para texto CurtO), que usa LLMs
justamente nas fases em que o aprendizado ativo clássico é mais frágil: o *cold
start* (via DRI-SL, construção do conjunto inicial sem nenhum rótulo, combinando
densidade semântica e variedade lexical) e a rotulagem (via oráculos LLM
progressivos, com custo instrumentado). Avaliamos em uma base pública de ~250
mil descrições e 621 categorias em português, com instrumentação estatística
declarada (Wilson, McNemar, Wilcoxon, bootstrap). Achados: DRI-SL supera o
melhor indivíduo de uma busca supervisionada sem usar rótulos; oráculos LLM
atingem um platô de 77–83% de acurácia com spread de custo de 26× dentro de
empate estatístico; a autoavaliação enviesa a decisão de liberação em direções
opostas conforme a métrica; e a hipótese central (≥95% do Macro F1 com ≤30% dos
rótulos) é refutada em 30% mas sustentada a partir de ~50% do pool, com o
gargalo no critério de parada — não no oráculo nem na seleção.

## Tasks (marcar)
- Active Learning
- Text Classification
- Short Text Classification
- (opcional) Low-Resource / Portuguese NLP

## Methods / keywords
Active Learning, LLM-as-oracle, Cold Start, DRI-SL, Uncertainty Sampling,
Cost-aware annotation, Short Text, Portuguese.

## Code
- Repositório oficial: https://github.com/GHDaru/activelearning
  (biblioteca `falco-active-learning` no PyPI + interface FlowBuilder)
- Marcar como **official implementation**.

## Dataset
- Nome: Classificação de Produtos — Varejo CPG PR/BR
- DOI: https://doi.org/10.34740/kaggle/dsv/4265348
- Kaggle: https://www.kaggle.com/datasets/gilsileydaru/classificao-produtos-varejo-cpg-prbr
- ~250 mil descrições, 621 categorias, português (BR).

## Paper URL (escolha uma)
- DOI da tese no Zenodo (após o passo 3 do guia), ou
- arXiv (se publicar um dos artigos A1–A5 / a tese).

## Resultados-chave (para a descrição / eventual leaderboard)
| Achado | Número |
|--------|--------|
| Platô de acurácia dos oráculos (621 classes) | 77–83% |
| Spread de custo em empate estatístico | 26× |
| Viés de acurácia interna (E6, 8 sementes) | −17,1 ± 1,0 p.p. |
| Piso em que a hipótese se sustenta | ~50% do pool |
| Melhor braço (E35, 70%) vs. supervisão completa | supera em Acc e Macro F1 |
