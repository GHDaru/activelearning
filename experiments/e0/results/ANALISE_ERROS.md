# E0 — Anatomia dos erros do melhor oráculo (deepseek-v4-pro, T=0)

Gerado em 17/07/2026 a partir das anotações oficiais (S-rand n=1000; S-strat
n=1863). Script: reexecutável via trecho em `analyze_noise_impact.py` /
histórico da sessão; dados brutos nos JSONL de `results/`.

## Distribuição dos 324 erros na S-strat (acc 82,6%)

| Categoria de erro | % | Exemplo |
|---|---|---|
| **Categoria-irmã** (pred e gold compartilham palavra) | **31%** | `antisseptico bucal → enxaguante bucal`; `ervilha → ervilha em conserva`; `cadeira de praia → cadeira` |
| **Gold é classe guarda-chuva "outro …"** | 17% | gold `outro farma`, pred `xarope` (o modelo escolhe a específica; o catálogo usa a agregadora) |
| Pred é "outro …" | 7% | inverso do anterior |
| Envolve `_rare_` | 1% | `JOGO UNO → brinquedo` (gold `_rare_`) |
| Rótulo inválido (fora do schema) | ~1% | 4 casos em 1.863 |
| Erro semântico genuíno | ~restante (≈40%) | `STIKADINHO → bala` (gold: chocolate); `fisioterapia → outro condimento` |

Top confusões: `acessorio de audio → fone de ouvido` (um fone COM fio É um fone
de ouvido — a fronteira é convenção do catálogo), `gel antisseptico p/ maos →
alcool` (álcool gel é ambos), `preparo para suco → refresco`, `multiuso →
produto de limpeza de piso`, `lava roupas → detergente/lava louça`.

Casos de gold questionável fora do conjunto de conflitos: `CREME LEITE
PIRACANJUBA → gold "queijo"` (pred `creme de leite` está certa).

## Leitura central

A maioria dos erros do oráculo **não é ignorância sobre o produto — é
desconhecimento das convenções DESTE catálogo**: onde termina `enxaguante` e
começa `antisseptico bucal`, quando usar a classe guarda-chuva `outro …`, qual
granularidade o varejista escolheu. Essas fronteiras são regras da casa,
aprendíveis apenas com exemplos rotulados — exatamente o que o oráculo
zero-shot não tem.

## Por que o PVBin supervisionado chega a ~90% e o LLM zero-shot a ~82%

Não é contradição — são regimes de supervisão opostos:

| | PVBin (Daru, 2024) | Oráculo LLM (E0) |
|---|---|---|
| Rótulos vistos antes de prever | ~250 mil (treino supervisionado) | **zero** |
| Conhece as convenções do catálogo | Sim (aprendidas dos dados) | Não |
| Acurácia | 89,56% | 82,6% (v4-pro, multi-gold 82,9%) |
| **Macro F1** | **70,09%** | **~78–80%** (S-strat) |

Dois pontos:
1. **82,6% zero-shot está a 7 p.p. do teto supervisionado com 250 mil rótulos**
   — para um anotador que nunca viu um exemplo, é a medida da viabilidade do
   oráculo, não da sua inferioridade. A tese vive exatamente no meio: rotular
   ~30% do pool com esse oráculo e treinar um classificador que recupere ≥95%
   do teto.
2. **No Macro F1 a relação se INVERTE**: o LLM zero-shot (~0,78–0,80) supera o
   PVBin treinado (0,70) nas classes raras — o conhecimento de mundo compensa a
   falta de exemplos justamente onde o supervisionado sofre (poucas amostras).
   (Comparação indicativa: protocolos de avaliação distintos; o E1/E3 fará a
   comparação controlada.)

## Implicações para o FALCO

- Erros de categoria-irmã e de "outro …" são **sistemáticos e mapeáveis** — um
  pós-processamento leve (ou few-shot com exemplos das fronteiras difíceis no
  prompt) pode recuperar parte dos ~48% de erros dessas duas classes. Candidato
  a prompt v4 / trabalho futuro.
- O ruído do oráculo não é uniforme: concentra-se em pares de classes vizinhas.
  Para o classificador treinado (E3), ruído estruturado desse tipo tende a ser
  menos danoso que ruído uniforme — argumento a verificar no E4.
