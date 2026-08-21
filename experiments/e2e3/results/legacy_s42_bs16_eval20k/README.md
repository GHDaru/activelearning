# Semente 42, regime antigo (bs=16, eval-limit=20000) — SUPERSEDIDO

Estes são os `e3prime_*_s42.json` originais, movidos para cá em 2026-08-16
quando o autor decidiu **refazer a semente 42 no regime canônico**
(`--batch-size 128 --eval-limit 0`, avaliação nos 177.490 itens da população
inteira) para que as três sementes (42, 7, 123) fiquem comparáveis — o
média ± desvio pedido pela banca só é legítimo entre execuções do mesmo regime.

Diferenças deste regime antigo para o canônico:

| | aqui (antigo) | canônico |
|---|---|---|
| batch size do ajuste fino | 16 | 128 |
| conjunto de avaliação | amostra estratificada de 20.092 | população inteira, 177.490 |

**Não misture**: um arquivo desta pasta não é comparável com um
`e3prime_*_s<semente>.json` da pasta de cima. O efeito é grande — na semente
123 canônica, o Macro F1 de E35 é 0,3461 contra 0,4627 do E35 daqui, em boa
parte porque a população inteira contém classes raras que a amostra de 20k
sub-representa.

**`mcnemar_s42.json` e `bootstrap_f1_s42.json`** desta pasta foram calculados
sobre as predições DESTE regime antigo — os pares vêm dos `*_pred.json` desta
pasta. **Já foram refeitos** sobre as predições canônicas (2026-08-17, cache
do oráculo re-coletado) e os arquivos homônimos na pasta de cima são agora os
canônicos — comparação entre os dois pares está documentada na mensagem de
conclusão ao principal (coordenacao/caixa, tesedaru).

**Achado crítico da comparação**: dois dos três contrastes citados no
Cap. 5 TROCAM DE SINAL entre os regimes.
- **B−C** (valor da seleção): legado +0,0204 (seleção melhor) → canônico
  −0,0120 (seleção PIOR que aleatório em F1, CI exclui zero).
- **E35−D** (a alegação "menos é mais, também no transformer", §5,
  `\label{sec:res-e3p-varredura}`): legado +0,0117 com McNemar não
  significativo (p=0,103, "35k ≈ 50k") → canônico −0,0030 com CI
  [−0,0055;−0,0008] (exclui zero) e McNemar p=2,5e-15 (MUITO
  significativo). **E35 NÃO supera D em nenhuma das 3 sementes, em nenhuma
  métrica** — a alegação (iii) da seção precisa de revisão.

A anomalia que o revisor2 apontou (ponto fora do próprio IC bootstrap em D e
E35, `20260817-0545`) **não se reproduz** no regime canônico — a população de
avaliação 177k vs 20k reduz o viés de classe-ausente-na-reamostragem.

Os braços com rótulos do pipeline real (A) e seus controles (B, C) desta
pasta foram os únicos existentes até 2026-08-17: dependiam do
`annotation_cache_nemotron.jsonl` original, perdido. Um cache RE-COLETADO
(ver `experiments/e5cycle/results/recoleta-20260817/`) já produziu A/B/C
canônicos nas 3 sementes — não comparáveis com os A/B/C desta pasta (regime
diferente E proveniência do oráculo diferente).
