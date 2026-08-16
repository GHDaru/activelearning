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

**`mcnemar_s42.json` e `bootstrap_f1_s42.json`** (ainda na pasta de cima, para
não quebrar os links do plano) foram calculados sobre as predições DESTE
regime — os pares vêm dos `*_pred.json` desta pasta. Depois que a s42 canônica
existir (e A/B/C canônicos, quando o cache do oráculo chegar), essas
estatísticas precisam ser refeitas sobre as predições canônicas.

Os únicos braços com rótulos do pipeline real (A) e seus controles (B, C) só
existem neste regime por enquanto: dependem do
`annotation_cache_nemotron.jsonl`, que não está no repositório.
