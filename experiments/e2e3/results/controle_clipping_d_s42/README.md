# Controle do efeito do gradient clipping (D, semente 42)

`e3prime_D_s42_bs16_com_clipping.json` é o braço D (lote 16, avaliação
canônica) **retreinado com** `clip_grad_norm_(max_norm=1.0)`
(`bertimbau.py`, commit `1dabdbb`), para medir o efeito da correção num
braço que **já treinava bem sem ela** — não faz parte da varredura oficial,
é só o experimento de controle.

Comparação com `experiments/e2e3/results/e3prime_D_s42_bs16.json` (o D
oficial da varredura, sem clipping, já publicado antes da correção):

| | sem clipping | com clipping | delta |
|---|---|---|---|
| Macro F1 | 0,4523 | 0,4625 | +0,0102 (+2,26% relativo) |
| Acurácia | 0,8821 | 0,8873 | +0,0052 |

Efeito real, positivo, mas pequeno — muito menor que os efeitos de tamanho
de lote (20-100%+) que a varredura investiga. Decisão sobre se os outros 24
braços já publicados sem clipping precisam ser regerados por consistência:
ver coordenação (tesedaru), mensagem de 2026-08-18.
