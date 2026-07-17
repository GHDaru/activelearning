# Critério de parada e drift: quando liberar o modelo e quando retreinar

Guia operacional para o ciclo de vida do classificador treinado pelo FALCO —
da decisão de PARAR de rotular à decisão de RETREINAR em produção. As seções
1–2 já estão implementadas na biblioteca; a seção 3 é o desenho recomendado
para o FlowBuilder (com apontadores de implementação).

## 1. Parada do laço de AL (implementado)

O laço não roda até esgotar o orçamento cegamente. Três gatilhos, na ordem:

1. **Estagnação na validação** (`run_falco`): se o Macro-F1 no conjunto de
   validação não melhora mais que `eps=1e-3` por `patience=5` iterações, a
   fase atual termina. Sem oráculo Avançado configurado, o ciclo PARA — o
   smoke E5 real parou em 910/1000 rótulos exatamente por isso. O teste
   NUNCA participa dessa decisão (correção A1 da revisão R1).
2. **Orçamento** (`budget`): teto absoluto de rótulos pagos.
3. **Pool esgotado**: sem instâncias restantes, fim.

Racional estatístico: a curva de aprendizado tem retornos decrescentes; o
custo marginal do rótulo é constante, mas o ganho marginal cai — quando o
ganho observável na validação fica abaixo da resolução do próprio conjunto de
validação (≈1/√n_val), continuar rotulando é pagar por ruído.

## 2. Quando LIBERAR o modelo (critério de release)

Recomendação em três condições simultâneas, todas mensuráveis com o que a
biblioteca já produz:

- **(a) Estagnação atingida** (gatilho 1 acima) — o modelo parou de melhorar
  com o oráculo disponível;
- **(b) Desempenho de validação ≥ alvo de negócio** — o alvo é externo (ex.:
  "Macro-F1 ≥ 0,95 × teto supervisionado conhecido", a hipótese da tese). Sem
  alvo externo, use o teto empírico: o braço oráculo-total (treinar com TODO o
  pool rotulado pelo mesmo oráculo) custa só inferência LLM e diz o máximo que
  ESTE oráculo permite — se a curva estagnou perto dele, mais rótulos não
  resolvem, só um oráculo melhor;
- **(c) IC compatível com o alvo** — o intervalo de Wilson do desempenho de
  validação deve estar acima do piso aceitável (com n_val=1000, a meia-largura
  é ±2–3 p.p.; valide com folga ou aumente n_val).

Só então avalia-se UMA vez no teste intocado — esse número é o que se reporta.

## 3. Drift em produção: quando RETREINAR

Textos curtos de varejo derivam rápido (novos produtos, novas marcas, novas
grafias). Três camadas de monitoramento, da mais barata à mais cara:

| Camada | Sinal | Custo | Gatilho sugerido |
|---|---|---|---|
| 1. Drift de entrada (sem rótulo) | % de tokens fora do vocabulário do treino; distância da distribuição de embeddings (PSI/KS sobre projeções) | zero | OOV-rate 2× o do treino, ou PSI > 0,2 |
| 2. Drift de confiança (sem rótulo) | queda da confiança média/entropia média das predições em produção | zero | média móvel 7d abaixo de μ−2σ do período de referência |
| 3. Verificação amostral (com oráculo) | acurácia numa amostra pequena rotulada pelo LLM Inicial (ex.: 200 itens/semana ≈ US$ 0,01) | ~zero | acurácia amostral abaixo do IC inferior da validação de release |

Política recomendada:

- **Camadas 1–2 disparam** → antecipar a camada 3 (rotular amostra agora).
- **Camada 3 confirma queda** → retreinar. O retreino no FALCO é barato por
  desenho: os rótulos antigos não expiram; roda-se um ciclo INCREMENTAL de AL
  só sobre o pool novo (o classificador atual seleciona o que não entende do
  fluxo recente; o oráculo LLM rotula; retreina-se do zero com L_antigo ∪
  L_novo — retreinar do zero preserva comparabilidade e evita os problemas de
  warm-start).
- **Cadência mínima**: mesmo sem gatilho, uma verificação amostral mensal —
  drift de catálogo pode ser gradual demais para as camadas 1–2.

Implementação no FlowBuilder: as camadas 1–2 são estatísticas do próprio
`predict_proba` + vocabulário do vetorizador (nenhuma dependência nova); a
camada 3 reusa `EvaluateOracle` com `n` pequeno. Um endpoint
`GET /api/models/{id}/health` com esses três sinais é o próximo incremento
natural.

## Conexão com a tese

- Parada por estagnação em V: Cap. 3 (§ transição de fases) — constantes
  p=5, ε=1e-3 justificadas lá.
- Teto por oráculo (braço oráculo-total): Cap. 3, desenho do E3 (5º braço).
- Retreino×warm-start: discussão no diário 17/07 e literatura de
  warm-starting (Ash & Adams, 2020).
- Drift: tema de trabalhos futuros no Cap. 6 (operacionalização).
