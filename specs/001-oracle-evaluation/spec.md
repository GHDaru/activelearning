# Spec 001 — E0: Avaliação de Oráculos LLM (redesenho fatorial)

## Contexto

No legado (`activetextclassification`), a acurácia medida dos oráculos foi deflacionada
por um defeito de instrumento: o schema JSON não restringia `predicted_category` por
`enum`, e variações de fraseado ("ovos de pascoa" vs. "ovo de pascoa") contaram como
erro. Piloto com o instrumento corrigido (gpt-4o-mini, N=8): 7/8 corretos, 0% de rótulo
inválido, ~94% do input servido do cache de prompt, ~US$0,29/1k rótulos.

## Perguntas de pesquisa (o que o E0 testa)

| RQ | Pergunta | Alimenta na tese |
|----|----------|------------------|
| **RQ1 — Assertividade** | Qual a acurácia global (com IC 95%) e o macro-F1 de cada LLM candidato na tarefa? Diferenças entre modelos são significativas? | Escolha do par (LLM Inicial, LLM Avançado); Cap. Resultados P3 |
| **RQ2 — Custo** | Qual o custo por 1.000 rótulos e a latência de cada modelo (com prompt caching quando disponível)? | Análise custo-benefício vs. rotulagem humana (rec. 12 do orientador) |
| **RQ3 — Perfil de erro** | Em que classes o oráculo erra? O erro concentra-se em pares confundíveis? Qual a taxa de rótulo inválido? | Matriz de confusão/análise qualitativa (rec. 13); base para discussão de robustez a ruído (rec. 14) e para E4 |
| **RQ4 — Efeito do instrumento** | Quanto da inacurácia medida no legado era artefato do schema sem enum? | Achado metodológico (validade de medição de oráculos LLM); justifica a divergência com os números legados |

## Desenho experimental

### Fatores

- **Modelo** (7 níveis): `gpt-4o-mini`, `gpt-4o` (OpenAI); `gemini-2.0-flash` (Google);
  `DeepSeek-V4`, `GLM-5.2` (Huawei MaaS, OpenAI-compatible/Ascend-vLLM);
  `gemma3`, `qwen2.5` (Ollama local).
- **Modo do schema** (2 níveis, apenas RQ4): `enum` (restrito) × `free` (string livre com
  lista no prompt — reproduz o instrumento do legado). O modo `free` roda em 3 modelos
  representativos: `gpt-4o-mini` (API premium), `DeepSeek-V4` (MaaS) e `gemma3` (local).
- **Temperatura**: fixa em 0.0 — justificada pelo piloto legado (insensibilidade
  observada entre 0.0 e 1.0) e pelo requisito de reprodutibilidade em produção.
- **Itens por chamada (b_call)**: rotulagem EM LOTE (itens numerados numa única
  chamada; resposta como array indexado). Reduz drasticamente chamadas e custo
  (~5x no piloto), essencial sob rate limits (MaaS 3 rpm: lote 25 => ~40 min/modelo
  em vez de ~16 h). Como lote muda o instrumento (contaminação de contexto, efeitos
  de posição), há uma CALIBRAÇÃO prévia (`config_calibration.json`): b_call em
  {1, 10, 25} pareado em 100 itens da S-rand (gpt-4o-mini e deepseek-v4-flash);
  o E0 principal usa o maior b_call sem degradação significativa (McNemar).
  O b_call usado fica gravado no oracle_id (sufixo @b{n}).

### Amostras (pareadas — todos os modelos rotulam as MESMAS instâncias)

| Amostra | Construção | Tamanho | Serve a |
|---------|------------|---------|---------|
| **S-rand** | Aleatória simples, seed 42 | 1.000 | RQ1 (acurácia global — distribuição de produção), RQ2, RQ4; McNemar pareado entre modelos |
| **S-strat** | Estratificada, 3 por classe, seed 42 | ~1.866 (622×3) | RQ1 (macro-F1 confiável), RQ3 (confusão por classe) |

Racional: com 622 classes, S-rand deixa a maioria das classes sem suporte — macro-F1 e
matriz de confusão exigem a amostra estratificada; já a acurácia "de produção" e o custo
devem ser medidos na distribuição real (S-rand).

### Métricas e análise

- Acurácia global + **IC de Wilson 95%** (por modelo, por amostra).
- **Macro-F1** (S-strat) + F1 por classe; top-N pares de confusão.
- `invalid_label_rate` (só possível > 0 no modo `free` ou por falha de provedor).
- Custo total, custo/1k rótulos, tokens cacheados, latência média.
- **McNemar** (χ² com correção de continuidade) par-a-par entre modelos na S-rand
  (mesmas instâncias ⇒ teste pareado correto).
- RQ4: Δacurácia (enum − free) por modelo, com McNemar pareado (mesma amostra S-rand).

### Critérios de decisão (gate para E3)

- **LLM Inicial** = argmax(acurácia/custo por 1k) sujeito a acurácia ≥ 85% (S-rand).
- **LLM Avançado** = argmax(acurácia), desde que significativamente superior ao
  Inicial (McNemar, α=0,05); caso contrário, o FALCO degenera para oráculo único e a
  Fase 3 é rediscutida.
- Se nenhum modelo ≥ 85%: E4 (robustez a ruído) torna-se obrigatório e a narrativa
  da tese muda para "aprender com oráculo ruidoso".

## Requisitos de implementação

1. Modo `enum`: `response_format json_schema` (OpenAI/MaaS-vLLM), `response_schema`
   (Gemini), `format` (Ollama). Modo `free`: mesma estrutura JSON sem `enum` no rótulo,
   lista de categorias injetada no system prompt (estático ⇒ ainda cacheável).
2. Resposta fora do schema ⇒ `invalid_label` contabilizado (nunca aceita/descartada).
3. Huawei MaaS: adapter OpenAI-compatible com `base_url` configurável; **desativar
   thinking/reasoning** no DeepSeek (bug conhecido de structured output no vLLM com
   thinking ativo); preços por config (não hardcoded).
4. Execução retomável (JSONL incremental); amostras derivadas por seed fixa.
5. `oracle_id` distingue o modo do schema (sufixo `#free`) — artefatos não colidem.
6. Saídas por modelo×modo×amostra: `annotations_*.jsonl`, `report_*.json`; consolidado:
   `e0_summary.json` + `e0_mcnemar.json` (análise pareada).

## Critérios de aceitação

- [ ] `uv run pytest` verde (inclui Wilson, McNemar, schema free).
- [ ] Piloto N=8 com 1 provider real em ambos os modos (enum/free) produz artefatos.
- [ ] `analyze_e0.py` gera tabela consolidada + matriz McNemar a partir dos JSONLs.
- [ ] Documentada a comparação com os números legados (RQ4).

## Estimativa de custo (ordem de grandeza)

~2.900 chamadas/modelo (S-rand + S-strat) × 7 modelos ≈ 20k chamadas no modo enum
(+3.000 no modo free). Com cache: gpt-4o-mini ≈ US$1; gpt-4o ≈ US$13; Gemini/MaaS
tipicamente abaixo do gpt-4o; locais custo zero. Total esperado < US$25.
