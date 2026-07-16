# Spec 001 — E0: Avaliação de Oráculos LLM com Saída Restrita (enum)

## Contexto e problema

No legado (`activetextclassification`), a acurácia medida dos oráculos LLM foi:
gemini-1.5-flash 57,9% (N=2020), gpt-4o-mini 47,9% (N=1020), gemma3 28,1% (N=1020),
qwen2.5 21,2% (N=1020). O schema JSON usado (`prompts.py`) definia
`predicted_category` como string **sem `enum`**, permitindo respostas fora da lista de
categorias. Análise qualitativa mostrou erros que são variações de fraseado
("ovos de pascoa" vs. ouro "ovo de pascoa"; "bebida de leite" vs. "bebida lactea"),
não erros de classificação. **A medição está contaminada pelo instrumento.**

## Objetivo

Medir a acurácia real dos LLMs candidatos a oráculo do FALCO com a saída
estruturalmente restrita ao `CategorySchema`, e produzir a análise de custo e de erros
que o Capítulo de Resultados da tese (pilar P3) exige.

## Requisitos

1. O rótulo predito DEVE ser restringido por `enum` gerado de `CategorySchema`
   (OpenAI `response_format=json_schema`; Gemini `response_schema`; Ollama `format`).
2. Resposta fora do schema (só possível por falha do provedor) DEVE ser contabilizada
   como `invalid_label` — nunca aceita nem descartada.
3. Cada chamada DEVE registrar tokens, latência e custo estimado (USD).
4. N ≥ 1000 instâncias por modelo, amostragem com semente fixa (42), temperatura 0.0.
5. Execução retomável: anotações persistidas incrementalmente em JSONL; instâncias já
   anotadas são puladas em re-execuções.
6. Saídas: `annotations_<oracle>.jsonl`, `report_<oracle>.json`,
   `e0_summary.json` (tabela consolidada para a tese).

## Modelos avaliados

| Provider | Modelo | Papel candidato no FALCO |
|---|---|---|
| openai | gpt-4o-mini | LLM Inicial |
| openai | gpt-4o | LLM Avançado |
| gemini | gemini-2.0-flash | LLM Inicial (alternativa) |
| ollama | gemma3 | LLM Inicial local (custo zero) |
| ollama | qwen2.5 | LLM Inicial local (alternativa) |

## Critérios de aceitação

- [ ] `uv run pytest` verde (domínio: CategorySchema/enum, Annotation, report).
- [ ] `run_e0.py --config config.json` executa ponta a ponta com ao menos um provider
      disponível e produz os três artefatos.
- [ ] Relatório inclui: accuracy, macro-F1, invalid_label_rate, custo por 1k rótulos,
      matriz de confusão (para análise de erros da tese).
- [ ] Comparação com os números do legado documentada no relatório do experimento
      (mesma amostra-fonte, schema corrigido) — quantifica o efeito do `enum`.

## Decisões pós-execução (gate para E3)

- Se melhor modelo ≥ ~85% acc: narrativa original do FALCO mantida (oráculo substitui
  rotulagem humana com ruído conhecido).
- Se < ~85%: E4 (robustez a ruído) torna-se obrigatório e a discussão da tese muda o
  foco para "aprender com oráculo ruidoso" + validação humana amostral.
- Escolha do par (LLM Inicial, LLM Avançado) por custo-benefício: maximizar
  `accuracy / cost_per_1k_labels`, com o Avançado sendo o de maior acurácia absoluta.
