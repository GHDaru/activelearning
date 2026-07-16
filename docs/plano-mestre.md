# Plano Mestre — Tese FALCO: desenvolvimento e escrita

**Repositórios:** código/experimentos em `GHDaru/activelearning` · texto em `GHDaru/tesedaru`
**Governança:** constituição spec-kit (`.specify/memory/constitution.md`) · parecer da banca simulada em `tesedaru/docs/parecer-fase-menos-1.md`
**Regra de ouro:** nenhum número entra na tese sem artefato rastreável (config + git SHA + seeds + JSONL) neste repositório.

---

## 1. Estado atual (o que já está pronto)

| Item | Estado |
|---|---|
| Parecer crítico (banca simulada) + plano de reescrita capítulo a capítulo | ✅ aprovado |
| Biblioteca `activelearning`: DDD+Hexagonal, domínio puro, 31 testes | ✅ |
| Adapters de oráculo: OpenAI, Gemini, Huawei MaaS, OpenRouter, Simulado(ε) | ✅ validados ao vivo |
| Instrumentação: enum/json-prompt/free, lote por chamada (@b), prompt caching, throttle/429, custo por rótulo | ✅ |
| E0 desenhado (spec 001): 4 RQs, amostras pareadas S-rand/S-strat, Wilson+McNemar, calibração de lote | ✅ |
| Auditoria da base de teste | ✅ (ver §2) |
| Prompt do oráculo revisado (v3) | ✅ (ver §3) |
| `tesedaru` com template ppginf + dados PPGMNE + estrutura de capítulos | ✅ |
| Piloto E0 (amostra reduzida, 7 configurações de oráculo) | ✅ resultados em `experiments/e0/results_pilot/` |

## 2. Auditoria da base (`data/dataset.csv`, 250.365 linhas)

Achados que entram na tese (Cap. Metodologia, seção do conjunto de dados):

1. **Conflitos de rótulo-ouro:** 719 textos únicos (1.807 linhas, 0,7%) aparecem com
   2+ rótulos distintos (ex.: "milharina quaker 500g" → {farinha de milho, floco de
   milho, fuba, polenta}). Consequência: teto de acurácia mensurável ≈ 99,3% na
   S-rand; em comparações pareadas afeta todos os modelos igualmente. **Decisão:**
   manter (ruído real de produção), reportar como limitação de medição.
2. **Rótulo operacional `inativo`:** 144 linhas com status de cadastro no lugar de
   categoria ("UVA PASSA ESCURA 150G" → inativo). **Decisão:** excluído das amostras
   (`exclude_labels` no config; documentado).
3. **Duplicatas exatas (texto+rótulo):** 19.356 linhas (7,7%). Inócuo para o E0;
   **crítico para o E3**: o particionamento treino/teste DEVE deduplicar por texto
   antes do split (risco de vazamento treino→teste). Registrado como requisito do E3.
4. **Espaço de rótulos:** 795 classes brutas; 621 com ≥5 amostras + `_rare_` = 622 no
   schema; apenas 0,13% das linhas caem em `_rare_`. Sem pares singular/plural
   coexistentes (pós-normalização). 32 classes "outro …" (≈10 mil linhas) — legítimas,
   difíceis por definição.
5. **Comprimentos:** 4–50 chars, mediana 32 — consistente com a caracterização de
   texto curto da tese.

## 3. Prompt do oráculo (v3) — registro de decisão

Mudanças da v2→v3 (gravadas em `prompt_version` de cada anotação):
- Contexto de domínio explícito (cupom fiscal/e-commerce, caixa alta, tabela de
  abreviações frequentes).
- Instrução de classificar o TIPO do produto ignorando marca/tamanho/embalagem.
- **Política de `_rare_`:** preferir categoria plausível; `_rare_` só se nenhuma
  categoria descrever o tipo (a v2 induzia fuga para `_rare_` — visto no piloto:
  medicamento → `_rare_` com ouro "outro farma").
- 3 exemplos canônicos estáticos (mantêm o prefixo cacheável).
- Modo `free` (RQ4) compartilha o mesmo contexto, isolando o fator enum.

## 4. Trilha de desenvolvimento (activelearning)

| ID | Entrega | Depende de | Executor | Critério de pronto |
|----|---------|-----------|----------|--------------------|
| D1 | Calibração de lote + **E0 completo** + commit de `results/` | chaves/cotas | **Gilsiley** | `e0_summary/table/mcnemar.json` versionados |
| D2 | Análise estatística E0 + decisão do par (Inicial, Avançado) pelo gate da spec 001 | D1 | Claude | Nota de decisão em `specs/001/decision.md` |
| D3 | Portar **DRI-SL** para `domain/coldstart.py` com testes; regenerar figuras P2 dos logs legados | — | Claude | Figuras reproduzidas = logs legados |
| D4 | `RunActiveLearning` + `BinaryVectorizerClassifier` (PVBin) + **E1** (estratégias × lote, oráculo simulado) | D3 | Claude (CPU) | Curvas + LCE de RND/ENT/LC/SM/HYB |
| D5 | `BertimbauClassifier` (HF/transformers) + **E2** (épocas × \|L\|) | — | Claude (código) + Gilsiley (GPU) | Curvas de loss justificando épocas |
| D6 | `RunFalco` (PhasePolicy, oráculo progressivo) + **E3: FALCO vs RS vs US** (pool ~50k dedup, B=30%, 3–5 seeds, Wilcoxon) | D2, D4, D5 | ambos | Curvas, LCE, testes; preenche `[Suposição:]` |
| D7 | E4 robustez a ruído (ε ∈ {0, 0.1, 0.2, 0.4}) — **condicional** ao gate do E0 | D2 | Claude (código) + GPU | Curvas por ε |
| D8 | FlowBuilder novo: backend FastAPI (driving adapter da lib) + frontend React | D6 | ambos | Fluxo de AA executável via UI |

## 5. Trilha de escrita (tesedaru)

| ID | Capítulo/Seção | Fonte | Depende de | Pode começar |
|----|----------------|-------|-----------|--------------|
| W1 | **3-metodo** (reescrita ~60%): dados+auditoria §2, particionamento com dedupe, DRI-SL único (IPR cortado), AG com parâmetros reais, LCE única (Simpson), desenho E0–E4, justificativas | draft + specs | — | **JÁ** |
| W2 | **4-resultados-l0** P1: sensibilidade L0 (migrar ~80%) + AG (condensar 575→~200 linhas, correlações → apêndice) | dados legados | — | **JÁ** |
| W3 | **4-resultados-l0** P2: DRI-SL vs AG/aleatório com figuras reais | D3 | D3 | após D3 |
| W4 | **5-resultados-falco** P3: oráculo (RQ1–RQ4, tabela+IC+McNemar, custo, matriz de confusão, efeito do instrumento) | E0 | D1/D2 | rascunho com piloto **JÁ**; números finais após D1 |
| W5 | **5-resultados-falco** P4: FALCO vs RS/US (curvas, LCE, Wilcoxon) | E3 | D6 | após D6 |
| W6 | **6-conclusao**: discussão+conclusão sem nenhum `[Suposição:]`; comparação com trabalhos relacionados (migrar ~90%) | draft + resultados | W4, W5 | após W4/W5 |
| W7 | **1-intro** (ajustes: hipótese falseável, 4 pilares, organização) + **2-fundam** (condensar ML ~75%, migrar AL/STC/revisão) | draft | — | **JÁ** |
| W8 | Apêndices a1–a6 (LCE, AG, DRI-SL, biblioteca nova, prompts v3 + schemas, tabelas) | vários | W1–W5 | a1/a2/a6 já; a4/a5 após D2 |
| W9 | Pré-textuais: resumo/abstract com números reais, siglas, agradecimentos | tudo | W6 | último |

## 6. Sequência imediata (próximas ~2 semanas)

1. **Gilsiley:** rotacionar as 3 chaves de API → rodar `config_calibration.json` →
   rodar `config.json` (E0 completo) → commit dos `results/`.
2. **Claude (paralelo):** D3 (DRI-SL) + W1 (Metodologia) + W2 (P1) + rascunho W4 com
   dados do piloto.
3. **Juntos:** D2 (gate de decisão do oráculo) assim que o E0 completo chegar.
4. Na sequência: D4/E1 (barato, destrava W-análises), depois D5/D6 (E3 = coração da
   defesa).

## 7. Riscos abertos

- **Gate do oráculo** (acc < 85% em todos) → E4 obrigatório + mudança de narrativa
  (previsto na spec 001; texto da tese será escrito para os dois desfechos).
- **Cota MaaS 3 rpm** → mitigada com lote 25 (~40 min/modelo); contratar cota
  encurta mais.
- **GPU para BERTimbau (E2/E3)** — confirmar disponibilidade da RTX 3090; senão,
  reduzir pool/seeds (o desenho já prevê ~50k + Wilcoxon).
- **Prazo:** trilha de escrita W1/W2/W7 não depende de experimento nenhum — mantê-la
  sempre à frente para o prazo não ficar refém das execuções.
