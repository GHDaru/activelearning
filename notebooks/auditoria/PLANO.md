# Plano de atualização dos experimentos — auditoria + reexecução

> Proposta do `executor01` ao `principal` (planejamento é dele, PROTOCOLO §2-bis).
> Origem: pedido do autor em 2026-08-16 — "refazer todos os experimentos no
> Kaggle com GPU" + "tem coisa da edição do ano passado" — combinado com a
> tarefa `20260816-2026_principal_executor01_tarefa_notebooks-auditaveis`.

> **Nomes dos experimentos**: os códigos `E0`…`E6` viraram nomes legíveis;
> o mapa de-para e a convenção de nomes de artefato estão em `NOMES.md`.

## A ideia que organiza tudo

**Auditar e refazer são o mesmo trabalho.** Cada notebook começa reproduzindo o
número que está na tese hoje. Se reproduz, o experimento está auditado e o
notebook vira o artefato auditável que o autor pediu. Se **não** reproduz, a
divergência é o item de atualização — e aí sabemos exatamente o que refazer e
por quê, em vez de refazer tudo no escuro.

Isso evita o pior cenário: gastar dezenas de horas de GPU reexecutando e, no
fim, não saber se um número mudou porque o experimento mudou ou porque estava
errado antes.

## Correção de premissa: GPU só serve para UM experimento

| Experimento | Perfil | Custo | Kaggle ajuda? |
|---|---|---|---|
| E2/E3′ BERTimbau | **GPU** | min–h | **Sim, muito** — é o único caso de GPU |
| E0 oráculos | Rede + API LLM | ~US$ 4, horas (limite de taxa) | Só como ambiente; precisa de chave |
| E0-P prompts | Rede + API LLM | ~US$ 0,10, ~2 h | idem |
| E5 ciclo real | Rede + API (NIM, grátis) | ~1 h | idem |
| E1/E1b/E4 | **CPU** | ~9 h | Sim, como CPU grátis (não GPU) |
| E6 população | **CPU** | 2–6 h por braço | Sim, como CPU grátis |
| P1/P2 (Cap. 4) | **CPU** | ~2 h | Sim, como CPU grátis |

O ganho do Kaggle é **compute grátis + notebook auditável**, não GPU. Só o
BERTimbau usa GPU de verdade. Vale dizer isso alto porque muda o sequenciamento:
não faz sentido esperar cota de GPU para rodar E1/E4/E6/P1.

## O que já verifiquei (fatos, não suposição)

1. **Dois protocolos de dados convivem — e é deliberado**, documentado no
   Cap. 3: o *CategorySchema* de **621 categorias** (620 + `_rare_`, sobre
   250.221 linhas cruas) governa os experimentos de oráculo; os experimentos
   populacionais (E5, E6, E3′) usam a visão deduplicada de **231.490 textos e
   714 classes**. Meu pré-voo reproduz o segundo **exatamente**:
   `dedup=231490 pool=50000 população=177490 classes=714`. Não é dívida: é
   desenho declarado. Mas precisa aparecer no cabeçalho de cada notebook, senão
   qualquer leitor lê como inconsistência.

2. **A dívida real da edição antiga é o Capítulo 4.** Os números de P1/P2 vêm
   do "programa experimental da pesquisa" (repositório `Tese-Vers-o-Draft`,
   somente leitura). O replay independente existe como código
   (`experiments/p1/replay_l0_sensitivity.py`, `replay_ga.py`) mas
   **`experiments/p1/results/` não existe** — nenhum artefato foi commitado.
   Hoje o Cap. 4 **não é reproduzível a partir deste repositório**.

3. **A tabela `tab:e3p-sweep` do Cap. 5 não sobrevive ao regime canônico**, e
   agora com duas sementes no MESMO regime — o `executor02` reexecutou a
   semente 42 em canônico (kernel `falco-e3prime-s42`), o que torna a
   comparação legítima pela primeira vez:

   | Braço | s42 publicado (lote 16) | s42 canônico | s7 canônico |
   |---|---|---|---|
   | E35 | 0,463 | 0,3660 | 0,3440 |
   | D (régua) | 0,451 | 0,3691 | 0,3771 |
   | E35 supera D? | **sim** | não | não |

   A leitura (iii) do capítulo — "menos é mais também no transformer", o E35
   superando a supervisão completa — **só vale no regime de lote 16**. Nas duas
   sementes canônicas, não vale. Além disso as sementes canônicas discordam
   entre si sobre o piso de orçamento (em s42 o E35 cruza o critério; em s7
   nenhum braço cruza), o que é sensibilidade real à semente — exatamente o que
   a banca mandou medir. Item de maior risco desta lista.

4. **Sem chaves de API nesta sessão** (`NVIDIA_API_KEY`, `OPENROUTER_API_KEY`,
   `OPENAI_API_KEY`, `GEMINI_API_KEY` — todas ausentes). Mas isso importa menos
   do que parecia: veja o item 6.

6. **As anotações cruas do E0 e do E0-P estão versionadas** — 33 e 9 arquivos,
   incluindo os `annotations_*.jsonl` de cada provedor. Elas escaparam do
   `.gitignore` porque a regra `experiments/*/results/*.jsonl` casa **um só
   nível**, e as do E0 moram em `results/rand/` e `results/strat/`. O cache do
   E5, que fica direto em `results/`, foi apanhado pela mesma regra — é por isso
   que um sobreviveu e o outro não. Consequência prática: **E0 e E0-P podem ser
   reauditados de graça, sem chave e sem gastar nada**; só uma recoleta nova
   precisa de crédito.

5. **`annotation_cache_nemotron.jsonl` continua fora do repositório**, travando
   os braços A/B/C do E3′ e qualquer replay do E5 a partir do cache.

## As ondas

### Onda 0 — destravar (só o autor pode; nada aqui é trabalho meu)
- Commitar `annotation_cache_nemotron.jsonl` (`git add -f`) ou subir como
  Kaggle Dataset privado.
- **Decidir o regime do E3′** (as três opções estão na mensagem
  `20260816-2130`). Enquanto não decidir, toda semente nova é aposta.
- Dizer quais chaves de API existem e qual orçamento a Onda **3b** pode gastar.
  (Perdeu urgência: a Onda 3a não depende disso.)

### Onda 1 — reproduzir o que já tem artefato (começa já; sem depender da Onda 0)
**CONCLUÍDA** — `escala-populacional.ipynb` (E6) e `classificador-forte.ipynb`
(E3′). Placar: **27 de 29 afirmações do Cap. 5 conferem**, com McNemar e
bootstrap recomputados. Duas divergências: a população do E6 (≈140 mil contra
181.490) e um arredondamento no E3′ (3,0 p.p. contra 2,93).

### Onda 2 — a dívida do Capítulo 4 (a "edição do ano passado")
Notebook `conjunto-inicial.ipynb` (P1/P2): roda os dois replays que nunca foram
commitados e coloca lado a lado a série antiga (do draft) e a nova. É CPU,
~2 h. **É aqui que o pedido do autor sobre a edição antiga se resolve.**

### Onda 3a — reanálise do oráculo (GRÁTIS, não espera ninguém)
`escolha-do-oraculo.ipynb` (E0) e `efeito-do-prompt.ipynb` (E0-P) recalculam as métricas a partir das
anotações já versionadas (item 6 acima). Sem chave, sem custo, sem repetir uma
única chamada ao LLM. É auditoria de verdade: confere se os números publicados
saem das respostas que estão gravadas.

### Onda 3b — recoleta do oráculo (bloqueada; custa dinheiro)
Só se a 3a apontar divergência que exija reexecução, ou se o autor quiser
provedor/modelo novo. `ciclo-completo.ipynb` (E5) cai aqui porque depende do cache que
falta. Estimativa da tabela de reprodução: ~US$ 4 no E0, ~US$ 0,10 no E0-P,
US$ 0 no E5 (NIM).

### Onda 4 — CPU longo
`estrategias-de-selecao.ipynb` (E1) e `robustez-ao-ruido.ipynb` (E4), ~9 h. Cabe numa sessão de CPU do Kaggle com
retomada por estado, como o E3′ faz.

### Onda 5 — fechamento
`00-visao-geral.ipynb` (índice: o que cada experimento responde e o estado da
última execução) + consolidação multi-semente do E3′ quando o regime estiver
decidido.

## Padrão de cada notebook (o que "auditável" quer dizer)

1. **Cabeçalho**: pergunta do experimento, hipótese, artefato esperado, e **qual
   protocolo de dados** (621 categorias ou 714 classes) — o ponto 1 acima.
2. **Identidade dos dados impressa na tela**: md5 do CSV, linhas, classes,
   tamanho das partições. É o que prova que o notebook viu a mesma base.
3. **Reprodução primeiro**: carrega o artefato publicado e compara com o
   recém-calculado, com veredito explícito na tela.
4. **Execução parametrizada**: semente, orçamento, braços.
5. **Saída visual**: tabela + gráfico (curva, comparação de braços, matriz de
   confusão quando couber).
6. **Rodapé**: caminho do JSON gravado e a linha de comando equivalente.

## Regra de divergência (inegociável)

Número que não reproduz vira **bloqueio ao `principal`**, com os dois valores e
o artefato de cada um. **Não ajusto número da tese** — nem o do repositório para
casar com a tese, nem o contrário.

## O que este plano NÃO inclui

Editar texto da tese, decidir qual número entra no Cap. 4 ou 5, e escolher o
regime do E3′. Tudo isso é do `principal` e do autor. Eu produzo a evidência.
