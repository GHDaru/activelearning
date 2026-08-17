"""Fonte legível do índice de auditoria (Onda 5 — fechamento).

Edite ESTE arquivo, não o .ipynb — depois rode:

    python notebooks/auditoria/build_visao_geral.py
"""
from __future__ import annotations

import json
from pathlib import Path

SAIDA = Path(__file__).resolve().parent / "00-visao-geral.ipynb"


def md(texto: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": texto.strip("\n").split("\n")}


def code(texto: str) -> dict:
    linhas = texto.strip("\n").split("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [],
            "source": [l + "\n" for l in linhas[:-1]] + [linhas[-1]]}


celulas = []

celulas.append(md(r"""
# Índice de auditoria — experimentos da tese FALCO

> Mapa dos 8 experimentos, o que cada um responde, e o estado da auditoria de
> cada um. Gerado pelo `executor01`, tarefa de rastreabilidade
> (`20260816-2205`), ordem do autor de seguir todas as ondas
> (`20260817-1210`). Nomes conforme `NOMES.md`; números conforme
> `docs/records/rastreabilidade.json` no `tesedaru`.

## Como ler este índice

- **Notebook de auditoria**: reconstrói número publicado a partir do artefato
  (ou, quando o artefato não existia, o gera pela primeira vez). Roda local e
  no Kaggle (CPU, gratuito — nenhum destes experimentos precisa de GPU exceto
  o classificador forte).
- **Divergência**: artefato existe e diz outra coisa que a tese. Reportada ao
  `principal`; nunca corrigida por este notebook.
- **Sem evidência**: nem a tese nem o repositório têm o dado bruto.
"""))

celulas.append(code(r'''
# Carrega o estado ATUAL da rastreabilidade — esta célula não fica desatualizada
# porque lê o arquivo, não repete números fixos no texto.
import json, os, subprocess, sys
from pathlib import Path
from collections import Counter, defaultdict

def achar_tesedaru() -> Path:
    aqui = Path.cwd()
    for base in [aqui, *aqui.parents]:
        candidato = base.parent / "tesedaru" / "docs/records/rastreabilidade.json"
        if candidato.exists():
            return candidato.parent
        candidato2 = base / "tesedaru" / "docs/records/rastreabilidade.json"
        if candidato2.exists():
            return candidato2.parent
    destino = Path("/tmp/tesedaru/docs/records")
    if not destino.exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/GHDaru/tesedaru.git", "/tmp/tesedaru"],
                       check=True, capture_output=True)
    return destino

PASTA = achar_tesedaru()
dados = json.loads((PASTA / "rastreabilidade.json").read_text(encoding="utf-8"))
print(f"rastreabilidade.json: {len(dados['itens'])} itens · {dados['resumo']}")
print(f"cobertura declarada: {dados['cobertura']}")
'''))

celulas.append(code(r'''
# Tabela: um experimento por linha, com o estado da própria auditoria.
EXPERIMENTOS = [
    ("escolha-do-oraculo", "E0", "qual LLM anota melhor, a que custo, com que perfil de erro",
     "escolha-do-oraculo.ipynb", "falco-auditoria-escolha-do-oraculo"),
    ("efeito-do-prompt", "E0-P", "quanto do resultado é do modelo vs. da forma de perguntar",
     "efeito-do-prompt.ipynb", "falco-auditoria-efeito-do-prompt"),
    ("estrategias-de-selecao", "E1", "qual estratégia de seleção vence com oráculo perfeito",
     "estrategias-e-robustez.ipynb", "falco-auditoria-estrategias-e-robustez"),
    ("robustez-ao-ruido", "E4", "o laço aguenta oráculo que erra? que erro é benigno?",
     "estrategias-e-robustez.ipynb", "falco-auditoria-estrategias-e-robustez"),
    ("escala-populacional", "E6", "seletores na base inteira; o quanto a autoavaliação engana",
     "escala-populacional.ipynb", "falco-auditoria-escala-populacional"),
    ("classificador-forte", "E3′", "o BERTimbau valida o que o pipeline barato entrega",
     "classificador-forte.ipynb", "falco-auditoria-classificador-forte"),
    ("conjunto-inicial", "P1/P2", "composição e tamanho do L0 importam? qual o teto?",
     "conjunto-inicial.ipynb", "falco-auditoria-conjunto-inicial"),
]

por_exp = defaultdict(Counter)
for item in dados["itens"]:
    por_exp[item["experimento"]][item["status"]] += 1

print(f"{'experimento':<24}{'código':<7}{'rastr.':>7}{'diverg.':>8}{'s/evid.':>8}{'legado':>7}  notebook")
for nome, codigo, _, notebook, _ in EXPERIMENTOS:
    c = por_exp.get(nome, Counter())
    print(f"{nome:<24}{codigo:<7}{c['rastreado']:>7}{c['divergente']:>8}"
          f"{c['sem-evidencia']:>8}{c['legado']:>7}  {notebook}")

total_geral = por_exp.get("todos", Counter())
if total_geral:
    print(f"\n(sem experimento específico — infraestrutura): {dict(total_geral)}")
'''))

celulas.append(md(r"""
## Os 8 experimentos, um a um

| Nome | Código | Pergunta | Notebook | Kaggle |
|---|---|---|---|---|
| **escolha-do-oraculo** | E0 | qual LLM anota melhor, a que custo, com que perfil de erro | `escolha-do-oraculo.ipynb` | `falco-auditoria-escolha-do-oraculo` |
| **efeito-do-prompt** | E0-P | quanto é do modelo vs. da forma de perguntar | `efeito-do-prompt.ipynb` | `falco-auditoria-efeito-do-prompt` |
| **estrategias-de-selecao** | E1 | qual estratégia vence com oráculo perfeito | `estrategias-e-robustez.ipynb` | `falco-auditoria-estrategias-e-robustez` |
| **robustez-ao-ruido** | E4 | o laço aguenta oráculo que erra? | `estrategias-e-robustez.ipynb` | (mesmo kernel) |
| **ciclo-completo** | E5 | o pipeline ponta a ponta com oráculo real funciona? | *(não auditado ainda)* | — |
| **escala-populacional** | E6 | seletores na base inteira; viés da autoavaliação | `escala-populacional.ipynb` | `falco-auditoria-escala-populacional` |
| **classificador-forte** | E3′ | o BERTimbau valida o pipeline barato? | `classificador-forte.ipynb` | `falco-auditoria-classificador-forte` |
| **conjunto-inicial** | P1/P2 | composição e tamanho do L0 importam? | `conjunto-inicial.ipynb` | `falco-auditoria-conjunto-inicial` |

**E5 (ciclo-completo) fica de fora desta rodada**: depende do
`annotation_cache_nemotron.jsonl`, que segue fora do repositório — é o
mesmo bloqueio que trava os braços A/B/C do `classificador-forte`. Não é
esquecimento; é dependência não resolvida, registrada desde a mensagem das
19:35 do dia 16/08.

## Os achados que pedem decisão (não deste notebook)

Lista completa e atualizada está em `rastreabilidade.json` — aqui, só os que
mudam leitura do texto, não número solto:

1. **E0/RQ1 — significância sem lastro.** A tese afirma
   `b=43, c=16, p<0,001`; o pipeline real, rodado do zero sobre as respostas
   cruas do LLM, dá `b=73, c=91, p=0,184` — não significativo. Confirmado com
   confiança máxima (zero divergência em todo o resto do pipeline do E0).
2. **classificador-forte — a leitura (iii) da varredura é efeito de regime.**
   "E35 supera a régua" só vale no regime de lote 16 (o publicado). Nas três
   sementes canônicas (42, 7, 123), não vale em nenhuma.
3. **escala-populacional — população reservada, e o Cap. 3 já tinha o número certo.**
   O Cap. 5 diz ≈140 mil; artefato e recálculo dão 181.490. Achado novo do
   R5/Cap. 3 (20260817-1940): o **Cap. 3, na Seção de particionamento, já diz
   "≈177 mil"** — que bate com o artefato (177.490, calculado por
   50.000+4.000+177.490=231.490). Ou seja: é o Cap. 5 que diverge, tanto do
   artefato quanto do próprio Cap. 3 — dois números diferentes para a mesma
   quantidade em dois capítulos.
4. **conjunto-inicial — AG: divergência de magnitude, causa já identificada.**
   +1,3 p.p. medido contra +5,2 p.p. relatado — CORREÇÃO desta auditoria:
   D-002 (`docs/decisoes.md`) documenta que o replay usa escala
   deliberadamente reduzida (N_pop=30/40 gerações vs. original 50/100); a
   divergência de magnitude é esperada por desenho, não um mistério a
   investigar. O número segue divergente, mas a causa está registrada.
5. **BERTimbau — dois hiperparâmetros do Cap. 3 não batem com o código, em
   nenhum lugar do repositório** (achado novo, R5/Cap. 3): taxa de
   aprendizado — tese diz 3×10⁻⁵, código usa 5×10⁻⁵ (default da classe, do
   notebook do E2 e do `train_full.py`, sem exceção); lote de treinamento —
   tese diz 32, código usa 16 (default) ou 128 (notebook real do E2), nunca
   32. Hipótese não confirmada: 32 é o default de `max_length` (comprimento
   de token), um parâmetro *diferente* no mesmo construtor — troca provável
   na redação, não confirmada.
6. **escolha-do-oraculo — o piso de 85% do critério de decisão não é
   atingido por nenhum oráculo** (achado novo, R5/Cap. 3). O Cap. 3 define a
   escolha do LLM Inicial como sujeita a acurácia mínima de 85% na S-rand;
   nenhum oráculo do E0 chega lá (melhor: 82,1%, deepseek-v4-pro). O próprio
   `experiments/e0/config.json` anota deepseek-v4-flash (78,3%) como
   "LLM Inicial candidato" — abaixo do piso declarado no texto.

## O que falta para fechar 100%

- Cache do oráculo (`annotation_cache_nemotron.jsonl`) — destrava E5, os
  braços A/B/C do E3′, e o cálculo de `|A|/|D|≈18%` do Cap. 3.
- Figuras de `experiments/plots/` — únicos itens de Cap. 4/5 ainda
  `sem-evidencia`.
- Três números do Cap. 3 sem artefato recuperável: janela de estagnação
  (p=5, ε=10⁻³), conjunto de |L| varrido no E2, calibração de lote do E0
  (1/10/25 por McNemar).
- Decisão do `principal`/autor sobre os seis achados acima.
"""))

notebook = {
    "cells": celulas,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

SAIDA.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("escrito:", SAIDA)
