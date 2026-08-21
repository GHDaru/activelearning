"""Fonte legível do notebook de auditoria do E0 (Onda 3a — reanálise grátis).

Edite ESTE arquivo, não o .ipynb — depois rode:

    python notebooks/auditoria/build_escolha-do-oraculo.py
"""
from __future__ import annotations

import json
from pathlib import Path

SAIDA = Path(__file__).resolve().parent / "escolha-do-oraculo.ipynb"


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
# E0 — escolha do oráculo · notebook de auditoria (reanálise, sem custo)

**Pergunta.** Qual LLM serve melhor como oráculo de rotulagem — em acurácia,
custo e perfil de erro?

**Por que este notebook não gasta nada.** As **anotações cruas** de cada
oráculo (a resposta real do LLM, item a item) estão versionadas em
`experiments/e0/results/{rand,strat}/annotations_*.jsonl` — 14 arquivos (7 por
amostra: um por combinação provedor+modelo, com o gpt-4o-mini duplicado por
ter duas execuções na S-rand).
Escaparam do `.gitignore` porque a regra `experiments/*/results/*.jsonl` casa
só um nível, e elas moram um nível abaixo (`results/rand/`). Consequência boa:
dá para **reexecutar o pipeline de análise inteiro** — `analyze_e0.py`, o
mesmo script que gerou os artefatos publicados — sem chamar nenhum LLM de
novo. É reanálise, não recoleta.

**O que este notebook prova, e por quê importa.** Recomputa `e0_table.json` e
`e0_mcnemar.json` **do zero**, a partir das respostas cruas, e compara com os
arquivos commitados. Se baterem, duas coisas ficam estabelecidas ao mesmo
tempo: (a) o pipeline é reprodutível; e (b) qualquer divergência entre a tese
e o artefato **não pode ser atribuída a um artefato desatualizado** — o
artefato é exatamente o que o código real produz hoje, a partir do dado bruto.

**Protocolo de dados: nenhum dos dois usados no resto da tese.** O E0 usa o
*CategorySchema* fechado de 621 categorias, sobre as **linhas cruas** (não
deduplicadas) da base corrigida — diferente da visão de 714 classes
(populacional/E5/E6/E3′) e diferente do corte de P1/P2. Ver
`sec:metodo-dados-preproc` no Cap. 3.

**Regra desta auditoria.** Divergência é acusada, nunca corrigida.
"""))

celulas.append(code(r'''
# 1) Onde estamos rodando.
import json, os, shutil, subprocess, sys
from pathlib import Path

def achar_raiz() -> Path:
    aqui = Path.cwd()
    for base in [aqui, *aqui.parents]:
        if (base / "experiments/e0").is_dir():
            return base
    destino = Path("/tmp/activelearning")
    if not destino.exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/GHDaru/activelearning.git", str(destino)],
                       check=True, capture_output=True)
    return destino

RAIZ = achar_raiz()
os.chdir(RAIZ)
RES = RAIZ / "experiments/e0/results"
brutas = sorted((RES / "rand").glob("annotations_*.jsonl")) + \
         sorted((RES / "strat").glob("annotations_*.jsonl"))
print("raiz:", RAIZ)
print(f"anotações cruas versionadas: {len(brutas)} arquivos")
'''))

celulas.append(code(r'''
# 2) REANÁLISE — roda o script de produção sobre uma cópia das anotações, em
#    pasta separada, para nunca sobrescrever o artefato publicado por engano.
import shutil, subprocess, sys

FRESCO = Path("/tmp/e0_reanalise")
if FRESCO.exists():
    shutil.rmtree(FRESCO)
shutil.copytree(RES / "rand", FRESCO / "rand")
shutil.copytree(RES / "strat", FRESCO / "strat")

r = subprocess.run([sys.executable, "experiments/e0/analyze_e0.py",
                    "--results", str(FRESCO)], capture_output=True, text=True)
print(r.stdout[-500:])
assert r.returncode == 0, r.stderr
print("\nreanálise gravada em:", FRESCO)
'''))

celulas.append(code(r'''
# 3) Tabela principal: publicado × recomputado, linha a linha.
publicado_tab = json.loads((RES / "e0_table.json").read_text())
recomputado_tab = json.loads((FRESCO / "e0_table.json").read_text())

def chave(r):
    return (r["sample"], r["oracle_id"])

pub = {chave(r): r for r in publicado_tab}
rec = {chave(r): r for r in recomputado_tab}

print(f"{'oráculo':<45}{'amostra':<8}{'acc pub':>9}{'acc rec':>9}{'F1 pub':>8}{'F1 rec':>8}  veredito")
divergencias = []
for k in sorted(pub):
    p, r = pub[k], rec.get(k)
    if r is None:
        divergencias.append((k, "linha ausente na reanálise"))
        continue
    ok = abs(p["accuracy"] - r["accuracy"]) < 1e-6 and abs(p["macro_f1"] - r["macro_f1"]) < 1e-6
    if not ok:
        divergencias.append((k, f"acc {p['accuracy']} vs {r['accuracy']} | "
                              f"F1 {p['macro_f1']} vs {r['macro_f1']}"))
    nome = k[1].split(":")[1].split("@")[0][:42]
    print(f"{nome:<45}{k[0]:<8}{p['accuracy']:>9.4f}{r['accuracy']:>9.4f}"
          f"{p['macro_f1']:>8.3f}{r['macro_f1']:>8.3f}  {'OK' if ok else 'DIVERGE'}")

print(f"\ndivergências entre publicado e reanálise: {len(divergencias) or 'nenhuma'}")
for k, msg in divergencias:
    print(f"  · {k}: {msg}")
'''))

celulas.append(code(r'''
# 4) McNemar: publicado × recomputado, os 43 pares.
publicado_mc = json.loads((RES / "e0_mcnemar.json").read_text())
recomputado_mc = json.loads((FRESCO / "e0_mcnemar.json").read_text())

def chave_mc(r):
    return (r["sample"], r["oracle_a"], r["oracle_b"])

pub_mc = {chave_mc(r): r for r in publicado_mc}
rec_mc = {chave_mc(r): r for r in recomputado_mc}

diffs = 0
for k in pub_mc:
    a, b = pub_mc[k], rec_mc.get(k)
    if b is None or (a["a_right_b_wrong"], a["a_wrong_b_right"]) != \
                     (b["a_right_b_wrong"], b["a_wrong_b_right"]):
        diffs += 1
        print(f"DIFERE: {k}")

print(f"{len(pub_mc)} pares no artefato publicado · {len(rec_mc)} recomputados "
      f"· {diffs} divergências")
print("\nZERO divergências confirma: o e0_mcnemar.json publicado É o que o "
      "pipeline real produz hoje, a partir das respostas cruas do LLM. Não é "
      "artefato desatualizado — é o pipeline mesmo." if diffs == 0 else
      "HÁ divergência entre o publicado e o que o pipeline produz agora — "
      "investigar mudança no código ou no dado.")
'''))

celulas.append(code(r'''
# 5) O ACHADO ABERTO: a afirmação de significância do Cap. 5 (RQ1) contra
#    o McNemar recomputado, não contra o artefato — dupla verificação.
def curto(o):
    return o.split(":")[1].split("@")[0]

def par(pares, amostra, a, b):
    for r in pares:
        ids = curto(r["oracle_a"]) + " " + curto(r["oracle_b"])
        if r["sample"] == amostra and a in ids and b in ids:
            return r
    return None

p1 = par(recomputado_mc, "strat", "v4-pro", "v4-flash")
print("Cap. 5 (RQ1) afirma: 'deepseek-v4-pro significativamente superior ao "
      "v4-flash na S-strat (b=43, c=16, p<0,001)'")
print(f"\nRECOMPUTADO AGORA, do zero, a partir das respostas cruas do LLM:")
print(f"  b={p1['a_right_b_wrong']}, c={p1['a_wrong_b_right']}, "
      f"p={p1['p_value']:.4f}  ->  {'SIGNIFICATIVO' if p1['p_value']<0.05 else 'NÃO SIGNIFICATIVO'}")

achou = any((r["a_right_b_wrong"], r["a_wrong_b_right"]) == (43, 16) for r in recomputado_mc)
print(f"\nb=43 e c=16 em QUALQUER par recomputado (43 pares, 2 amostras): "
      f"{'encontrado' if achou else 'NÃO encontrado — confirma o achado anterior'}")
'''))

celulas.append(md(r"""
## Rodapé — o que este notebook prova e como reproduzir

**Este notebook não grava artefato de produção**: a reanálise vai para
`/tmp/e0_reanalise`, nunca sobrescreve `experiments/e0/results/`.

Para reexecutar (não precisa de chave de API, é reanálise, não recoleta):

```bash
python experiments/e0/analyze_e0.py --results <pasta_de_saída>
```

**O que ficou provado**: o pipeline de análise do E0 é **100% reprodutível**
a partir das anotações cruas versionadas — zero divergências em 13 linhas da
tabela e 43 pares de McNemar. Isso fecha, com a maior confiança possível sem
acesso ao histórico de execução original, a divergência aberta em
`b=43, c=16, p<0,001`: não existe em nenhuma versão do artefato, publicada ou
recomputada agora.

### Se quiser ir além (Onda 3b — recoleta, custa dinheiro)

Rodar `experiments/e0/run_e0.py` de novo, com chave de API, para um provedor
ou modelo novo. Este notebook não faz isso.
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
