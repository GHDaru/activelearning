"""Fonte legível do notebook de auditoria do E0-P (Onda 3a — reanálise grátis).

Edite ESTE arquivo, não o .ipynb — depois rode:

    python notebooks/auditoria/build_efeito-do-prompt.py
"""
from __future__ import annotations

import json
from pathlib import Path

SAIDA = Path(__file__).resolve().parent / "efeito-do-prompt.ipynb"


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
# E0-P — o prompt como variável do instrumento · notebook de auditoria

**Pergunta.** Regras de fronteira explícitas no *prompt* melhoram o oráculo
mais barato (gpt-4o-mini)? E o ganho medido numa distribuição se sustenta na
outra?

**Por que este notebook não gasta nada.** As anotações cruas das três
variantes (v3 base, v4a regras, v4b regras+exemplos) já estão versionadas:
v4a/v4b em `experiments/e0p/results/{rand,strat}/`, e a v3 é **reaproveitada**
das anotações do próprio E0
(`experiments/e0/results/{rand,strat}/annotations_openai_gpt-4o-mini_T0.0_b10.jsonl`,
restrita aos primeiros 500 itens — é assim que `analyze_e0p.py` já faz, e é
por isso que os dois experimentos precisam da mesma amostra e do mesmo
`config.json`). Reanálise, não recoleta.

**Protocolo de dados**: o mesmo do E0 — CategorySchema de 621 categorias sobre
linhas cruas. Ver `sec:metodo-dados-preproc` no Cap. 3.

**Regra desta auditoria.** Divergência é acusada, nunca corrigida.
"""))

celulas.append(code(r'''
# 1) Onde estamos rodando.
import json, os, shutil, subprocess, sys
from pathlib import Path

def achar_raiz() -> Path:
    aqui = Path.cwd()
    for base in [aqui, *aqui.parents]:
        if (base / "experiments/e0p").is_dir():
            return base
    destino = Path("/tmp/activelearning")
    if not destino.exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/GHDaru/activelearning.git", str(destino)],
                       check=True, capture_output=True)
    return destino

RAIZ = achar_raiz()
os.chdir(RAIZ)
RES = RAIZ / "experiments/e0p/results"
brutas_e0p = sorted(RES.glob("*/annotations_*.jsonl"))
brutas_v3 = sorted((RAIZ / "experiments/e0/results").glob(
    "*/annotations_openai_gpt-4o-mini_T0.0_b10.jsonl"))
print("raiz:", RAIZ)
print(f"anotações v4a/v4b (E0-P): {len(brutas_e0p)} · v3 reaproveitada do E0: {len(brutas_v3)}")
'''))

celulas.append(code(r'''
# 2) REANÁLISE — roda o script real de produção sobre cópias das anotações,
#    em pasta separada, e reaproveita as anotações v3 do E0 (link/copy).
FRESCO = Path("/tmp/e0p_reanalise")
if FRESCO.exists():
    shutil.rmtree(FRESCO)
# a estrutura que analyze_e0p.py espera: experiments/e0/results/<sample>/... (v3)
# e experiments/e0p/results/<sample>/... (v4a/v4b) — replicamos ao lado, sem
# tocar o repositório real.
(FRESCO / "experiments/e0/results").mkdir(parents=True)
(FRESCO / "experiments/e0p/results").mkdir(parents=True)
(FRESCO / "experiments/e0/config.json").write_text(
    (RAIZ / "experiments/e0/config.json").read_text())
(FRESCO / "data").mkdir(parents=True)
shutil.copy(RAIZ / "data/dataset.csv", FRESCO / "data/dataset.csv")
shutil.copytree(RAIZ / "src", FRESCO / "src")
shutil.copytree(RAIZ / "experiments/e0", FRESCO / "experiments/e0", dirs_exist_ok=True)
shutil.copytree(RAIZ / "experiments/e0p", FRESCO / "experiments/e0p", dirs_exist_ok=True)

r = subprocess.run([sys.executable, str(FRESCO / "experiments/e0p/analyze_e0p.py")],
                   cwd=FRESCO, capture_output=True, text=True)
print(r.stdout[-400:])
if r.returncode != 0:
    print(r.stderr[-2000:])
assert r.returncode == 0
print("\nreanálise gravada em:", FRESCO / "experiments/e0p/results/analysis.json")
'''))

celulas.append(code(r'''
# 3) Publicado × recomputado, linha a linha.
publicado = json.loads((RES / "analysis.json").read_text())
recomputado = json.loads((FRESCO / "experiments/e0p/results/analysis.json").read_text())

print(f"{'amostra':<8}{'variante':<10}{'acc pub':>9}{'acc rec':>9}  veredito")
divergencias = []
for amostra in ("rand", "strat"):
    for variante in ("v3", "v4a", "v4b"):
        p = publicado[amostra][variante]
        r = recomputado[amostra][variante]
        ok = p["n_correct"] == r["n_correct"] and abs(p["accuracy"] - r["accuracy"]) < 1e-6
        if not ok:
            divergencias.append((amostra, variante, p, r))
        print(f"{amostra:<8}{variante:<10}{p['accuracy']:>9.4f}{r['accuracy']:>9.4f}  "
              f"{'OK' if ok else 'DIVERGE'}")

print(f"\n{'amostra':<8}{'par':<14}{'b pub':>6}{'c pub':>6}{'b rec':>6}{'c rec':>6}{'p pub':>9}{'p rec':>9}")
for amostra in ("rand", "strat"):
    for par in ("v3_vs_v4a", "v3_vs_v4b", "v4a_vs_v4b"):
        p = publicado[amostra][f"mcnemar_{par}"]
        r = recomputado[amostra][f"mcnemar_{par}"]
        ok = (p["only_a_correct"], p["only_b_correct"]) == (r["only_a_correct"], r["only_b_correct"])
        if not ok:
            divergencias.append((amostra, par, p, r))
        print(f"{amostra:<8}{par:<14}{p['only_a_correct']:>6}{p['only_b_correct']:>6}"
              f"{r['only_a_correct']:>6}{r['only_b_correct']:>6}{p['p_value']:>9.4f}"
              f"{r['p_value']:>9.4f}  {'OK' if ok else 'DIVERGE'}")

print(f"\ndivergências: {len(divergencias) or 'nenhuma'}")
'''))

celulas.append(md(r"""
## Rodapé

**Não grava artefato de produção**: a reanálise vai para `/tmp/e0p_reanalise`.

Para reexecutar (sem chave de API, é reanálise):

```bash
python experiments/e0p/analyze_e0p.py
```

Zero divergências aqui, somado ao E0, fecha a Onda 3a: os dois pipelines de
oráculo são reprodutíveis ponta a ponta a partir do dado bruto versionado.
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
