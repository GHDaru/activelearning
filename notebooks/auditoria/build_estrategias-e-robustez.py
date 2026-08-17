"""Fonte legível do notebook de auditoria do E1/E4 (Onda 4).

Edite ESTE arquivo, não o .ipynb — depois rode:

    python notebooks/auditoria/build_estrategias-e-robustez.py
"""
from __future__ import annotations

import json
from pathlib import Path

SAIDA = Path(__file__).resolve().parent / "estrategias-e-robustez.ipynb"


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
# E1/E4 — estratégias de seleção e robustez ao ruído · notebook de auditoria

**Pergunta.** Qual estratégia de seleção ativa vence com oráculo perfeito
(E1)? E o valor da seleção sobrevive quando o oráculo erra na taxa dos LLMs
reais (E4)?

**Este notebook difere do E0/E0-P: não é reanálise, é reexecução.** O
`sweeps.jsonl` — as 104 células brutas (5 estratégias × 8 sementes + ablação
de lote + E4: 3 níveis de ruído × 2 estratégias × 8 sementes) — **nunca foi
versionado**; só sobreviveram os agregados (`analysis.json`, `baseline.json`).
Achado do lote 2 da Etapa 2. Sem o bruto, não há reanálise possível: este
notebook roda `run_sweeps.py` de novo, do zero. CPU, ~30 min (medido: ~18 s
por célula × 104).

**Protocolo de dados: outro corte, específico deste experimento.** Pool de
20.000 (amostrado da base deduplicada, semente 7) e teste de 5.000 — nem o
CategorySchema de 621 do E0, nem a visão de 714 do populacional, nem o corte
de P1/P2. Ver `sec:metodo-falco-baselines` no Cap. 3.

**Regra desta auditoria.** Divergência é acusada, nunca corrigida.
"""))

celulas.append(code(r'''
# 1) Onde estamos rodando.
import json, os, subprocess, sys
from pathlib import Path

def achar_raiz() -> Path:
    aqui = Path.cwd()
    for base in [aqui, *aqui.parents]:
        if (base / "experiments/e1e4").is_dir():
            return base
    destino = Path("/tmp/activelearning")
    if not destino.exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/GHDaru/activelearning.git", str(destino)],
                       check=True, capture_output=True)
    return destino

RAIZ = achar_raiz()
os.chdir(RAIZ)
RES = RAIZ / "experiments/e1e4/results"
print("raiz:", RAIZ)
'''))

celulas.append(code(r'''
# 2) EXECUÇÃO — reexecução completa, não reanálise. Retomável: célula já
#    presente em sweeps.jsonl é pulada, então reexecutar após queda é seguro.
#    ~30 min em CPU (medido: ~18s/célula × 104). Não gasta cota de GPU.
import subprocess, sys, time

t0 = time.time()
proc = subprocess.Popen([sys.executable, "experiments/e1e4/run_sweeps.py"],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for linha in proc.stdout:
    print(linha, end="", flush=True)
rc = proc.wait()
assert rc == 0, f"run_sweeps.py terminou com código {rc}"
print(f"\ntotal: {(time.time()-t0)/60:.1f} min")
'''))

celulas.append(code(r'''
# 3) E1 — as cinco estratégias, recalculadas do sweeps.jsonl (média ± desvio
#    em 8 sementes), confrontadas com a Tabela e1 do Cap. 5.
import statistics as est

sweeps = [json.loads(l) for l in (RES / "sweeps.jsonl").read_text().splitlines()]
e1 = [r for r in sweeps if r["exp"] == "e1"]

publicado = {
    "smallest_margin": (0.528, 0.013, 0.418, 0.013),
    "least_confidence": (0.518, 0.010, 0.421, 0.009),
    "entropy": (0.493, 0.006, 0.398, 0.008),
    "hybrid": (0.476, 0.014, 0.379, 0.008),
    "random": (0.444, 0.011, 0.339, 0.006),
}

print(f"{'estratégia':<18}{'n':>3}{'LCE rec':>10}{'F1 rec':>10}{'LCE pub':>10}{'F1 pub':>10}  veredito")
divergencias = []
for estrategia, (lce_p, lce_sd_p, f1_p, f1_sd_p) in publicado.items():
    execs = [r for r in e1 if r["strategy"] == estrategia]
    if len(execs) < 8:
        print(f"{estrategia:<18}{len(execs):>3}  incompleto — pule esta linha por ora")
        continue
    lce_m = est.mean(r["lce"] for r in execs)
    f1_m = est.mean(r["final_macro_f1"] for r in execs)
    ok = abs(lce_m - lce_p) < 0.02 and abs(f1_m - f1_p) < 0.02   # folga: nova amostragem do pool
    if not ok:
        divergencias.append((estrategia, lce_m, f1_m, lce_p, f1_p))
    print(f"{estrategia:<18}{len(execs):>3}{lce_m:>10.4f}{f1_m:>10.4f}"
          f"{lce_p:>10.3f}{f1_p:>10.3f}  {'OK' if ok else 'DIVERGE'}")

print(f"\ndivergências (folga 0,02 — o pool é reamostrado, não é o mesmo do "
      f"artefato original): {len(divergencias) or 'nenhuma'}")
print("nota: folga maior que nos outros notebooks porque run_sweeps.py monta o "
      "pool com random.Random(7).sample(...) a cada execução — não há garantia "
      "de que seja o MESMO pool de 20k do artefato original, só o mesmo desenho.")
'''))

celulas.append(code(r'''
# 4) E4 — robustez ao ruído, mesma lógica.
e4 = [r for r in sweeps if r["exp"] == "e4"]
pub_e4 = {
    (0.1, "entropy"): 0.347, (0.1, "random"): 0.294,
    (0.2, "entropy"): 0.294, (0.2, "random"): 0.252,
    (0.4, "entropy"): 0.215, (0.4, "random"): 0.186,
}
print(f"{'ε':<6}{'estratégia':<12}{'n':>3}{'F1 rec':>10}{'F1 pub':>10}  veredito")
for (eps, estrategia), f1_p in pub_e4.items():
    execs = [r for r in e4 if r["noise"] == eps and r["strategy"] == estrategia]
    if len(execs) < 8:
        print(f"{eps:<6}{estrategia:<12}{len(execs):>3}  incompleto")
        continue
    f1_m = est.mean(r["final_macro_f1"] for r in execs)
    ok = abs(f1_m - f1_p) < 0.02
    print(f"{eps:<6}{estrategia:<12}{len(execs):>3}{f1_m:>10.4f}{f1_p:>10.3f}  "
          f"{'OK' if ok else 'DIVERGE'}")
'''))

celulas.append(code(r'''
# 5) Gráfico: curvas de aprendizado da varredura E1 (uma semente por
#    estratégia, para não poluir — a tabela acima já deu a média das 8).
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

COR = {"entropy": "#2a78d6", "least_confidence": "#eb6834", "smallest_margin": "#1baf7a",
       "hybrid": "#eda100", "random": "#e87ba4"}
TINTA, TINTA2, GRADE = "#0b0b0b", "#52514e", "#d8d7d2"
# :.2f, não :.1f — o matplotlib escolhe passo de 0,05 aqui, e uma casa decimal
# duplicava o rótulo (0,05 e 0,10 caíam ambos em "0,1"). Só apareceu ao OLHAR
# a figura renderizada, não ao ler o código.
vg = FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ","))

plt.rcParams.update({"figure.dpi": 120, "font.size": 9,
                     "axes.edgecolor": GRADE, "axes.labelcolor": TINTA2,
                     "xtick.color": TINTA2, "ytick.color": TINTA2,
                     "axes.spines.top": False, "axes.spines.right": False})

fig, eixo = plt.subplots(figsize=(7.5, 4.3))
for estrategia, cor in COR.items():
    r = next((x for x in e1 if x["strategy"] == estrategia and x["seed"] == 0), None)
    if not r:
        continue
    xs = [p[0] for p in r["curve"]]
    ys = [p[1] for p in r["curve"]]
    eixo.plot(xs, ys, color=cor, linewidth=2, label=estrategia.replace("_", " "))
eixo.set_xlabel("rótulos gastos")
eixo.set_ylabel("Macro F1")
eixo.yaxis.set_major_formatter(vg)
eixo.set_title("E1 · curvas de aprendizado (semente 0)", color=TINTA, fontsize=11, loc="left")
eixo.grid(True, color=GRADE, linewidth=0.6, alpha=0.7)
eixo.set_axisbelow(True)
eixo.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout()
plt.show()
'''))

celulas.append(md(r"""
## Rodapé

Este notebook **grava** `experiments/e1e4/results/sweeps.jsonl` — o artefato
que nunca existiu no repositório. **Use `git add -f`**: a linha 7 do
`.gitignore` casa com ele.

Para reexecutar do zero:

```bash
rm experiments/e1e4/results/sweeps.jsonl
python experiments/e1e4/run_sweeps.py   # ~30 min, retomável
```
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
