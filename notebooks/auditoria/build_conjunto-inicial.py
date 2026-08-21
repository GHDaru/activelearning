"""Fonte legível do notebook de auditoria do P1/P2 (Cap. 4 — conjunto inicial).

Edite ESTE arquivo, não o .ipynb — depois rode:

    python notebooks/auditoria/build_conjunto-inicial.py
"""
from __future__ import annotations

import json
from pathlib import Path

SAIDA = Path(__file__).resolve().parent / "conjunto-inicial.ipynb"


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
# P1/P2 — composição do conjunto inicial · notebook de auditoria

**Pergunta.** O tamanho e a composição de $L_0$ (o conjunto inicial, antes de
qualquer aprendizado ativo) afetam o desempenho? E dá para fazer melhor que
sortear?

**Hipótese.** No regime de poucos rótulos existem $L_0$ bons e ruins do
*mesmo* tamanho (P1); um algoritmo genético consegue encontrar composições
melhores que o sorteio (P2), mas o ganho da otimização evolutiva pode estar
inflado se a mesma partição usada para pontuar o indivíduo também o avalia
(circularidade).

**Este notebook resolve uma dívida, não reproduz um número publicado.** Os
resultados do Cap. 4 vêm do programa experimental original — fora deste
repositório — e o relatório que os declarava "já auditados"
(`docs/convergencia-replays.md`) citava dois artefatos que **nunca foram
commitados**: `experiments/p1/results/replay_l0.jsonl` e `replay_ga.jsonl`.
Achado do `executor01`, 2026-08-17. Este notebook **gera esses artefatos de
verdade** e os compara ao original — pela primeira vez de forma auditável.

**Protocolo de dados: nenhum dos dois** protocolos do resto da tese. P1/P2
usam a base **corrigida, sem deduplicação por classe rara** (250.221 linhas),
particionada por este próprio script (teste 20%, capado em 20 mil por custo de
avaliação; o resto é o *pool*). Não é o *CategorySchema* de 621 categorias
(esse é do E0) nem a visão de 714 classes (populacional/E3′/E6) — é outro
corte, específico deste experimento. Ver `sec:metodo-dados-preproc` no Cap. 3.

**Regra desta auditoria.** Divergência é acusada, nunca corrigida.
"""))

celulas.append(code(r'''
# 1) Onde estamos rodando.
import json, os, subprocess, sys
from pathlib import Path

def achar_raiz() -> Path:
    aqui = Path.cwd()
    for base in [aqui, *aqui.parents]:
        if (base / "experiments/p1").is_dir():
            return base
    destino = Path("/tmp/activelearning")
    if not destino.exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/GHDaru/activelearning.git", str(destino)],
                       check=True, capture_output=True)
    return destino

RAIZ = achar_raiz()
os.chdir(RAIZ)
RES = RAIZ / "experiments/p1/results"
print("raiz:", RAIZ)
'''))

celulas.append(code(r'''
# 2) EXECUÇÃO — não é auditoria de artefato pronto, é geração dele.
#    Retomável (P1) e incremental por célula (P2); reexecutar após queda é
#    seguro. Em CPU: P1 leva ~15-20 min (150 ajustes, os maiores de ~4-6 s);
#    P2 leva ~3-4 min (4 células de 40 gerações × 30 indivíduos).
import subprocess, sys, time

t0 = time.time()
for script in ["experiments/p1/replay_l0_sensitivity.py", "experiments/p1/replay_ga.py"]:
    print(f"\n=== {script} ===", flush=True)
    proc = subprocess.Popen([sys.executable, script], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for linha in proc.stdout:
        print(linha, end="", flush=True)
    rc = proc.wait()
    assert rc == 0, f"{script} terminou com código {rc}"
print(f"\ntotal: {(time.time()-t0)/60:.1f} min")
'''))

celulas.append(code(r'''
# 3) P1 — sensibilidade: replay (15 tamanhos × 10 reps) vs. original
#    (47 tamanhos × 30 reps). Divergência é medida ponto a ponto nos tamanhos
#    em comum, não estimada.
import statistics as est

replay = [json.loads(l) for l in (RES / "replay_l0.jsonl").read_text().splitlines()]
por_tamanho = {}
for r in replay:
    por_tamanho.setdefault(r["size"], []).append(r)

# (tamanho, acurácia média publicada no Cap. 4, %)
publicado_acc = {10: 6.7, 100: 24.7, 1_000: 55.9, 10_000: 76.9, 200_000: 89.1}

print(f"{'|L0|':>8}{'acc. replay':>13}{'acc. publicada':>16}{'Δ p.p.':>9}{'n reps':>8}  veredito")
divergencias = []
for tam in sorted(por_tamanho):
    execs = por_tamanho[tam]
    media = est.mean(e["accuracy"] for e in execs) * 100
    pub = publicado_acc.get(tam)
    if pub is None:
        print(f"{tam:>8}{media:>12.1f}%{'—':>16}{'—':>9}{len(execs):>8}  (sem par no Cap. 4)")
        continue
    delta = media - pub
    ok = abs(delta) <= 1.0        # tolerância do próprio relatório de convergência: 0,7 p.p.
    if not ok:
        divergencias.append((tam, media, pub, delta))
    print(f"{tam:>8}{media:>12.1f}%{pub:>15.1f}%{delta:>+8.1f}{len(execs):>8}  "
          f"{'OK' if ok else 'DIVERGE'}")

print(f"\ndivergências (>1,0 p.p.): {len(divergencias) or 'nenhuma'}")
print("critério: o relatório de convergência original (agora sem artefato) "
      "declarava divergência máxima de 0,7 p.p. — usamos 1,0 p.p. de folga.")
'''))

celulas.append(code(r'''
# 4) P1 — a variabilidade em I=100 (a "evidência direta" do Cap. 4).
execs_100 = [e["accuracy"] for e in por_tamanho.get(100, [])]
if execs_100:
    print(f"I=100: acurácia entre repetições de {min(execs_100)*100:.1f}% "
          f"a {max(execs_100)*100:.1f}% — amplitude de "
          f"{(max(execs_100)-min(execs_100))*100:.1f} p.p.")
    print("Cap. 4 (I=100, 47×30 original): amplitude de 6,4 p.p. (21,7% a 28,1%)")
    print("Aqui a amostra é menor (10 reps vs. 30): a amplitude tende a ser menor "
          "por construção, não é uma refutação do fenômeno.")
'''))

celulas.append(code(r'''
# 5) P2 — o AG: mecanismo reproduzido + a circularidade quantificada de novo.
ga = [json.loads(l) for l in (RES / "replay_ga.jsonl").read_text().splitlines()]

print(f"{'cenário':<10}{'|L0|':>6}{'aptidão (partição)':>20}{'reaval. (teste)':>18}{'inflação':>11}")
inflacao = {}
for r in ga:
    infl = (r["fitness_best"] - r["test_reeval"]) * 100
    inflacao[(r["scenario"], r["size"])] = infl
    print(f"{r['scenario']:<10}{r['size']:>6}{r['fitness_best']:>19.4f}"
          f"{r['test_reeval']:>18.4f}{infl:>+10.1f} p.p.")

# achado original do relatório perdido: max_f1 I=500, inflação de 6,3 p.p.
alvo = inflacao.get(("max_f1", 500))
print(f"\nachado do relatório original (agora sem artefato): max_f1 I=500, "
      f"inflação de +6,3 p.p. (19,4% na partição de aptidão vs. 13,1% no teste)")
if alvo is not None:
    print(f"replay: inflação de {alvo:+.1f} p.p.")
    print("mesma direção e ordem de grandeza" if 3 <= alvo <= 10 else
          "ORDEM DE GRANDEZA DIFERENTE — investigar")
'''))

celulas.append(code(r'''
# 6) P2 — o mecanismo em si: o AG bate a média aleatória em I=50?
#    A média aleatória em macro F1, no mesmo tamanho, sai do próprio replay do P1.
f1_aleatorio_50 = [e["macro_f1"] for e in por_tamanho.get(50, [])]
ga_max_f1_50 = next((r for r in ga if r["scenario"] == "max_f1" and r["size"] == 50), None)

if f1_aleatorio_50 and ga_max_f1_50:
    media_aleatoria = sum(f1_aleatorio_50) / len(f1_aleatorio_50)
    ganho = (ga_max_f1_50["test_reeval"] - media_aleatoria) * 100
    print(f"I=50, Macro F1: aleatório (média, {len(f1_aleatorio_50)} reps) = "
          f"{media_aleatoria:.4f} | AG (reavaliado no teste) = "
          f"{ga_max_f1_50['test_reeval']:.4f}")
    print(f"ganho do AG sobre a média aleatória: {ganho:+.1f} p.p.")
    print("relatório original (sem artefato): +5,2 p.p. em I=50")
    print("mesma direção" if ganho > 0 else "MECANISMO NÃO REPRODUZIDO: AG não superou a média")
'''))

celulas.append(code(r'''
# 7) Gráficos.
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

AZUL, LARANJA, VERDE = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA2, GRADE = "#0b0b0b", "#52514e", "#d8d7d2"
vg = FuncFormatter(lambda v, _: f"{v:.0f}".replace(".", ","))

plt.rcParams.update({"figure.dpi": 120, "font.size": 9,
                     "axes.edgecolor": GRADE, "axes.labelcolor": TINTA2,
                     "xtick.color": TINTA2, "ytick.color": TINTA2,
                     "axes.spines.top": False, "axes.spines.right": False})

fig, eixo = plt.subplots(figsize=(7.5, 4.3))
tamanhos = sorted(por_tamanho)
medias = [est.mean(e["accuracy"] for e in por_tamanho[t]) * 100 for t in tamanhos]
minimos = [min(e["accuracy"] for e in por_tamanho[t]) * 100 for t in tamanhos]
maximos = [max(e["accuracy"] for e in por_tamanho[t]) * 100 for t in tamanhos]
eixo.fill_between(tamanhos, minimos, maximos, color=AZUL, alpha=0.15, linewidth=0)
eixo.plot(tamanhos, medias, color=AZUL, linewidth=2, marker="o", markersize=5,
          markeredgecolor="#fcfcfb", markeredgewidth=1, label="replay (10 reps)")
tam_pub = sorted(publicado_acc)
eixo.plot(tam_pub, [publicado_acc[t] for t in tam_pub], color=LARANJA, linewidth=0,
          marker="D", markersize=7, markeredgecolor="#fcfcfb", markeredgewidth=1,
          label="publicado no Cap. 4 (30 reps)", zorder=5)
eixo.set_xscale("log")
eixo.set_xlabel("|L0| (escala log)")
eixo.set_ylabel("acurácia (%)")
eixo.yaxis.set_major_formatter(vg)
eixo.set_title("P1 · sensibilidade à composição de L0 — replay vs. publicado",
               color=TINTA, fontsize=11, loc="left")
eixo.grid(True, which="both", color=GRADE, linewidth=0.5, alpha=0.6)
eixo.set_axisbelow(True)
eixo.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout()
plt.show()
'''))

celulas.append(md(r"""
## Rodapé — o que sai daqui e como reproduzir

Este notebook **grava** dois artefatos que nunca existiram no repositório:

```
experiments/p1/results/replay_l0.jsonl
experiments/p1/results/replay_ga.jsonl
```

**Atenção ao commitar**: a linha 7 do `.gitignore`
(`experiments/*/results/*.jsonl`) casa com os dois. Use
`git add -f experiments/p1/results/*.jsonl`, ou eles somem de novo — pela
segunda vez, desta vez sem desculpa.

Para reexecutar do zero (apaga o que já foi calculado):

```bash
rm experiments/p1/results/replay_l0.jsonl experiments/p1/results/replay_ga.jsonl
python experiments/p1/replay_l0_sensitivity.py   # ~15-20 min, retomável
python experiments/p1/replay_ga.py               # ~3-4 min
```

Sem GPU: os dois usam o PVBin, que é CPU. Não gaste cota de GPU do Kaggle
com este notebook.

### Se alguma célula acusou divergência

Poste bloqueio ao agente `principal`. Note que este notebook compara contra
números **publicados na tese**, não contra outro artefato — não há "fonte da
verdade" recuperável para o experimento original em si, só o relatório de
convergência (`docs/convergencia-replays.md`), que também não tinha artefato.
Se o replay divergir muito do publicado, o que se sabe piora, não melhora: nem
o original nem o replay ficam auditáveis sozinhos.
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
