"""Fonte legível do notebook de auditoria do E6.

Edite ESTE arquivo, não o .ipynb — depois rode:

    python notebooks/auditoria/build_e6.py

Convenção adotada do `build_nb.py` do executor02: o notebook é artefato
gerado; o código revisável é o Python aqui.
"""
from __future__ import annotations

import json
from pathlib import Path

SAIDA = Path(__file__).resolve().parent / "e6-populacao.ipynb"


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
# E6 — seletores em escala populacional · notebook de auditoria

**Pergunta.** Uma estratégia de seleção ativa (entropia) alcança o mesmo
desempenho de amostragem aleatória com menos rótulos? E quanto um praticante
*sem conjunto reservado* se engana ao medir o próprio modelo?

**Hipótese.** A entropia satura com bem menos rótulos que o aleatório, e a
avaliação *interna* (nos 20% do que o próprio processo rotulou) **superestima**
o Macro F1 e **subestima** a acurácia frente à população reservada.

**Artefatos que este notebook audita** (já no repositório, nada é recalculado
do zero):

| Arquivo | O que guarda |
|---|---|
| `experiments/e6population/results/popcurve_<clf>_<estrategia>.jsonl` | a curva ponto a ponto (lote de 500) |
| `…/popcurve_*_summary.json` | metadados da execução (pool, população, tempo) |
| `…/analysis.json` | tetos, saturações e viés — semente única |
| `…/analysis_multiseed.json` | a campanha de 8 sementes (entropia × aleatório) |

**Protocolo de dados: o POPULACIONAL.** A tese carrega dois espaços de rótulos,
e confundi-los é o erro mais fácil de cometer aqui. Este experimento usa a visão
**deduplicada** com filtro brando (≥2 instâncias): **231.490 textos, 714
classes**, das quais as primeiras 50.000 (embaralhadas com semente 42) formam o
*pool* e **todo o restante** é a população reservada. O outro espaço — o
*CategorySchema* de 621 categorias sobre as linhas cruas — governa os
experimentos de oráculo (E0, E0-P) e **não** vale aqui.

**Regra desta auditoria.** Onde o número recalculado divergir do publicado, o
notebook **acusa e não corrige**. Ajustar número para casar é falsificar
auditoria.
"""))

celulas.append(code(r'''
# 1) Onde estamos rodando. Funciona igual na máquina local e no Kaggle.
import json, os, subprocess, sys
from pathlib import Path

def achar_raiz() -> Path:
    aqui = Path.cwd()
    for base in [aqui, *aqui.parents]:
        if (base / "experiments/e6population").is_dir():
            return base
    # no Kaggle o repositório ainda não existe: clona (é público)
    destino = Path("/tmp/activelearning")
    if not destino.exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/GHDaru/activelearning.git", str(destino)],
                       check=True, capture_output=True)
    return destino

RAIZ = achar_raiz()
os.chdir(RAIZ)
RES = RAIZ / "experiments/e6population/results"
print("raiz:", RAIZ)
print("artefatos do E6:", len(list(RES.glob("*"))), "arquivos")
'''))

celulas.append(code(r'''
# 2) IDENTIDADE DOS DADOS — o portão da auditoria.
#
#    Nenhum número abaixo significa coisa alguma se a base não for a mesma. O
#    md5 prova o arquivo; a contagem prova o pré-processamento; as partições
#    provam o particionamento. Se qualquer linha der DIVERGE, pare de ler o
#    resto do notebook.
import csv, hashlib, random
from collections import Counter

CSV = RAIZ / "data/dataset.csv"
DATA_SEED, POOL_SIZE = 42, 50_000

md5 = hashlib.md5(CSV.read_bytes()).hexdigest()

sys.path.insert(0, str(RAIZ / "src"))
from activelearning.domain.instances import normalize_label

linhas = []
with CSV.open(encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        linhas.append((r["nm_item"], normalize_label(r["nm_product"])))
contagem = Counter(l for _, l in linhas)
filtradas = [(t, l) for t, l in linhas if contagem[l] >= 2]      # filtro brando
vistos, dedup = set(), []
for t, l in filtradas:
    k = t.strip().lower()
    if k not in vistos:
        vistos.add(k)
        dedup.append((t, l))
random.Random(DATA_SEED).shuffle(dedup)
pool, populacao = dedup[:POOL_SIZE], dedup[POOL_SIZE:]

sumario = json.loads((RES / "popcurve_pvbin_entropy_summary.json").read_text())

esperado = [
    ("md5 do dataset.csv",   md5,              None),
    ("linhas cruas",         len(linhas),      None),
    ("textos deduplicados",  len(dedup),       231_490),
    ("classes presentes",    len({l for _, l in dedup}), 714),
    ("pool",                 len(pool),        50_000),
    ("população reservada",  len(populacao),   sumario["population"]),
]
print(f"{'o quê':<24}{'medido agora':>16}{'esperado':>12}  veredito")
falhou = False
for nome, medido, alvo in esperado:
    if alvo is None:
        print(f"{nome:<24}{str(medido):>16}{'—':>12}  (registro)")
        continue
    ok = medido == alvo
    falhou |= not ok
    print(f"{nome:<24}{medido:>16}{alvo:>12}  {'OK' if ok else 'DIVERGE'}")
print("\nPORTÃO:", "FECHADO — não confie no resto" if falhou else "ABERTO — a base é a mesma")
'''))

celulas.append(code(r'''
# 3) Carrega as curvas publicadas e recalcula teto e saturação a partir DELAS.
#
#    A saturação é "o primeiro |L| em que o Macro F1 externo alcança 95% do
#    teto da própria curva". Recalcular a partir do JSONL é o que transforma o
#    analysis.json de afirmação em resultado conferível.
def curva(nome):
    caminho = RES / f"popcurve_{nome}.jsonl"
    return [json.loads(l) for l in caminho.read_text(encoding="utf-8").splitlines() if l.strip()]

def teto_e_saturacao(pontos):
    teto = max(p["f1_ext"] for p in pontos)
    alvo = 0.95 * teto
    return round(teto, 4), next((p["n_labels"] for p in pontos if p["f1_ext"] >= alvo), None)

publicado = json.loads((RES / "analysis.json").read_text())
BRACOS = ["pvbin_entropy", "pvbin_random", "pvbin_drisl", "pvbin_drisl-c",
          "sgd_entropy", "sgd_random", "sgd_drisl", "sgd_drisl-c"]

print(f"{'braço':<16}{'teto pub':>9}{'teto rec':>9}{'sat pub':>9}{'sat rec':>9}  veredito")
divergencias = []
for nome in BRACOS:
    if nome not in publicado or not (RES / f"popcurve_{nome}.jsonl").exists():
        continue
    teto, sat = teto_e_saturacao(curva(nome))
    tp = publicado[nome]["f1_ext_teto"]
    sp = publicado[nome]["saturacao_95pct_teto"]
    ok = abs(teto - tp) < 1e-4 and sat == sp
    if not ok:
        divergencias.append((nome, tp, teto, sp, sat))
    print(f"{nome:<16}{tp:>9.4f}{teto:>9.4f}{sp:>9}{sat:>9}  {'OK' if ok else 'DIVERGE'}")
print(f"\ndivergências: {len(divergencias) or 'nenhuma'}")
'''))

celulas.append(code(r'''
# 4) A campanha de 8 sementes, recalculada dos JSONL das próprias sementes.
#    É o que sustenta a inferência do capítulo (a semente única é descritiva).
import statistics as est

# A campanha são as sementes s43..s50 e NÃO inclui a original (42). Isso não é
# detalhe: incluí-la faz as quatro médias divergirem do publicado (sgd_entropy
# cai de 9062 para 8944), e a leitura errada seria "o artefato não reproduz".
# A separação é proposital — a semente 42 é a descritiva da Tabela e6, e a
# campanha são 8 execuções independentes. Confere com o texto do capítulo, que
# situa os 8.000 da semente original na "borda otimista" da faixa (mín. 8.500).
SEMENTES = [f"_s{n}" for n in range(43, 51)]
multi = json.loads((RES / "analysis_multiseed.json").read_text())
assert multi["n_sementes"] == len(SEMENTES), "a campanha mudou de tamanho"

def campanha(clf, estrategia):
    tetos, sats = [], []
    for s in SEMENTES:
        caminho = RES / f"popcurve_{clf}_{estrategia}{s}.jsonl"
        if not caminho.exists():
            continue
        t, sat = teto_e_saturacao(curva(f"{clf}_{estrategia}{s}"))
        tetos.append(t); sats.append(sat)
    return tetos, sats

print(f"{'braço':<16}{'n':>3}{'teto pub':>10}{'teto rec':>10}{'sat pub':>9}{'sat rec':>9}  veredito")
for clf in ("sgd", "pvbin"):
    for estrategia in ("entropy", "random"):
        chave = f"{clf}_{estrategia}"
        tetos, sats = campanha(clf, estrategia)
        p = multi["por_braco"][chave]
        tm, sm = round(est.mean(tetos), 4), round(est.mean(sats))
        ok = abs(tm - p["teto_f1_mean"]) < 5e-4 and abs(sm - p["sat_mean"]) <= 1
        print(f"{chave:<16}{len(tetos):>3}{p['teto_f1_mean']:>10.4f}{tm:>10.4f}"
              f"{p['sat_mean']:>9}{sm:>9}  {'OK' if ok else 'DIVERGE'}")

w = multi["wilcoxon_entropia_vs_aleatorio"]
print(f"\nWilcoxon pareado (teto entropia vs aleatório), {w['sgd']['n']} sementes:")
for clf in ("sgd", "pvbin"):
    print(f"  {clf:<6} vitórias {w[clf]['vitorias']}  W={w[clf]['W']}  p={w[clf]['p_min_bicaudal']}")
'''))

celulas.append(code(r'''
# 5) CONFRONTO COM O TEXTO DA TESE.
#
#    A comparação é NUMÉRICA, com tolerância declarada — não textual. Comparar
#    string aqui produziria falso positivo em toda linha, porque a tese escreve
#    "9,1k" com vírgula decimal e o artefato traz 9062. O que a auditoria
#    precisa saber é se o número da tese é o mesmo arredondado, e a tolerância
#    é justamente a casa em que ela arredondou.
def br(x, casas=1):
    return f"{x:.{casas}f}".replace(".", ",")

mb, vies = multi["por_braco"], multi["vies_int_ext_sgd_entropy_L10k"]

#             rótulo,                        texto na tese,      valor da tese, valor do artefato,       tolerância
afirmacoes = [
    ("população reservada",                  "≈140 mil",              140_000, len(populacao),                5_000),
    ("saturação SGD entropia",               "9,1k ± 0,6k",             9_100, mb["sgd_entropy"]["sat_mean"],     50),
    ("desvio da saturação SGD entropia",     "± 0,6k",                    600, mb["sgd_entropy"]["sat_sd"],       50),
    ("mínimo da faixa SGD entropia",         "8,5k",                    8_500, mb["sgd_entropy"]["sat_min"],       1),
    ("máximo da faixa SGD entropia",         "10k",                    10_000, mb["sgd_entropy"]["sat_max"],       1),
    ("saturação SGD aleatório",              "15,5k ± 1,1k",           15_500, mb["sgd_random"]["sat_mean"],      50),
    ("saturação PVBin entropia",             "19,3k ± 0,7k",           19_300, mb["pvbin_entropy"]["sat_mean"],   50),
    ("saturação PVBin aleatório",            "40,6k ± 1,4k",           40_600, mb["pvbin_random"]["sat_mean"],    50),
    ("p de Wilcoxon (SGD)",                  "0,0078",                 0.0078, w["sgd"]["p_min_bicaudal"],     1e-6),
    ("viés de acurácia em |L|=10k",          "-17,1 p.p.",              -17.1, vies["acuracia"]["mean_pp"],     0.05),
    ("desvio do viés",                       "± 1,0 p.p.",                1.0, vies["acuracia"]["sd_pp"],       0.05),
    ("saturação da Tabela e6 (semente 42)",  "8.000",                   8_000, publicado["sgd_entropy"]["saturacao_95pct_teto"], 1),
]

print(f"{'afirmação do Cap. 5':<38}{'na tese':>14}{'no artefato':>14}  veredito")
achados = []
for nome, texto, na_tese, no_artefato, tol in afirmacoes:
    bate = abs(no_artefato - na_tese) <= tol
    if not bate:
        achados.append((nome, texto, no_artefato))
    mostra = f"{no_artefato:,}".replace(",", ".") if isinstance(no_artefato, int) else br(no_artefato, 4).rstrip("0").rstrip(",")
    print(f"{nome:<38}{texto:>14}{mostra:>14}  {'OK' if bate else 'DIVERGE'}")

# as contagens são exatas; comparam-se à parte
for nome, na_tese, no_artefato in [("vitórias de teto SGD", "8/8", w["sgd"]["vitorias"]),
                                   ("vitórias de teto PVBin", "6/8", w["pvbin"]["vitorias"])]:
    bate = na_tese == no_artefato
    if not bate:
        achados.append((nome, na_tese, no_artefato))
    print(f"{nome:<38}{na_tese:>14}{no_artefato:>14}  {'OK' if bate else 'DIVERGE'}")

print()
if not achados:
    print("Nenhuma divergência: o Cap. 5 está sustentado pelos artefatos do E6.")
else:
    print(f"{len(achados)} DIVERGÊNCIA(S) — não corrija aqui; poste bloqueio ao principal:")
    for nome, texto, valor in achados:
        print(f"  · {nome}: a tese diz {texto}, o artefato diz {valor}")
'''))

celulas.append(code(r'''
# 6) Gráficos. Paleta categórica validada (ΔE CVD adjacente >= 8); a identidade
#    do seletor fixa a cor, então filtrar braços NÃO repinta os que ficam.
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

COR = {"entropy": "#2a78d6", "random": "#eb6834", "drisl": "#1baf7a",
       "drisl-c": "#eda100", "drisl-cs": "#e87ba4"}
NOME = {"entropy": "entropia", "random": "aleatório", "drisl": "DRI-SL",
        "drisl-c": "DRI-SL-C", "drisl-cs": "DRI-SL-C (ablação)"}
TINTA, TINTA2, GRADE = "#0b0b0b", "#52514e", "#d8d7d2"

plt.rcParams.update({"figure.dpi": 120, "font.size": 9,
                     "axes.edgecolor": GRADE, "axes.labelcolor": TINTA2,
                     "xtick.color": TINTA2, "ytick.color": TINTA2,
                     "axes.spines.top": False, "axes.spines.right": False})

fig, eixos = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for eixo, clf in zip(eixos, ("pvbin", "sgd")):
    for estrategia, cor in COR.items():
        caminho = RES / f"popcurve_{clf}_{estrategia}.jsonl"
        if not caminho.exists():
            continue
        pontos = curva(f"{clf}_{estrategia}")
        xs = [p["n_labels"] for p in pontos]
        ys = [p["f1_ext"] for p in pontos]
        eixo.plot(xs, ys, color=cor, linewidth=2, label=NOME[estrategia])
    # Sem rótulo direto na ponta: as cinco curvas convergem quase no mesmo
    # ponto em 50k e os rótulos viram um amontoado ilegível — defeito que só
    # apareceu ao olhar a figura renderizada. O alívio exigido pelo aviso de
    # contraste da paleta é atendido pela outra via prevista: este notebook é
    # feito de tabelas, e as células 3 a 5 trazem cada número em texto.
    eixo.set_title(f"{clf.upper()} — Macro F1 na população reservada",
                   color=TINTA, fontsize=10, loc="left")
    eixo.set_xlabel("rótulos gastos |L|")
    eixo.grid(True, color=GRADE, linewidth=0.6, alpha=0.7)
    eixo.set_axisbelow(True)
    eixo.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    eixo.set_xlim(0, 51_000)
    eixo.legend(frameon=False, fontsize=8, loc="lower right")
eixos[0].set_ylabel("Macro F1 (externo)")
eixos[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))
fig.suptitle("E6 · curvas de aprendizado por seletor", x=0.005, ha="left",
             color=TINTA, fontsize=12)
fig.tight_layout()
plt.show()
'''))

celulas.append(code(r'''
# 7) O instrumento novo do E6: o quanto a autoavaliação engana.
#    Duas linhas, MESMA métrica e MESMA escala — nunca dois eixos y.
pontos = curva("sgd_entropy")
xs = [p["n_labels"] for p in pontos]
interno = [p["acc_int"] for p in pontos]
externo = [p["acc_ext"] for p in pontos]

fig, eixo = plt.subplots(figsize=(7.2, 4.2))
eixo.fill_between(xs, interno, externo, color="#e34948", alpha=0.12, linewidth=0)
eixo.plot(xs, externo, color="#2a78d6", linewidth=2, label="população reservada (real)")
eixo.plot(xs, interno, color="#e34948", linewidth=2, label="teste interno (o que o praticante vê)")
eixo.annotate("população reservada", (xs[-1], externo[-1]), color=TINTA2, fontsize=7.5,
              xytext=(-4, 8), textcoords="offset points", ha="right")
eixo.annotate("teste interno", (xs[-1], interno[-1]), color=TINTA2, fontsize=7.5,
              xytext=(-4, -12), textcoords="offset points", ha="right")

i10 = min(range(len(xs)), key=lambda i: abs(xs[i] - 10_000))
lacuna = (interno[i10] - externo[i10]) * 100
eixo.annotate(f"em |L|=10k a lacuna é {lacuna:+.1f} p.p.".replace(".", ","),
              (xs[i10], (interno[i10] + externo[i10]) / 2), color=TINTA, fontsize=8.5,
              xytext=(28, -34), textcoords="offset points",
              arrowprops=dict(arrowstyle="-", color=TINTA2, linewidth=0.9))
eixo.set_title("E6 · SGD + entropia: o viés da autoavaliação em acurácia",
               color=TINTA, fontsize=11, loc="left")
eixo.set_xlabel("rótulos gastos |L|"); eixo.set_ylabel("acurácia")
eixo.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))
eixo.grid(True, color=GRADE, linewidth=0.6, alpha=0.7); eixo.set_axisbelow(True)
eixo.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
eixo.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout(); plt.show()

print(f"lacuna recalculada em |L|={xs[i10]}: {lacuna:+.1f} p.p.".replace(".", ","))
print(f"publicado na campanha de 8 sementes: "
      f"{multi['vies_int_ext_sgd_entropy_L10k']['acuracia']['mean_pp']} "
      f"± {multi['vies_int_ext_sgd_entropy_L10k']['acuracia']['sd_pp']} p.p.")
'''))

celulas.append(code(r'''
# 8) Saturação nas 8 sementes: é a comparação que sustenta a inferência.
fig, eixo = plt.subplots(figsize=(7.2, 3.2))
rotulos, posicao = [], 0
for clf in ("sgd", "pvbin"):
    for estrategia in ("entropy", "random"):
        p = multi["por_braco"][f"{clf}_{estrategia}"]
        eixo.errorbar(p["sat_mean"], posicao, xerr=p["sat_sd"], fmt="o",
                      color=COR[estrategia], markersize=8, capsize=4,
                      linewidth=2, markeredgecolor="#fcfcfb", markeredgewidth=1.5)
        eixo.annotate(f"{p['sat_mean']/1000:.1f}k ± {p['sat_sd']/1000:.1f}k".replace(".", ","),
                      (p["sat_mean"], posicao), color=TINTA2, fontsize=8,
                      xytext=(12, 0), textcoords="offset points", va="center")
        rotulos.append(f"{clf.upper()} · {NOME[estrategia]}")
        posicao += 1
eixo.set_yticks(range(len(rotulos))); eixo.set_yticklabels(rotulos, color=TINTA)
eixo.invert_yaxis()
eixo.set_xlabel("rótulos até saturar (95% do teto da própria curva)")
eixo.set_title("E6 · saturação, média ± desvio em 8 sementes",
               color=TINTA, fontsize=11, loc="left")
eixo.grid(True, axis="x", color=GRADE, linewidth=0.6, alpha=0.7)
eixo.set_axisbelow(True); eixo.set_xlim(0, 56_000)  # folga p/ o rótulo do último ponto
eixo.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
fig.tight_layout(); plt.show()
'''))

celulas.append(md(r"""
## Rodapé — o que sai daqui e como repetir

**Este notebook não grava artefato**: ele audita os que já existem. Os arquivos
conferidos estão todos em `experiments/e6population/results/`.

Para regerar os artefatos do zero (CPU, 2–6 h por braço — **não** precisa de GPU):

```bash
python experiments/e6population/run_population_curve.py \
    --classifier both --strategy entropy      # e depois random, drisl, drisl-c
```

A campanha multi-semente varia a semente mantendo o *pool* fixo; o log está em
`results/campanha_sementes.log`.

**Para rodar no Kaggle**: nenhuma configuração especial. Este notebook é CPU —
não gaste cota de GPU com ele. Basta *Internet ligada* para o clone da célula 1.

---

### Se alguma célula acusou divergência

Não corrija o número, nem aqui nem na tese. Poste bloqueio ao agente
`principal` com: a afirmação, o valor publicado, o valor recalculado e o
arquivo de onde ele saiu. Quem decide o que entra no texto é o `principal` com
o autor.
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
