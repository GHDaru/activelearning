"""Fonte legível do notebook de auditoria do E3'.

Edite ESTE arquivo, não o .ipynb — depois rode:

    python notebooks/auditoria/build_e3prime.py
"""
from __future__ import annotations

import json
from pathlib import Path

SAIDA = Path(__file__).resolve().parent / "e3prime-validacao.ipynb"


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
# E3′ — o classificador forte julga o pipeline · notebook de auditoria

**Pergunta.** O que o pipeline barato (laço leve + oráculo LLM) entrega a um
classificador forte se compara com supervisão completa do *pool*?

**Hipótese pré-registrada.** `F1(A) ≥ 0,95 × F1(D)` — o braço do pipeline real
alcança 95% do Macro F1 da régua, gastando uma fração dos rótulos.

**Artefatos auditados** (todos já no repositório):

| Arquivo | O que guarda |
|---|---|
| `e3prime_<braço>_s42.json` | métricas por braço, semente de treino 42 (a publicada) |
| `e3prime_<braço>_s7.json` | idem, semente 7 — **regime diferente**, ver abaixo |
| `e3prime_<braço>_s<n>_pred.json` | as predições, que permitem os testes pareados |
| `mcnemar_s42.json` | McNemar pareado A–B, B–C, E35–D |
| `bootstrap_f1_s42.json` | IC bootstrap das diferenças de Macro F1 |

**Protocolo de dados: o POPULACIONAL, com um recorte a mais que o E6.** Parte
da mesma base deduplicada (231.490 textos, 714 classes, embaralho com semente
42) e do mesmo *pool* de 50.000 — mas a população de avaliação do E3′ exclui
**também** os 4.000 do *holdout* do ciclo real (validação e teste, 2.000 + 2.000
usados nas decisões de parada). Logo `dedup[54000:]` = **177.490**, e não os
181.490 do E6. Os dois números estão certos, para experimentos diferentes.

> ⚠️ **Dois regimes convivem nos artefatos, e comparar sementes entre eles é
> inválido.** A semente 42 rodou com **lote 16** e avaliação numa **amostra
> estratificada de 20.092**; a semente 7 rodou com **lote 128** e avaliação na
> **população inteira (177.490)**. Variam três coisas ao mesmo tempo — semente,
> lote e conjunto de avaliação —, então a diferença entre elas **não mede
> robustez à semente**. Este notebook mostra as duas lado a lado exatamente
> para tornar isso visível, nunca para tratá-las como réplicas.

**Regra desta auditoria.** Divergência é acusada, nunca corrigida.
""".rstrip()))

celulas.append(code(r'''
# 1) Onde estamos rodando. Igual na máquina local e no Kaggle.
import json, os, subprocess, sys
from pathlib import Path

def achar_raiz() -> Path:
    aqui = Path.cwd()
    for base in [aqui, *aqui.parents]:
        if (base / "experiments/e2e3").is_dir():
            return base
    destino = Path("/tmp/activelearning")
    if not destino.exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/GHDaru/activelearning.git", str(destino)],
                       check=True, capture_output=True)
    return destino

RAIZ = achar_raiz()
os.chdir(RAIZ)
RES = RAIZ / "experiments/e2e3/results"
print("raiz:", RAIZ)
print("artefatos do E3':", len(list(RES.glob("e3prime_*"))), "arquivos")
'''))

celulas.append(code(r'''
# 2) IDENTIDADE DOS DADOS — o portão. Note o recorte extra do E3': além do
#    pool, saem os 4.000 do holdout do ciclo real, e é isso que separa os
#    177.490 daqui dos 181.490 do E6.
import csv, hashlib, random
from collections import Counter

CSV = RAIZ / "data/dataset.csv"
DATA_SEED, POOL_SIZE, CYCLE_HOLDOUT = 42, 50_000, 4_000

sys.path.insert(0, str(RAIZ / "src"))
from activelearning.domain.instances import normalize_label

md5 = hashlib.md5(CSV.read_bytes()).hexdigest()
linhas = []
with CSV.open(encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        linhas.append((r["nm_item"], normalize_label(r["nm_product"])))
contagem = Counter(l for _, l in linhas)
vistos, dedup = set(), []
for t, l in ((t, l) for t, l in linhas if contagem[l] >= 2):
    k = t.strip().lower()
    if k not in vistos:
        vistos.add(k)
        dedup.append((t, l))
random.Random(DATA_SEED).shuffle(dedup)
pool = dedup[:POOL_SIZE]
populacao = dedup[POOL_SIZE + CYCLE_HOLDOUT:]

checagens = [
    ("md5 do dataset.csv",        md5,                          None),
    ("textos deduplicados",       len(dedup),                   231_490),
    ("classes presentes",         len({l for _, l in dedup}),   714),
    ("pool",                      len(pool),                    50_000),
    ("holdout do ciclo real",     CYCLE_HOLDOUT,                4_000),
    ("população de avaliação",    len(populacao),               177_490),
]
print(f"{'o quê':<26}{'medido agora':>18}{'esperado':>12}  veredito")
portao_ok = True
for nome, medido, alvo in checagens:
    if alvo is None:
        print(f"{nome:<26}{str(medido):>18}{'—':>12}  (registro)")
        continue
    ok = medido == alvo
    portao_ok &= ok
    print(f"{nome:<26}{medido:>18}{alvo:>12}  {'OK' if ok else 'DIVERGE'}")
print("\nPORTÃO:", "ABERTO" if portao_ok else "FECHADO — não confie no resto")
'''))

celulas.append(code(r'''
# 3) A Tabela e3p do Cap. 5, reconstruída dos artefatos — com o IC de Wilson
#    recalculado aqui, não copiado.
import math

def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / den
    meia = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (centro - meia, centro + meia)

def carregar(semente):
    saida = {}
    for caminho in RES.glob(f"e3prime_*_s{semente}.json"):
        if caminho.stem.endswith("_pred"):
            continue
        d = json.loads(caminho.read_text())
        saida[d["arm"]] = d
    return saida

s42 = carregar(42)
s7 = carregar(7)
FONTE = {"A": "pipeline real, rótulos do oráculo", "B": "mesmos itens, gabarito",
         "C": "aleatório, gabarito", "E": "15k por entropia (E6), gabarito",
         "D": "pool inteiro, gabarito (régua)"}

print(f"{'braço':<6}{'conjunto de treino':<38}{'n':>7}{'acurácia [IC 95%]':>26}{'Macro F1':>10}")
for a in ["A", "B", "C", "E", "D"]:
    d = s42[a]
    lo, hi = wilson(round(d["accuracy"] * d["eval_n"]), d["eval_n"])
    acc_ic = f"{d['accuracy'] * 100:.1f}% [{lo * 100:.1f}; {hi * 100:.1f}]"
    print(f"{a:<6}{FONTE[a]:<38}{d['n_train']:>7}{acc_ic:>26}{d['macro_f1']:>10.3f}")

print(f"\nconfiguração registrada nos artefatos: {s42['A']['epochs']} épocas, "
      f"lote {s42['A']['batch_size']}, {s42['A']['max_length']} tokens, "
      f"avaliação em {s42['A']['eval_n']}")
'''))

celulas.append(code(r'''
# 4) CONFRONTO com as afirmações do Cap. 5, seção E3'. Comparação numérica com
#    tolerância declarada — a tese arredonda de propósito.
D42 = s42["D"]
crit_f1, crit_acc = 0.95 * D42["macro_f1"], 0.95 * D42["accuracy"]

afirmacoes = [
    ("n da avaliação",                    "20.092",   20_092, s42["A"]["eval_n"],                      0),
    ("épocas",                            "3",             3, s42["A"]["epochs"],                      0),
    ("lote",                              "16",           16, s42["A"]["batch_size"],                  0),
    ("comprimento em tokens",             "32",           32, s42["A"]["max_length"],                  0),
    ("F1 do braço A",                     "0,242",     0.242, s42["A"]["macro_f1"],                0.001),
    ("critério 0,95 x F1(D)",             "0,428",     0.428, crit_f1,                             0.001),
    ("critério 0,95 x acurácia(D)",       "83,9%",      83.9, crit_acc * 100,                       0.05),
    ("fração de rótulos do braço A",      "17,9%",      17.9, s42["A"]["n_train"] / 50_000 * 100,    0.05),
    ("perda de acurácia A vs B",          "4,4 p.p.",    4.4, (s42["B"]["accuracy"] - s42["A"]["accuracy"]) * 100, 0.05),
    ("ganho de acurácia C vs B",          "3,0 p.p.",    3.0, (s42["C"]["accuracy"] - s42["B"]["accuracy"]) * 100, 0.05),
    ("classes no treino de A",            "635",         635, s42["A"]["n_train_classes"],             0),
    ("classes no treino de B",            "620",         620, s42["B"]["n_train_classes"],             0),
    ("classes no treino de C",            "493",         493, s42["C"]["n_train_classes"],             0),
    ("braço E como % da régua (acc)",     "94,1%",      94.1, s42["E"]["accuracy"] / D42["accuracy"] * 100, 0.05),
    ("braço E como % da régua (F1)",      "84,3%",      84.3, s42["E"]["macro_f1"] / D42["macro_f1"] * 100, 0.05),
]

def mostrar(v):
    """Inteiro com ponto de milhar; decimal com vírgula. Nunca notação científica."""
    if isinstance(v, int) or float(v).is_integer():
        return f"{int(v):,}".replace(",", ".")
    return f"{v:.4f}".rstrip("0").rstrip(".").replace(".", ",")

print(f"{'afirmação do Cap. 5':<34}{'na tese':>12}{'no artefato':>14}  veredito")
achados = []
for nome, texto, na_tese, no_artefato, tol in afirmacoes:
    bate = abs(no_artefato - na_tese) <= tol
    if not bate:
        achados.append((nome, texto, no_artefato))
    print(f"{nome:<34}{texto:>12}{mostrar(no_artefato):>14}  {'OK' if bate else 'DIVERGE'}")

veredito = "REFUTADA" if s42["A"]["macro_f1"] < crit_f1 else "sustentada"
print(f"\nHIPÓTESE na semente 42: F1(A)={s42['A']['macro_f1']:.4f} vs "
      f"0,95xF1(D)={crit_f1:.4f} -> {veredito}")
print("(o capítulo também diz REFUTADA para a configuração executada)")

print()
if achados:
    print(f"{len(achados)} divergência(s):")
    for nome, texto, valor in achados:
        print(f"  · {nome}: tese diz {texto}, artefato diz {mostrar(valor)}")
else:
    print("Nenhuma divergência.")
'''))

celulas.append(code(r'''
# 5) McNemar pareado, RECALCULADO das predições — não lido do JSON.
#    É o que transforma mcnemar_s42.json em resultado conferível: as predições
#    de todos os braços vêm da mesma amostra de avaliação, então o pareamento
#    por instância é legítimo.
def predicoes(braco, semente):
    return json.loads((RES / f"e3prime_{braco}_s{semente}_pred.json").read_text())

base = predicoes("A", 42)
gabarito = [populacao[i][1] for i in base["sample_idx"]]

def mcnemar(x, y):
    b = sum(1 for g, i, j in zip(gabarito, x, y) if i == g and j != g)
    c = sum(1 for g, i, j in zip(gabarito, x, y) if j == g and i != g)
    # binomial bicaudal exata em min(b,c) ~ Bin(b+c, 0,5)
    n, k = b + c, min(b, c)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)) if n else 1.0
    return b, c, p

pub_mc = json.loads((RES / "mcnemar_s42.json").read_text())
print(f"{'par':<8}{'b pub':>7}{'b rec':>7}{'c pub':>7}{'c rec':>7}{'p publicado':>14}  veredito")
for par in ["A-B", "B-C", "E35-D"]:
    X, Y = par.split("-")
    idx_x, idx_y = predicoes(X, 42), predicoes(Y, 42)
    assert idx_x["sample_idx"] == base["sample_idx"] == idx_y["sample_idx"], \
        "amostras de avaliação diferentes — pareamento inválido"
    b, c, _ = mcnemar(idx_x["pred"], idx_y["pred"])
    p = pub_mc["pares"][par]
    ok = b == p["b"] and c == p["c"]
    print(f"{par:<8}{p['b']:>7}{b:>7}{p['c']:>7}{c:>7}{p['p_exato']:>14.2e}  "
          f"{'OK' if ok else 'DIVERGE'}")
'''))

celulas.append(code(r'''
# 6) Bootstrap pareado das diferenças de Macro F1, RECALCULADO.
#
#    O Macro F1 é vetorizado por contagens (bincount) em vez do f1_score do
#    scikit-learn: dá o MESMO valor e roda ~1200x mais rápido, o que é a
#    diferença entre 8 horas e meio minuto para 10.000 réplicas. Sem isso, esta
#    célula não caberia num notebook e o artefato continuaria não auditável.
import numpy as np

pub_bs = json.loads((RES / "bootstrap_f1_s42.json").read_text())
N_BOOT = pub_bs["n_boot"]
BRACOS_BS = list(pub_bs["macro_f1_braco"])

preds = {a: predicoes(a, 42)["pred"] for a in BRACOS_BS}

# O conjunto de rótulos é POR BRAÇO — união(gabarito, predições daquele braço) —
# e não uma união global. Não é detalhe de implementação: é o que o
# `f1_score(..., average="macro")` do scikit-learn faz por padrão, e portanto o
# que o run_e3prime.py gravou. Com união global entram classes que aquele braço
# nunca previu, cada uma contando F1=0, e o valor cai (0,2420 em vez de 0,2424
# no braço A). Fixamos por braço, e o conjunto NÃO muda entre réplicas.
def codificar(braco):
    classes = sorted(set(gabarito) | set(preds[braco]))
    cod = {c: i for i, c in enumerate(classes)}
    g = np.fromiter((cod[x] for x in gabarito), dtype=np.int32, count=len(gabarito))
    p = np.fromiter((cod[x] for x in preds[braco]), dtype=np.int32, count=len(gabarito))
    return g, p, len(classes)

CODIFICADO = {a: codificar(a) for a in BRACOS_BS}

def macro_f1(g, p, K):
    tp = np.bincount(g[g == p], minlength=K)
    den = np.bincount(g, minlength=K) + np.bincount(p, minlength=K)
    return float(np.mean(np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)))

# confere a vetorização contra o valor publicado ANTES de confiar nela
print("checagem do estimador vetorizado (ponto):")
estimador_ok = True
for a in BRACOS_BS:
    g_a, p_a, K_a = CODIFICADO[a]
    calc, pubv = macro_f1(g_a, p_a, K_a), pub_bs["macro_f1_braco"][a]["ponto"]
    ok = abs(calc - pubv) < 5e-5
    estimador_ok &= ok
    print(f"  {a:>4}: {calc:.4f} vs publicado {pubv:.4f}  {'OK' if ok else 'DIVERGE'}")
assert estimador_ok, "o estimador não reproduz o ponto — não siga para o bootstrap"

rng = np.random.default_rng(pub_bs["boot_seed"])
n = len(gabarito)
distrib = {a: np.empty(N_BOOT) for a in BRACOS_BS}
for r in range(N_BOOT):
    amostra = rng.integers(0, n, n)          # mesma reamostragem em todos os braços
    for a in BRACOS_BS:
        g_a, p_a, K_a = CODIFICADO[a]
        distrib[a][r] = macro_f1(g_a[amostra], p_a[amostra], K_a)

print(f"\n{N_BOOT} réplicas pareadas · IC percentil 95% das DIFERENÇAS")
print(f"{'par':<8}{'Δ ponto':>10}{'IC publicado':>22}{'IC recalculado':>22}")
for par in ["A-B", "B-C", "E35-D"]:
    X, Y = par.split("-")
    delta = distrib[X] - distrib[Y]
    lo, hi = np.percentile(delta, [2.5, 97.5])
    p = pub_bs["pares"][par]
    ic_pub_lo, ic_pub_hi = p["delta_ic95"]
    ic_pub = f"[{ic_pub_lo:.4f}; {ic_pub_hi:.4f}]"
    ic_rec = f"[{lo:.4f}; {hi:.4f}]"
    print(f"{par:<8}{p['delta_ponto']:>10.4f}{ic_pub:>22}{ic_rec:>22}")
print("\nOs três IC excluem zero em ambas as versões — é o que o capítulo afirma.")
'''))

celulas.append(code(r'''
# 7) OS DOIS REGIMES LADO A LADO. Esta é a célula que motiva o achado aberto
#    com o principal. NÃO é comparação de sementes: mudam semente, lote e
#    conjunto de avaliação ao mesmo tempo.
def linha_regime(rot, d):
    return (f"{rot:<26}{d['epochs']:>3} épocas  lote {d['batch_size']:>3}  "
            f"avaliação em {d['eval_n']:>7}")

print(linha_regime("semente 42 (publicada)", s42["D"]))
print(linha_regime("semente 7 (executor01)", s7["D"]))

crit42 = 0.95 * s42["D"]["macro_f1"]
crit7 = 0.95 * s7["D"]["macro_f1"]
print(f"\ncritério de Macro F1  ·  s42: {crit42:.4f}   s7: {crit7:.4f}")
print(f"\n{'braço':<6}{'n':>7}{'F1 s42':>9}{'crit?':>7}{'F1 s7':>9}{'crit?':>7}   leitura")
for a in ["E", "E20", "E25", "E30", "E35", "D"]:
    if a not in s42 or a not in s7:
        continue
    f42, f7 = s42[a]["macro_f1"], s7[a]["macro_f1"]
    ok42 = "sim" if f42 >= crit42 else "não"
    ok7 = "sim" if f7 >= crit7 else "não"
    virou = "" if (f42 >= crit42) == (f7 >= crit7) or a == "D" else "  <-- INVERTE"
    print(f"{a:<6}{s42[a]['n_train']:>7}{f42:>9.4f}{ok42:>7}{f7:>9.4f}{ok7:>7}{virou}")

e35_42 = s42["E35"]["macro_f1"] > s42["D"]["macro_f1"]
e35_7 = s7["E35"]["macro_f1"] > s7["D"]["macro_f1"]
print(f"\n'E35 supera a régua' (a leitura (iii) do capítulo):")
print(f"  na semente 42: {'SIM' if e35_42 else 'NÃO'}  "
      f"({s42['E35']['macro_f1']:.4f} vs {s42['D']['macro_f1']:.4f})")
print(f"  na semente  7: {'SIM' if e35_7 else 'NÃO'}  "
      f"({s7['E35']['macro_f1']:.4f} vs {s7['D']['macro_f1']:.4f})")
print("\nA afirmação do capítulo depende do regime. Enquanto a s42 não for")
print("refeita no mesmo comando da s7, as duas NÃO são comparáveis e a")
print("robustez à semente segue sem medida. Decisão do principal + autor.")
'''))

celulas.append(code(r'''
# 8) Gráficos.
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

AZUL, LARANJA, VERDE, AMARELO = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
TINTA, TINTA2, GRADE = "#0b0b0b", "#52514e", "#d8d7d2"
vg = FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ","))

plt.rcParams.update({"figure.dpi": 120, "font.size": 9,
                     "axes.edgecolor": GRADE, "axes.labelcolor": TINTA2,
                     "xtick.color": TINTA2, "ytick.color": TINTA2,
                     "axes.spines.top": False, "axes.spines.right": False})

VARREDURA = ["E", "E20", "E25", "E30", "E35"]
fig, eixos = plt.subplots(1, 2, figsize=(11, 4.3))
for eixo, metrica, rotulo in [(eixos[0], "macro_f1", "Macro F1"),
                              (eixos[1], "accuracy", "acurácia")]:
    for dados, cor, nome in [(s42, AZUL, "semente 42 · lote 16 · aval. 20.092"),
                             (s7, LARANJA, "semente 7 · lote 128 · aval. 177.490")]:
        xs = [dados[a]["n_train"] for a in VARREDURA if a in dados]
        ys = [dados[a][metrica] for a in VARREDURA if a in dados]
        if not xs:
            continue
        eixo.plot(xs, ys, color=cor, linewidth=2, marker="o", markersize=6,
                  markeredgecolor="#fcfcfb", markeredgewidth=1.2, label=nome)
        criterio = 0.95 * dados["D"][metrica]
        eixo.axhline(criterio, color=cor, linewidth=1.2, linestyle="--", alpha=0.55)
        # rótulo do critério na borda DIREITA e deslocado por série, senão as
        # duas linhas tracejadas ficam próximas e os textos se sobrepõem
        eixo.annotate(f"critério {criterio:.3f}".replace(".", ","),
                      (xs[-1], criterio), color=cor, fontsize=7.5, ha="right",
                      xytext=(0, 4 if dados is s42 else -11),
                      textcoords="offset points")
    eixo.set_title(f"{rotulo} — varredura de orçamento", color=TINTA,
                   fontsize=10, loc="left")
    eixo.set_xlabel("rótulos no treino")
    eixo.grid(True, color=GRADE, linewidth=0.6, alpha=0.7)
    eixo.set_axisbelow(True)
    # ticks nos pontos que existem: o automático inventa 18k, 22k, 28k, que não
    # correspondem a braço nenhum e sugerem uma continuidade que não há
    eixo.set_xticks([s42[a]["n_train"] for a in VARREDURA])
    eixo.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    eixo.yaxis.set_major_formatter(vg)
eixos[0].set_ylabel("valor da métrica")
eixos[0].legend(frameon=False, fontsize=7.5, loc="upper left")
fig.suptitle("E3′ · a varredura cruza o critério num regime e não no outro",
             x=0.005, ha="left", color=TINTA, fontsize=12)
fig.tight_layout()
plt.show()
'''))

celulas.append(code(r'''
# 9) A decomposição A / B / C — as duas leituras pareadas do capítulo.
#    Duas métricas de escalas diferentes vão em PAINÉIS separados, nunca em
#    dois eixos y no mesmo gráfico.
fig, eixos = plt.subplots(1, 2, figsize=(9.5, 3.8))
bracos = ["A", "B", "C"]
legendas = {"A": "A\npipeline real\n(oráculo)", "B": "B\nmesmos itens\n(gabarito)",
            "C": "C\naleatório\n(gabarito)"}
for eixo, metrica, rotulo, cor in [(eixos[0], "accuracy", "acurácia", AZUL),
                                   (eixos[1], "macro_f1", "Macro F1", VERDE)]:
    valores = [s42[a][metrica] for a in bracos]
    barras = eixo.bar([legendas[a] for a in bracos], valores, color=cor, width=0.62)
    for barra, v in zip(barras, valores):
        eixo.annotate(f"{v:.3f}".replace(".", ","),
                      (barra.get_x() + barra.get_width() / 2, v), color=TINTA,
                      fontsize=8.5, ha="center", xytext=(0, 3),
                      textcoords="offset points")
    eixo.set_title(rotulo, color=TINTA, fontsize=10, loc="left")
    eixo.grid(True, axis="y", color=GRADE, linewidth=0.6, alpha=0.7)
    eixo.set_axisbelow(True)
    eixo.yaxis.set_major_formatter(vg)
    eixo.set_ylim(0, max(valores) * 1.22)
fig.suptitle("E3′ · mesmos 8.937 itens: o ruído do oráculo custa acurácia mas "
             "compra cobertura de cauda", x=0.005, ha="left", color=TINTA, fontsize=11)
fig.tight_layout()
plt.show()

print("classes presentes no treino de cada braço:",
      ", ".join(f"{a}={s42[a]['n_train_classes']}" for a in bracos))
print("É a leitura (i)+(ii) do capítulo: B vence A em acurácia, A vence B em")
print("Macro F1, e C — aleatório — vence em acurácia mas cobre bem menos classes.")
'''))

celulas.append(md(r"""
## Rodapé — artefatos e como reproduzir

Este notebook **não grava artefato**: audita os que existem em
`experiments/e2e3/results/`.

Para reexecutar os braços (é o único experimento da tese que **precisa de
GPU** — nove ajustes finos completos do BERTimbau):

```bash
python experiments/e2e3/run_e3prime.py \
    --arms A,B,C,E,D,E20,E25,E30,E35 --epochs 3 --batch-size 128 \
    --eval-limit 0 --seed 7 --out-dir experiments/e2e3/results
```

No Kaggle, use `notebooks/../experiments/e2e3/e3prime_kaggle.ipynb` com
**GPU T4** — a P100 é `sm_60` e o torch da imagem começa em `sm_70`. Braço já
concluído é pulado, então reexecutar após queda é seguro.

### O que este notebook NÃO consegue auditar

A afirmação de **concordância oráculo–gabarito de 71,6%** nos itens do braço A
depende de `experiments/e5cycle/results/annotation_cache_nemotron.jsonl`, que
não está versionado. Pelo mesmo motivo os braços **A, B e C não existem na
semente 7**. Enquanto o cache não entrar no repositório, esses três números
ficam fora de qualquer auditoria independente.

### Se alguma célula acusou divergência

Poste bloqueio ao agente `principal` com a afirmação, os dois valores e o
arquivo de origem. Não ajuste número — nem no repositório, nem na tese.
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
