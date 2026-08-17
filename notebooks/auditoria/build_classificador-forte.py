"""Fonte legível do notebook de auditoria do E3'.

Edite ESTE arquivo, não o .ipynb — depois rode:

    python notebooks/auditoria/build_classificador-forte.py
"""
from __future__ import annotations

import json
from pathlib import Path

SAIDA = Path(__file__).resolve().parent / "classificador-forte.ipynb"


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

> ⚠️ **Dois regimes convivem. Não os misture.**
>
> | Regime | Lote | Avaliação | Onde mora | Papel |
> |---|---|---|---|---|
> | **pareado** | 16 | 20.092 | `results/legacy_s42_bs16_eval20k/` | o **publicado** no Cap. 5 |
> | **canônico** | 128 | 177.490 | `results/e3prime_*_s{42,7,123}.json` | três sementes |
>
> Comparar pareado com canônico mede **regime**, não semente. As três sementes
> canônicas são homogêneas entre si, e é só entre elas que a robustez à semente
> — pendência nº 1 do parecer da banca — pode ser medida. A célula 7 faz isso.
>
> ⚠️ **Cuidado de arquivo**: os dois regimes gravam o MESMO nome
> (`e3prime_D_s42.json`). Quando a semente 42 foi reexecutada em canônico, o
> nome simples mudou de significado e o publicado foi para `legacy_`. Por isso
> este notebook lê o pareado do diretório `legacy_`, nunca da raiz. Ver
> `NOMES.md`.

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

# LAYOUT DOS ARTEFATOS (convenção adotada pelo executor02 e mergeada na main):
#   results/e3prime_*_s<n>.json            -> regime CANÔNICO (lote 128, pop. inteira)
#   results/legacy_s42_bs16_eval20k/...    -> regime PAREADO, o publicado no Cap. 5
# O nome simples mudou de significado quando a s42 foi reexecutada em canônico.
# Por isso o pareado é lido do diretório legacy_, nunca da raiz.
PAREADO = RES / "legacy_s42_bs16_eval20k"

def carregar_pareado():
    saida = {}
    for caminho in PAREADO.glob("e3prime_*_s42.json"):
        if caminho.stem.endswith("_pred"):
            continue
        d = json.loads(caminho.read_text())
        saida[d["arm"]] = d
    return saida

s42p = carregar_pareado()                       # publicado no Cap. 5 (lote 16)
CANONICAS = {n: carregar(n) for n in (42, 7, 123)}
CANONICAS = {n: d for n, d in CANONICAS.items() if d}
s42 = s42p                                      # o Cap. 5 fala deste
print(f"regime pareado (publicado): {len(s42p)} braços · "
      f"canônicas disponíveis: {sorted(CANONICAS)}")
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
def predicoes(braco, semente, pasta=None):
    pasta = pasta or RES
    return json.loads((pasta / f"e3prime_{braco}_s{semente}_pred.json").read_text())

# as predições do Cap. 5 são as do regime PAREADO, que hoje vivem em legacy_
base = predicoes("A", 42, PAREADO)
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
    idx_x, idx_y = predicoes(X, 42, PAREADO), predicoes(Y, 42, PAREADO)
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

preds = {a: predicoes(a, 42, PAREADO)["pred"] for a in BRACOS_BS}

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
# 7) ROBUSTEZ MULTI-SEMENTE — a pendência nº 1 do parecer da banca.
#
#    Só é legítimo agora: as três sementes (42, 7, 123) rodaram no MESMO regime
#    canônico, lote 128 e avaliação na população inteira. Comparar qualquer uma
#    delas com o s42 pareado mede regime, não semente.
import statistics as est

VARREDURA_NOMES = ["E", "E20", "E25", "E30", "E35"]
ORDEM = VARREDURA_NOMES + ["D"]

print("regime canônico · 3 épocas · lote 128 · avaliação em 177.490")
cab = "".join(f"{'s'+str(n):>10}" for n in sorted(CANONICAS))
print(f"\n{'braço':<6}{'n':>7}{cab}{'média':>10}{'desvio':>9}")
media_f1 = {}
for a in ORDEM:
    vals = [CANONICAS[n][a]["macro_f1"] for n in sorted(CANONICAS) if a in CANONICAS[n]]
    if len(vals) < len(CANONICAS):
        continue
    m, sd = est.mean(vals), est.stdev(vals)
    media_f1[a] = (m, sd)
    celulas_v = "".join(f"{v:>10.4f}" for v in vals)
    n_tr = CANONICAS[sorted(CANONICAS)[0]][a]["n_train"]
    print(f"{a:<6}{n_tr:>7}{celulas_v}{m:>10.4f}{sd:>9.4f}")

# o critério é relativo à régua DE CADA SEMENTE, então varia com ela
print(f"\n{'braço':<6}  cruza 0,95xF1(D) em quantas sementes?")
consenso = {}
for a in VARREDURA_NOMES:
    quantas = [n for n in sorted(CANONICAS)
               if a in CANONICAS[n] and CANONICAS[n][a]["macro_f1"] >= 0.95 * CANONICAS[n]["D"]["macro_f1"]]
    consenso[a] = quantas
    print(f"{a:<6}  {len(quantas)} de {len(CANONICAS)}"
          + (f"  (sementes {', '.join(str(x) for x in quantas)})" if quantas else "  (nenhuma)"))

supera = [n for n in sorted(CANONICAS)
          if CANONICAS[n]["E35"]["macro_f1"] > CANONICAS[n]["D"]["macro_f1"]]
print(f"\n'E35 supera a régua' (leitura (iii) do Cap. 5):")
print(f"  no regime PAREADO, publicado: SIM "
      f"({s42p['E35']['macro_f1']:.4f} vs {s42p['D']['macro_f1']:.4f})")
print(f"  no regime canônico: {len(supera)} de {len(CANONICAS)} sementes"
      + (f" ({supera})" if supera else " — em NENHUMA"))
print("\nLeitura: a afirmação (iii) é um efeito do regime de lote 16, não um")
print("achado robusto. E o piso de orçamento não é estável entre sementes —")
print("é a variabilidade que a banca mandou medir, agora medida.")
'''))

celulas.append(code(r'''
# 8) Gráficos: o publicado contra a faixa das três sementes canônicas.
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

AZUL, LARANJA, VERDE = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA2, GRADE = "#0b0b0b", "#52514e", "#d8d7d2"
vg = FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ","))

plt.rcParams.update({"figure.dpi": 120, "font.size": 9,
                     "axes.edgecolor": GRADE, "axes.labelcolor": TINTA2,
                     "xtick.color": TINTA2, "ytick.color": TINTA2,
                     "axes.spines.top": False, "axes.spines.right": False})

fig, eixos = plt.subplots(1, 2, figsize=(11, 4.3))
for eixo, metrica, rotulo in [(eixos[0], "macro_f1", "Macro F1"),
                              (eixos[1], "accuracy", "acurácia")]:
    xs = [s42p[a]["n_train"] for a in VARREDURA_NOMES]

    # publicado (regime pareado, lote 16)
    eixo.plot(xs, [s42p[a][metrica] for a in VARREDURA_NOMES], color=AZUL,
              linewidth=2, marker="o", markersize=6, markeredgecolor="#fcfcfb",
              markeredgewidth=1.2, label="publicado · s42 · lote 16 · aval. 20.092")
    eixo.axhline(0.95 * s42p["D"][metrica], color=AZUL, linewidth=1.2,
                 linestyle="--", alpha=0.55)

    # canônico: média das 3 sementes, com faixa de mín–máx
    vals = np.array([[CANONICAS[n][a][metrica] for a in VARREDURA_NOMES]
                     for n in sorted(CANONICAS)])
    eixo.fill_between(xs, vals.min(axis=0), vals.max(axis=0), color=LARANJA,
                      alpha=0.16, linewidth=0)
    eixo.plot(xs, vals.mean(axis=0), color=LARANJA, linewidth=2, marker="o",
              markersize=6, markeredgecolor="#fcfcfb", markeredgewidth=1.2,
              label=f"canônico · {len(CANONICAS)} sementes · lote 128 · aval. 177.490")
    crit_canon = np.mean([0.95 * CANONICAS[n]["D"][metrica] for n in CANONICAS])
    eixo.axhline(crit_canon, color=LARANJA, linewidth=1.2, linestyle="--", alpha=0.55)

    eixo.annotate("critério", (xs[-1], 0.95 * s42p["D"][metrica]), color=AZUL,
                  fontsize=7.5, ha="right", xytext=(0, 4), textcoords="offset points")
    eixo.annotate("critério", (xs[-1], crit_canon), color=LARANJA, fontsize=7.5,
                  ha="right", xytext=(0, -11), textcoords="offset points")
    eixo.set_title(f"{rotulo} — varredura de orçamento", color=TINTA,
                   fontsize=10, loc="left")
    eixo.set_xlabel("rótulos no treino")
    eixo.grid(True, color=GRADE, linewidth=0.6, alpha=0.7)
    eixo.set_axisbelow(True)
    eixo.set_xticks(xs)
    eixo.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    eixo.yaxis.set_major_formatter(vg)
eixos[0].set_ylabel("valor da métrica")
eixos[0].legend(frameon=False, fontsize=7.5, loc="upper left")
fig.suptitle("E3′ · a faixa laranja é a variação entre as 3 sementes canônicas",
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
