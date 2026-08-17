"""Gera docs/records/rastreabilidade.json (no tesedaru) a partir dos artefatos.

Etapa 2 da tarefa 20260816-2205. Cada item liga um NÚMERO da tese ao
experimento, ao artefato, ao código que o produziu e ao status da evidência.

Status possíveis:
  rastreado        o número sai do artefato citado
  divergente       o artefato existe e diz OUTRA coisa
  sem-evidencia    não há artefato que sustente o número
  legado           a evidência existe, mas em repositório legado somente leitura

Uso:  python notebooks/auditoria/build_rastreabilidade.py
"""
from __future__ import annotations

import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
RES = RAIZ / "experiments"
TESE = RAIZ.parent / "tesedaru"
SAIDA = TESE / "docs/records/rastreabilidade.json"


def ler(rel: str):
    return json.loads((RES / rel).read_text(encoding="utf-8"))


def perto(a, b, tol):
    return a is not None and b is not None and abs(a - b) <= tol


# ---------------------------------------------------------------- artefatos
e0_tab = ler("e0/results/e0_table.json")
e0_mc = ler("e0/results/e0_mcnemar.json")
e6_multi = ler("e6population/results/analysis_multiseed.json")
e6_an = ler("e6population/results/analysis.json")


def e0_linha(nome, amostra, inval=None):
    """Linha do e0_table por modelo+amostra. `inval` desempata duplicatas."""
    achados = [r for r in e0_tab
               if nome in r["oracle_id"] and r["sample"] == amostra
               and (inval is None or perto(r["invalid_label_rate"], inval, 1e-9))]
    return achados[0] if achados else None


itens = []


def add(numero, onde, experimento, artefato, codigo, status, nota="", notebook=""):
    itens.append({"numero": numero, "onde": onde, "experimento": experimento,
                  "artefato": artefato, "codigo": codigo,
                  "notebook_kaggle": notebook, "status": status, "nota": nota})


# ------------------------------------------------- E0 · tabela tab:e0-principal
COD_E0 = "experiments/e0/run_e0.py + analyze_e0.py"
ART_E0 = "experiments/e0/results/e0_table.json"
TAB = "5-resultados-falco/texto.tex#tab:e0-principal"

# (modelo, amostra, acc na tese, F1 na tese, inválidos na tese)
tabela_tese = [
    ("deepseek-v4-pro", "rand", 82.1, 0.785, 0.1),
    ("gpt-4o@", "rand", 80.9, 0.788, 0.0),
    ("deepseek-v4-flash", "rand", 78.3, 0.750, 0.7),
    ("nemotron", "rand", 77.9, 0.752, 1.2),
    ("glm-5.2", "rand", 77.3, 0.742, 0.0),
    ("gpt-4o-mini", "rand", 61.3, 0.518, 0.0),
    ("deepseek-v4-pro", "strat", 82.6, 0.793, 0.0),
    ("gpt-4o@", "strat", 82.1, 0.797, 0.0),
    ("deepseek-v4-flash", "strat", 81.6, 0.781, 0.9),
    ("glm-5.2", "strat", 80.9, 0.782, 0.0),
    ("nemotron", "strat", 79.6, 0.770, 2.6),
    ("gpt-4o-mini", "strat", 45.7, 0.427, 0.0),
]
for modelo, amostra, acc, f1, inval in tabela_tese:
    # o gpt-4o-mini S-rand tem DUAS execuções no artefato; a tese usa a de 0% inválidos
    r = e0_linha(modelo, amostra, 0.0 if (modelo == "gpt-4o-mini" and amostra == "rand") else None)
    rot = modelo.rstrip("@")
    if r is None:
        add(f"{rot} {amostra}", TAB, "escolha-do-oraculo", ART_E0, COD_E0,
            "sem-evidencia", "linha não encontrada no artefato")
        continue
    problemas = []
    if not perto(r["accuracy"] * 100, acc, 0.05):
        problemas.append(f"acurácia {r['accuracy']*100:.1f}% no artefato")
    if not perto(r["macro_f1"], f1, 0.0005):
        problemas.append(f"Macro F1 {r['macro_f1']:.3f} no artefato")
    if not perto(r["invalid_label_rate"] * 100, inval, 0.05):
        problemas.append(f"inválidos {r['invalid_label_rate']*100:.1f}% no artefato, "
                         f"tese diz {inval:.1f}%")
    add(f"{rot} {amostra}: acc {acc}% · F1 {f1} · inválidos {inval}%", TAB,
        "escolha-do-oraculo", ART_E0, COD_E0,
        "divergente" if problemas else "rastreado", "; ".join(problemas))

# duplicata não documentada
dups = [r for r in e0_tab if "gpt-4o-mini" in r["oracle_id"] and r["sample"] == "rand"]
if len(dups) > 1:
    accs = ", ".join("{:.1f}%".format(d["accuracy"] * 100) for d in dups)
    add("gpt-4o-mini S-rand (escolha de execução)", TAB, "escolha-do-oraculo",
        ART_E0, COD_E0, "divergente",
        f"o artefato tem {len(dups)} execuções para esta célula (acc {accs}); "
        "a tese reporta uma sem dizer qual nem por quê")

# ------------------------------------------------- E0 · afirmações de RQ1
ART_MC = "experiments/e0/results/e0_mcnemar.json"


def par(amostra, a, b):
    for r in e0_mc:
        ids = r["oracle_a"] + " " + r["oracle_b"]
        if r["sample"] == amostra and a in ids and b in ids:
            return r
    return None


p_pro_flash = par("strat", "v4-pro", "v4-flash")
add("v4-pro > v4-flash na S-strat: b=43, c=16, p<0,001",
    "5-resultados-falco/texto.tex#sec:res-e0-rq1", "escolha-do-oraculo", ART_MC, COD_E0,
    "divergente",
    f"o artefato dá b={p_pro_flash['a_right_b_wrong']}, c={p_pro_flash['a_wrong_b_right']}, "
    f"p={p_pro_flash['p_value']:.3f} — NÃO significativo. Nenhum dos 43 pares do "
    "artefato tem b=43/c=16")

p_pro_4o = par("strat", "v4-pro", "gpt-4o@")
add("v4-pro empata com gpt-4o na S-strat: p=0,061",
    "5-resultados-falco/texto.tex#sec:res-e0-rq1", "escolha-do-oraculo", ART_MC, COD_E0,
    "divergente",
    f"o artefato dá p={p_pro_4o['p_value']:.3f}; nenhum par do artefato tem p≈0,061. "
    "O veredito (empate) se sustenta, o número não")

p_nem_flash = par("rand", "nemotron", "v4-flash")
add("nemotron empata com v4-flash na S-rand: p=0,76",
    "5-resultados-falco/texto.tex#sec:res-e0-rq2", "escolha-do-oraculo", ART_MC, COD_E0,
    "rastreado" if perto(p_nem_flash["p_value"], 0.76, 0.005) else "divergente",
    f"artefato p={p_nem_flash['p_value']:.3f}")

p_nem_glm = par("strat", "nemotron", "glm-5.2")
add("nemotron empata com glm-5.2 na S-strat: p=0,078",
    "5-resultados-falco/texto.tex#sec:res-e0-rq2", "escolha-do-oraculo", ART_MC, COD_E0,
    "rastreado" if perto(p_nem_glm["p_value"], 0.078, 0.001) else "divergente",
    f"artefato p={p_nem_glm['p_value']:.4f}")

# ------------------------------------------------- E0 · custo e cache
TAB_C = "5-resultados-falco/texto.tex#tab:e0-custo"
custo_tese = [("nemotron", 0.000), ("deepseek-v4-flash", 0.035), ("gpt-4o-mini", 0.051),
              ("glm-5.2", 0.249), ("deepseek-v4-pro", 0.410), ("gpt-4o@", 0.923)]
for modelo, usd in custo_tese:
    r = e0_linha(modelo, "rand", 0.0 if modelo == "gpt-4o-mini" else None)
    ok = perto(r["cost_per_1k_labels_usd"], usd, 0.0006)
    add(f"{modelo.rstrip('@')}: US$ {usd:.3f}/1k rótulos", TAB_C, "escolha-do-oraculo",
        ART_E0, COD_E0, "rastreado" if ok else "divergente",
        "" if ok else f"artefato: {r['cost_per_1k_labels_usd']}")

flash = e0_linha("deepseek-v4-flash", "rand")
g4o = e0_linha("gpt-4o@", "rand")
razao = g4o["cost_per_1k_labels_usd"] / flash["cost_per_1k_labels_usd"]
add("v4-flash é 26x mais barato que o gpt-4o", TAB_C, "escolha-do-oraculo", ART_E0,
    COD_E0, "rastreado" if 25.5 <= razao <= 26.5 else "divergente",
    f"razão calculada: {razao:.1f}x")

cache_max = max(r["cache_hit_rate"] for r in e0_tab) * 100
cache_min = min(r["cache_hit_rate"] for r in e0_tab if r["cache_hit_rate"] > 0) * 100
add("cache de prefixo atinge 88–95% nos provedores que o suportam",
    "5-resultados-falco/texto.tex#sec:res-e0-rq2", "escolha-do-oraculo", ART_E0, COD_E0,
    "rastreado" if (87.5 <= cache_min and cache_max <= 95.5) else "divergente",
    f"faixa no artefato: {cache_min:.1f}–{cache_max:.1f}%")

# ------------------------------------------------- E6 (auditado no notebook)
ART_E6 = "experiments/e6population/results/analysis_multiseed.json"
COD_E6 = "experiments/e6population/run_population_curve.py"
NB_E6 = "ghdaru/falco-auditoria-escala-populacional"
mb = e6_multi["por_braco"]
for rot, chave, tese_med, tese_sd in [
        ("saturação SGD entropia: 9,1k ± 0,6k", "sgd_entropy", 9100, 600),
        ("saturação SGD aleatório: 15,5k ± 1,1k", "sgd_random", 15500, 1100),
        ("saturação PVBin entropia: 19,3k ± 0,7k", "pvbin_entropy", 19300, 700),
        ("saturação PVBin aleatório: 40,6k ± 1,4k", "pvbin_random", 40600, 1400)]:
    ok = perto(mb[chave]["sat_mean"], tese_med, 50) and perto(mb[chave]["sat_sd"], tese_sd, 50)
    add(rot, "5-resultados-falco/texto.tex#sec:res-e6", "escala-populacional", ART_E6,
        COD_E6, "rastreado" if ok else "divergente",
        f"artefato: {mb[chave]['sat_mean']} ± {mb[chave]['sat_sd']}", NB_E6)

add("viés de autoavaliação em |L|=10k: −17,1 ± 1,0 p.p.",
    "5-resultados-falco/texto.tex#sec:res-e6", "escala-populacional", ART_E6, COD_E6,
    "rastreado", "artefato: -17.1 ± 1.0", NB_E6)
add("teto da entropia supera o aleatório em 8 das 8 sementes (p=0,0078)",
    "5-resultados-falco/texto.tex#sec:res-e6", "escala-populacional", ART_E6, COD_E6,
    "rastreado", "artefato: 8/8, p=0.0078", NB_E6)
add("população reservada do E6 ≈ 140 mil instâncias",
    "5-resultados-falco/texto.tex#sec:res-e6", "escala-populacional",
    "experiments/e6population/results/popcurve_pvbin_entropy_summary.json", COD_E6,
    "divergente",
    "artefato registra population=181490, e o runner faz dedup[50000:] = "
    "231490-50000. O próprio texto diz 'todo o restante', o que contradiz os 140 mil",
    NB_E6)

# ------------------------------------------------- E3' (auditado no notebook)
NB_E3 = "ghdaru/falco-auditoria-classificador-forte"
COD_E3 = "experiments/e2e3/run_e3prime.py"
PAREADO = "experiments/e2e3/results/legacy_s42_bs16_eval20k/"
add("Tabela e3p (A/B/C/E/D): acurácia, IC de Wilson e Macro F1",
    "5-resultados-falco/texto.tex#tab:e3p", "classificador-forte", PAREADO, COD_E3,
    "rastreado", "14 de 15 afirmações conferem; IC recalculado no notebook", NB_E3)
add("ganho de acurácia de C sobre B: +3,0 p.p.",
    "5-resultados-falco/texto.tex#sec:res-e3p", "classificador-forte", PAREADO, COD_E3,
    "divergente", "a conta dá 2,93 p.p.; arredondamento para cima", NB_E3)
add("McNemar A–B, B–C, E35–D e IC bootstrap das diferenças de Macro F1",
    "5-resultados-falco/texto.tex#sec:res-e3p", "classificador-forte",
    "experiments/e2e3/results/{mcnemar_s42,bootstrap_f1_s42}.json",
    "experiments/e2e3/stats_e3prime.py", "rastreado",
    "recomputados do zero no notebook: b/c exatos e os três IC idênticos", NB_E3)
add("E35 supera a régua D (leitura (iii) da varredura)",
    "5-resultados-falco/texto.tex#sec:res-e3p-varredura", "classificador-forte",
    "experiments/e2e3/results/e3prime_*_s{42,7,123}.json", COD_E3, "divergente",
    "vale no regime pareado publicado (lote 16), mas em NENHUMA das três sementes "
    "canônicas (lote 128). A afirmação é efeito do regime, não achado robusto", NB_E3)
add("piso de orçamento: critério de Macro F1 passa a valer em 25 mil rótulos",
    "5-resultados-falco/texto.tex#sec:res-e3p-varredura", "classificador-forte",
    "experiments/e2e3/results/e3prime_*_s{42,7,123}.json", COD_E3, "divergente",
    "no regime canônico o E25 não cruza em nenhuma das três sementes; só o E35 "
    "cruza, e em 2 de 3", NB_E3)

# ------------------------------------------------- Cap. 4 (legado)
add("sensibilidade do L0: acurácia de 6,7% (I=10) a 89,1% (I=200.000)",
    "4-resultados-l0/texto.tex#sec:res-l0-sens", "conjunto-inicial",
    "GHDaru/activetextclassification examples/data/sensibilidade/"
    "l0_random_impact_metrics_PVBin.xlsx",
    "GHDaru/activetextclassification examples/L0_experimento.ipynb", "legado",
    "0.891 está na saída gravada da célula 4. Evidência existe mas em repositório "
    "somente leitura e sem referência cruzada; o REPRODUCIBILITY.md aponta para "
    "'Tese-Vers-o-Draft', que não é este repositório")

# ---------------------------------------------------------------- gravação
resumo = {}
for it in itens:
    resumo[it["status"]] = resumo.get(it["status"], 0) + 1

doc = {
    "schema": "rastreabilidade/v1",
    "gerado_por": "executor01 · notebooks/auditoria/build_rastreabilidade.py",
    "cobertura": "Cap. 5 (E0, E6, E3′) + Cap. 4 (P1). Faltam: E0-P, E1, E4, "
                 "seção do gate, Caps. 3 e 6, apêndices, pré-textuais",
    "resumo": resumo,
    "legenda": {
        "rastreado": "o número sai do artefato citado",
        "divergente": "o artefato existe e diz outra coisa",
        "sem-evidencia": "não há artefato que sustente o número",
        "legado": "evidência em repositório legado, somente leitura",
    },
    "itens": itens,
}
SAIDA.parent.mkdir(parents=True, exist_ok=True)
SAIDA.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"escrito: {SAIDA}")
print(f"itens: {len(itens)} · {resumo}")
for it in itens:
    if it["status"] in ("divergente", "sem-evidencia"):
        print(f"  [{it['status']}] {it['numero']}")
