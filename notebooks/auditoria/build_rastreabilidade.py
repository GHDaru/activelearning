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
    """Comparação numérica com folga para o erro de ponto flutuante.

    O `1e-9` não é preciosismo: sem ele, 0,006 − 0,0055 dá
    0,0005000000000000004 e estoura uma tolerância de exatamente 0,0005 —
    o que marcaria como divergentes três números do E1/E4 que na verdade
    conferem. Auditoria que inventa divergência é pior que auditoria nenhuma.
    """
    return a is not None and b is not None and abs(a - b) <= tol + 1e-9


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

# ------------------------------------------------- E0/E0-P · confirmação da Onda 3a
# Reanálise pura (script real rodado sobre as anotações cruas versionadas, sem
# chave de API): zero divergências no pipeline inteiro. Reforça, não repete,
# os itens ponto-a-ponto já registrados acima — é o veredito de nível de
# PIPELINE que só um recálculo completo (não célula a célula) sustenta.
add("pipeline do E0 é reprodutível ponta a ponta do dado bruto",
    "experiments/e0/analyze_e0.py", "escolha-do-oraculo", ART_E0 + " + " + ART_MC,
    COD_E0, "rastreado",
    "reanálise completa (13/13 linhas da tabela, 43/43 pares de McNemar) roda "
    "zero-custo sobre as anotações já versionadas e reproduz os artefatos "
    "exatamente — inclusive a AUSÊNCIA de b=43/c=16 em qualquer par",
    "notebooks/auditoria/escolha-do-oraculo.ipynb")

# ------------------------------------------------- E0-P · ablação de prompt
e0p = ler("e0p/results/analysis.json")
ART_E0P = "experiments/e0p/results/analysis.json"
COD_E0P = "experiments/e0p/run_e0p.py + analyze_e0p.py"
TAB_P = "5-resultados-falco/texto.tex#tab:e0p"

add("pipeline do E0-P é reprodutível ponta a ponta do dado bruto",
    "experiments/e0p/analyze_e0p.py", "efeito-do-prompt", ART_E0P, COD_E0P,
    "rastreado",
    "reanálise completa (6/6 acurácias, 6/6 pares de McNemar) roda zero-custo "
    "e reproduz o analysis.json publicado exatamente",
    "notebooks/auditoria/efeito-do-prompt.ipynb")

for amostra, variante, acc, disc, pv in [
        ("rand", "v3", 60.4, None, None),
        ("rand", "v4a", 64.2, (50, 31), 0.045),
        ("rand", "v4b", 65.0, (50, 27), 0.012),
        ("strat", "v3", 51.8, None, None),
        ("strat", "v4a", 44.8, (20, 55), 0.001),
        ("strat", "v4b", 41.0, (14, 68), 0.001)]:
    d = e0p[amostra][variante]
    problemas = []
    if not perto(d["accuracy"] * 100, acc, 0.05):
        problemas.append(f"acurácia {d['accuracy']*100:.1f}% no artefato")
    if disc:
        mc = e0p[amostra][f"mcnemar_v3_vs_{variante}"]
        # a tese escreve (+/-) = (a favor da variante / contra)
        if (mc["only_b_correct"], mc["only_a_correct"]) != disc:
            problemas.append(f"discordantes {mc['only_b_correct']}/{mc['only_a_correct']} "
                             "no artefato")
        if pv == 0.001:                      # a tese escreve "<0,001"
            if not mc["p_value"] < 0.001:
                problemas.append(f"p={mc['p_value']} não é < 0,001")
        elif not perto(mc["p_value"], pv, 0.0006):
            problemas.append(f"p={mc['p_value']:.4f} no artefato")
    add(f"E0-P {amostra} {variante}: acurácia {acc}%"
        + (f" · discordantes {disc[0]}/{disc[1]} · p {pv}" if disc else ""),
        TAB_P, "efeito-do-prompt", ART_E0P, COD_E0P,
        "divergente" if problemas else "rastreado", "; ".join(problemas))

mc_ab = e0p["strat"]["mcnemar_v4a_vs_v4b"]
add("E0-P: v4b piora sobre v4a na S-strat com p=0,0013", TAB_P, "efeito-do-prompt",
    ART_E0P, COD_E0P,
    "rastreado" if perto(mc_ab["p_value"], 0.0013, 0.00005) else "divergente",
    f"artefato p={mc_ab['p_value']}")

# ------------------------------------------------- E1 · estratégias de seleção
e1e4 = ler("e1e4/results/analysis.json")
ART_E1 = "experiments/e1e4/results/analysis.json"
COD_E1 = "experiments/e1e4/run_sweeps.py + analyze_e1e4.py"
e1 = e1e4["e1_strategies_noise0_b100"]

add("teto supervisionado do pool completo: Macro F1 = 0,540",
    "5-resultados-falco/texto.tex#sec:res-e1", "estrategias-de-selecao", ART_E1, COD_E1,
    "rastreado" if perto(e1e4["baseline_macro_f1_full_pool"], 0.540, 0.0005) else "divergente",
    f"artefato: {e1e4['baseline_macro_f1_full_pool']}")

for chave, rot, lce, lce_sd, f1, f1_sd in [
        ("smallest_margin", "menor margem", 0.528, 0.013, 0.418, 0.013),
        ("least_confidence", "menor confiança", 0.518, 0.010, 0.421, 0.009),
        ("entropy", "entropia", 0.493, 0.006, 0.398, 0.008),
        ("hybrid", "híbrida", 0.476, 0.014, 0.379, 0.008),
        ("random", "aleatória", 0.444, 0.011, 0.339, 0.006)]:
    c = e1[chave]
    problemas = []
    for nome, obtido, esperado, tol in [
            ("LCE", c["lce"]["mean"], lce, 0.0005),
            ("desvio do LCE", c["lce"]["sd"], lce_sd, 0.0005),
            ("F1 final", c["final_macro_f1"]["mean"], f1, 0.0005),
            ("desvio do F1", c["final_macro_f1"]["sd"], f1_sd, 0.0005)]:
        if not perto(obtido, esperado, tol):
            problemas.append(f"{nome} {obtido} no artefato")
    add(f"E1 {rot}: LCE {lce}±{lce_sd} · F1 final {f1}±{f1_sd}",
        "5-resultados-falco/texto.tex#tab:e1", "estrategias-de-selecao", ART_E1, COD_E1,
        "divergente" if problemas else "rastreado", "; ".join(problemas))

melhor = max(e1[k]["final_macro_f1"]["mean"] for k in e1)
recuperado = melhor / e1e4["baseline_macro_f1_full_pool"] * 100
add("a melhor célula recupera 78% do teto supervisionado com 15% dos rótulos",
    "5-resultados-falco/texto.tex#sec:res-e1", "estrategias-de-selecao", ART_E1, COD_E1,
    "rastreado" if 77.5 <= recuperado <= 78.5 else "divergente",
    f"conta: {recuperado:.1f}%")

lotes = e1e4["e1b_batch_ablation"]["cells"]
for celula, lce, sd in [("b50", 0.492, 0.009), ("b100", 0.493, 0.006), ("b200", 0.481, 0.012)]:
    c = lotes[celula]["lce"]
    ok = perto(c["mean"], lce, 0.0005) and perto(c["sd"], sd, 0.0005)
    add(f"E1b ablação de lote {celula}: LCE {lce}±{sd}",
        "5-resultados-falco/texto.tex#sec:res-e1", "estrategias-de-selecao", ART_E1, COD_E1,
        "rastreado" if ok else "divergente", f"artefato: {c['mean']}±{c['sd']}")

# ------------------------------------------------- E4 · robustez ao ruído
e4 = e1e4["e4_noise"]
for eps, estrategia, rot, f1, sd, ret in [
        (0.1, "entropy", "entropia", 0.347, 0.008, 87.2),
        (0.1, "random", "aleatória", 0.294, 0.006, 86.7),
        (0.2, "entropy", "entropia", 0.294, 0.006, 74.0),
        (0.2, "random", "aleatória", 0.252, 0.006, 74.4),
        (0.4, "entropy", "entropia", 0.215, 0.010, 54.1),
        (0.4, "random", "aleatória", 0.186, 0.006, 54.8)]:
    c = e4[f"eps{eps}"][estrategia]
    problemas = []
    if not perto(c["final_macro_f1"]["mean"], f1, 0.0005):
        problemas.append(f"F1 {c['final_macro_f1']['mean']} no artefato")
    if not perto(c["final_macro_f1"]["sd"], sd, 0.0005):
        problemas.append(f"desvio {c['final_macro_f1']['sd']} no artefato")
    if not perto(c["f1_retention_vs_eps0"] * 100, ret, 0.05):
        problemas.append(f"retenção {c['f1_retention_vs_eps0']*100:.1f}% no artefato")
    add(f"E4 ε={eps} {rot}: F1 {f1}±{sd} · retenção {ret}%",
        "5-resultados-falco/texto.tex#tab:e4", "robustez-ao-ruido", ART_E1, COD_E1,
        "divergente" if problemas else "rastreado", "; ".join(problemas))

todos_p = [e4[f"eps{e}"]["wilcoxon_entropy_vs_random_final_macro_f1_p"] for e in (0.1, 0.2, 0.4)]
add("a vantagem da entropia sobrevive com p=0,0078 em todo ε",
    "5-resultados-falco/texto.tex#sec:res-e4", "robustez-ao-ruido", ART_E1, COD_E1,
    "rastreado" if all(perto(p, 0.0078, 0.0001) for p in todos_p) else "divergente",
    f"artefato: {todos_p}")

# ------------------------------------------------- E1/E4 (Onda 4 — reexecutado)
sweeps_path = RAIZ / "experiments/e1e4/results/sweeps.jsonl"
add("as 104 células do E1/E1b/E4 (sweeps.jsonl)",
    "5-resultados-falco/texto.tex#sec:res-e1", "estrategias-de-selecao",
    "experiments/e1e4/results/sweeps.jsonl", COD_E1,
    "rastreado" if sweeps_path.exists() else "sem-evidencia",
    "reexecutado na Onda 4 (run_sweeps.py roda de novo, não é reanálise — o dado "
    "bruto nunca esteve versionado). Pool determinístico (semente 7 fixa no "
    "script): a reprodução do E1 e do E4 bateu quase exata (3ª/4ª casa decimal) "
    "com os agregados publicados. git add -f — casa com a linha 7 do .gitignore",
    "notebooks/auditoria/estrategias-e-robustez.ipynb")

# ------------------------------------------------- dados brutos que faltam

add("figuras do Cap. 4 e do Cap. 5 geradas por script",
    "experiments/plots/", "todos", "experiments/plots/figures/*.{pdf,png}",
    "experiments/plots/make_figures.py", "sem-evidencia",
    "o script existe e nenhuma figura está versionada em experiments/plots/. As "
    "figuras publicadas vivem em tesedaru/N-*/imagens/, desacopladas do gerador — "
    "não há garantia de que a figura da tese corresponda ao artefato atual")

# ------------------------------------------------- P1/P2 (Onda 2 — reexecutado)
ART_P1 = "experiments/p1/results/replay_l0.jsonl"
ART_P2 = "experiments/p1/results/replay_ga.jsonl"
COD_P1 = "experiments/p1/replay_l0_sensitivity.py"
COD_P2 = "experiments/p1/replay_ga.py"
NB_P1 = "notebooks/auditoria/conjunto-inicial.ipynb"

for tam, acc_pub, delta in [(10, 6.7, -0.1), (100, 24.7, -0.7), (1_000, 55.9, -0.4),
                            (10_000, 76.9, -0.5), (200_000, 89.1, -0.3)]:
    add(f"P1 sensibilidade |L0|={tam}: acurácia {acc_pub}%",
        "4-resultados-l0/texto.tex#sec:res-l0-sens", "conjunto-inicial", ART_P1, COD_P1,
        "rastreado", f"replay: {acc_pub + delta:.1f}% (Δ {delta:+.1f} p.p., dentro de "
        "1,0 p.p. de folga)", NB_P1)

add("P1 amplitude em |L0|=100: 6,4 p.p. entre repetições",
    "4-resultados-l0/texto.tex#sec:res-l0-sens", "conjunto-inicial", ART_P1, COD_P1,
    "rastreado", "replay (10 reps, vs. 30 originais): 4,8 p.p. — mesmo fenômeno, "
    "amplitude menor por ter menos repetições, não é refutação", NB_P1)

add("P2 inflação de circularidade em max_f1 |L0|=500: +6,3 p.p.",
    "docs/convergencia-replays.md#C2", "conjunto-inicial", ART_P2, COD_P2, "rastreado",
    "replay REPRODUZ EXATAMENTE: +6,3 p.p. (19,4% partição de aptidão vs. 13,1% teste "
    "intocado, valores idênticos ao relatório que estava sem artefato)", NB_P1)

add("P2 ganho do AG sobre a média aleatória em |L0|=50: +5,2 p.p.",
    "docs/convergencia-replays.md#C2", "conjunto-inicial", ART_P2, COD_P2, "divergente",
    "replay dá +1,3 p.p. — mesma direção (o AG vence), magnitude ~4x menor. CORREÇÃO "
    "desta própria auditoria: a divergência de magnitude é ESPERADA, não anômala — "
    "D-002 (docs/decisoes.md) documenta que o replay usa escala deliberadamente "
    "reduzida (N_pop=30, 40 gerações) vs. o original (N_pop=50, 100 gerações); "
    "mantido como 'divergente' porque o NÚMERO ainda diverge do texto, mas a causa "
    "está identificada e documentada, não é acaso de investigar", NB_P1)

# ------------------------------------------------- Cap. 3 (método) — R5 casado
# Tarefa 20260817-1940. Cobre os números de 3-metodo/texto.tex que ainda não
# apareciam em nenhuma seção acima (os que já apareciam — S-rand/S-strat do
# E0, ε do E4 — ganham aqui só uma entrada de referência cruzada, não duplicam
# a auditoria: a rastreabilidade de fato já foi feita nas seções de Cap. 5).
M3 = "3-metodo/texto.tex"

add("N=250.221 descrições (versão corrigida)", f"{M3}#sec:metodo-dados",
    "todos", "data/dataset.csv", "leitura direta do CSV", "rastreado",
    "250.221 linhas — bate exato", "")

add("N=250.365 na versão original; 144 linhas 'inativo' removidas",
    f"{M3}#sec:metodo-dados", "todos", "data/dataset.csv",
    "leitura direta do CSV", "rastreado",
    "250.365 − 250.221 = 144 — inferido por subtração, não há artefato "
    "intermediário da versão 'original' pré-correção versionado; a aritmética "
    "fecha mas não há como auditar a lista das 144 linhas em si", "")

add("classe mais frequente 'biscoito': 14.292 instâncias (5,7%)",
    f"{M3}#sec:metodo-dados", "todos", "data/dataset.csv",
    "leitura direta do CSV (Counter por nm_product)", "rastreado",
    "14.292/250.221 = 5,71% — bate exato", "")

add("descrições curtas: 4 a 50 caracteres, mediana 32",
    f"{M3}#sec:metodo-dados", "todos", "data/dataset.csv",
    "leitura direta do CSV (len(nm_item))", "rastreado",
    "min=4, max=50, mediana=32 — bate exato", "")

add("conflitos de rótulo: 719 descrições distintas (1.807 linhas; 0,7%)",
    f"{M3}#sec:metodo-dados-auditoria", "escolha-do-oraculo",
    "experiments/e0/results/noise_impact.json",
    "experiments/e0/analyze_noise_impact.py", "divergente",
    "script canônico (o mesmo que sustenta a análise de sensibilidade citada no "
    "próprio parágrafo) reexecutado agora: 693 descrições / 1.720 linhas (0,69%). "
    "O percentual arredondado bate (0,7%), os inteiros não — 719 vs 693 (+3,7%), "
    "1.807 vs 1.720 (+5,1%). Não é ponto flutuante: são 26 descrições e 87 linhas "
    "de diferença. Divergência real, reportada — não corrigida",
    "notebooks/auditoria/escolha-do-oraculo.ipynb")

add("CategorySchema: 620 classes frequentes + 1 sentinela (_rare_) = 621",
    f"{M3}#sec:metodo-dados-preprocessamento", "todos", "data/dataset.csv",
    "min_samples_per_class=5 (experiments/e0/config.json)", "rastreado",
    "621 classes com ≥5 exemplos + _rare_ — bate exato", "")

add("337 linhas em classes raras (0,135%)",
    f"{M3}#sec:metodo-dados-preprocessamento", "todos", "data/dataset.csv",
    "min_samples_per_class=5 (experiments/e0/config.json)", "rastreado",
    "337/250.221 = 0,1347% ≈ 0,13% — bate", "")

add("particionamento deduplicado: 231.490 textos únicos / 714 classes presentes",
    f"{M3}#sec:metodo-dados-particionamento", "escala-populacional",
    "experiments/e6population/", "método já validado nas Ondas anteriores "
    "(Counter de rótulo bruto ANTES do dedup para o filtro ≥2, chave de dedup "
    "= texto.strip().lower(), não normalize_label completo)", "rastreado",
    "231.490 / 714 — bate exato com o método já usado nas auditorias de E6/E3′",
    "notebooks/auditoria/escala-populacional.ipynb")

add("pool=50.000; holdout=4.000 (2.000+2.000); população reservada≈177 mil",
    f"{M3}#sec:metodo-dados-particionamento", "escala-populacional",
    "experiments/e6population/", "50.000+4.000+177.490=231.490", "rastreado",
    "177.490 bate com '≈177 mil' do PRÓPRIO Cap. 3 — nota importante: isso NÃO é "
    "o mesmo achado da divergência já reportada em escala-populacional (Cap. 5), "
    "onde o texto do Cap. 5 diz '≈140 mil' para a mesma população reservada. "
    "Ou seja: o Cap. 3 (aqui) está CORRETO/bate com o artefato; é o Cap. 5 que "
    "diverge — do artefato E também do próprio Cap. 3. Reforça o achado #3 já "
    "levantado, agora com uma segunda fonte textual confirmando o valor certo",
    "notebooks/auditoria/escala-populacional.ipynb")

add("JS divergence pool×população = 0,0022; Spearman = 0,983",
    f"{M3}#sec:metodo-dados-particionamento", "escala-populacional",
    "experiments/e6population/results/analysis.json",
    "experiments/e6population/", "rastreado", "bate exato", "")

add("649/714 classes no pool (90,9%); 65 ausentes; 179 com <5 exemplos",
    f"{M3}#sec:metodo-dados-particionamento", "escala-populacional",
    "experiments/e6population/", "649+65=714; 649/714=90,90%", "rastreado",
    "bate exato", "")

add("≈70 exemplos/classe em média no pool",
    f"{M3}#sec:metodo-dados-particionamento", "escala-populacional",
    "experiments/e6population/", "50.000/714=70,03", "rastreado",
    "denominador correto é 714 (total de classes do schema), não 649 "
    "(classes presentes) — com 649 daria 77,04, não bate; com 714 bate", "")

add("BERTimbau: taxa de aprendizado 3×10⁻⁵",
    f"{M3}#sec:metodo-classificadores", "classificador-forte",
    "src/activelearning/adapters/classifiers/bertimbau.py",
    "BertimbauClassifier.__init__ (default learning_rate)", "divergente",
    "código usa 5×10⁻⁵ em TODO lugar onde a taxa aparece: default da classe "
    "(5e-5), experiments/e2e3/train_full.py --lr default (5e-5), "
    "bertimbau_colab_tpu.ipynb célula LR=5e-5 (o notebook do E2 de verdade). "
    "Nenhuma ocorrência de 3e-5 em nenhum script ou notebook do repositório. "
    "run_e3prime.py nem expõe --lr como argumento — sempre usa o default da "
    "classe, portanto sempre 5e-5. Divergência real, reportada — não corrigida",
    "notebooks/auditoria/classificador-forte.ipynb")

add("BERTimbau: lote de treinamento 32",
    f"{M3}#sec:metodo-classificadores", "classificador-forte",
    "src/activelearning/adapters/classifiers/bertimbau.py",
    "BertimbauClassifier.__init__ (default batch_size)", "divergente",
    "código usa batch_size=16 por default (CPU) e 128 no notebook real do E2 "
    "(bertimbau_colab_tpu.ipynb: 'BATCH=128 if DEVICE_KIND!=\"cpu\" else 16'); "
    "32 nunca aparece como tamanho de lote em lugar nenhum do repositório. "
    "HIPÓTESE (não confirmada, reportada como tal): o mesmo construtor "
    "(BertimbauClassifier.__init__) tem, na linha seguinte ao batch_size, um "
    "parâmetro DIFERENTE chamado max_length também com default 32 — é o "
    "comprimento máximo de tokens, não o tamanho do lote. Provável troca dos "
    "dois parâmetros na redação do Cap. 3. Divergência real, reportada — "
    "não corrigida, hipótese de causa incluída para o principal avaliar",
    "notebooks/auditoria/classificador-forte.ipynb")

add("BERTimbau: decaimento de peso 0,01",
    f"{M3}#sec:metodo-classificadores", "classificador-forte",
    "src/activelearning/adapters/classifiers/bertimbau.py",
    "torch.optim.AdamW(...) sem weight_decay explícito", "rastreado",
    "o código NÃO fixa weight_decay explicitamente — mas 0,01 é o default do "
    "próprio torch.optim.AdamW; portanto o valor efetivo bate, por herança do "
    "default da biblioteca, não por parâmetro escrito no código", "")

add("épocas por iteração determinadas empiricamente no E2 (|L|∈{10³,10⁴,5×10⁴})",
    f"{M3}#sec:metodo-classificadores", "classificador-forte",
    "experiments/e2e3/bertimbau_colab_tpu.ipynb", "leitura do notebook",
    "sem-evidencia",
    "o notebook real do E2 não fixa LIMIT∈{1000,10000,50000} como conjunto de "
    "varredura — não há artefato de resultado do E2 versionado (nenhum "
    "experiments/e2e3/results/e2*) que permita confirmar quais |L| foram de "
    "fato varridos. Sem evidência recuperável do conjunto exato usado",
    "")

add("S-rand n=1.000; S-strat 3/classe, n≈1.863",
    f"{M3}#sec:metodo-oraculo-desenho", "escolha-do-oraculo",
    "experiments/e0/config.json + experiments/e0/results/{rand,strat}/",
    "config.json: samples.random_size=1000, stratified_per_class=3",
    "rastreado", "wc -l dos JSONL reais: rand=1.000, strat=1.863 — bate exato. "
    "Referência cruzada: já coberto pelas entradas de E0 em Cap. 5; incluído "
    "aqui só porque o número também é citado no Cap. 3", "")

add("critério de decisão do oráculo: piso de 85% de acurácia na S-rand",
    f"{M3}#sec:metodo-oraculo-decisao", "escolha-do-oraculo",
    "experiments/e0/config.json + experiments/e0/results/e0_table.json",
    "e0_table.json (accuracy por oráculo, amostra rand)", "divergente",
    "NENHUM oráculo do E0 atinge 85% de acurácia na S-rand — o melhor é "
    "deepseek-v4-pro com 82,1%. O próprio experiments/e0/config.json anota "
    "deepseek-v4-flash (78,3% no rand) como '_nota: LLM Inicial candidato' e "
    "deepseek-v4-pro (82,1%) como '_nota: LLM Avancado candidato' — ou seja, a "
    "escolha real registrada no config não satisfaz o piso de 85% que o "
    "próprio Cap. 3 declara como critério. O texto também prevê esse caso "
    "('se nenhum modelo atingir o limiar, o E4 torna-se obrigatório'), mas não "
    "há, nos artefatos, registro de que essa branch alternativa foi de fato "
    "adotada — ambiguidade entre a regra escrita e a escolha registrada",
    "notebooks/auditoria/escolha-do-oraculo.ipynb")

add("E4: ε ∈ {0; 0,1; 0,2; 0,4}", f"{M3}#sec:metodo-falco", "robustez-ao-ruido",
    "experiments/e1e4/run_sweeps.py", "run_sweeps.py linha 5 e loop 'for noise "
    "in (0.1, 0.2, 0.4)'", "rastreado",
    "código varre noise∈{0,1;0,2;0,4} explicitamente; ε=0 é o próprio E1 "
    "(oráculo perfeito), reaproveitado como o ponto zero da mesma curva — "
    "não há um 4º valor de noise=0 rodado à parte, é E1 relido. Referência "
    "cruzada: já coberto pelas entradas de E1/E4 em Cap. 5",
    "notebooks/auditoria/estrategias-e-robustez.ipynb")

add("estagnação: p=5 iterações, ε=10⁻³ de tolerância",
    f"{M3}#sec:metodo-falco", "todos", "docs/architecture.md ou config do E5",
    "busca no repositório", "sem-evidencia",
    "não localizado nenhum config/script com p=5 ou tolerância 1e-3 de "
    "estagnação versionado e executável fora do E5 (que está bloqueado pelo "
    "cache do oráculo, mesmo bloqueio já registrado para E5/braços A-C do E3′)",
    "")

add("E3′ braço A: |A|/|D| ≈ 18% dos rótulos",
    f"{M3}#sec:metodo-falco-baselines", "classificador-forte",
    "experiments/e2e3/results/ (braços A/B/C)", "run_e3prime.py braços A/B/C",
    "sem-evidencia",
    "mesmo bloqueio já registrado: braços A/B/C do E3′ dependem do "
    "annotation_cache_nemotron.jsonl, fora do repositório — não há como "
    "calcular |A| (cardinalidade real do cache) para conferir a proporção",
    "")

add("gate de calibração de lote do E0: maior lote sem degradar acurácia "
    "(McNemar, lotes de 1/10/25)", f"{M3}#sec:metodo-oraculo-instrumentacao",
    "escolha-do-oraculo", "experiments/e0/", "busca no repositório",
    "sem-evidencia",
    "não localizado um artefato/script versionado com essa calibração "
    "específica (comparação de lotes 1 vs 10 vs 25 por McNemar) — os "
    "items_per_call finais (10 ou 25, por provedor) estão no config.json, mas "
    "a análise que justificou a escolha não está versionada", "")

# ---------------------------------------------------------------- gravação
resumo = {}
for it in itens:
    resumo[it["status"]] = resumo.get(it["status"], 0) + 1

doc = {
    "schema": "rastreabilidade/v1",
    "gerado_por": "executor01 · notebooks/auditoria/build_rastreabilidade.py",
    "cobertura": "Cap. 5 completo menos a seção do gate (E0, E0-P, E1, E4, E6, E3′) + Cap. 4 (P1/P2) + Cap. 3 (método, ~24 números casados com R5, tarefa 20260817-1940). Faltam: seção do gate, Cap. 6, apêndices, pré-textuais",
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
