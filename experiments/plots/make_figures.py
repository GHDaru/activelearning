"""Gera as figuras dos experimentos a partir dos artefatos (JSONL/JSON).

Fontes:
  - experiments/e1e4/results/sweeps.jsonl  -> curvas E1 e degradação E4
  - experiments/e0/results/{rand}/report_*.json -> custo × acurácia (E0)

Saída: experiments/plots/figures/*.{pdf,png} (PDF para o LaTeX, PNG para docs).

Paleta categórica validada (colorblind-safe, ordem fixa; ver docs/avaliacao-e-graficos.md):
azul, verde, magenta, amarelo, aqua. Identidade nunca só por cor — linhas têm
estilo/marcador distintos e rótulo direto no fim da linha.

Uso: python experiments/plots/make_figures.py [--only e1|e4|e0]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIGDIR = Path(__file__).parent / "figures"

# ordem fixa — nunca reciclar/reordenar (regra da paleta)
PALETTE = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a"]
STRATEGY_ORDER = ["smallest_margin", "least_confidence", "entropy", "hybrid", "random"]
STRATEGY_LABEL = {
    "smallest_margin": "menor margem",
    "least_confidence": "menor confiança",
    "entropy": "entropia",
    "hybrid": "híbrida",
    "random": "aleatória",
}
LINESTYLES = ["-", "--", "-.", ":", "-"]
MARKERS = ["o", "s", "^", "D", "v"]

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def _load_sweeps():
    path = ROOT / "experiments/e1e4/results/sweeps.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _save(fig, name: str):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"figura: {name}.pdf/.png")


def fig_e1_curvas():
    """Curvas de aprendizado do E1 (média de 8 sementes por estratégia)."""
    runs = [r for r in _load_sweeps() if r["exp"] == "e1" and r["batch"] == 100]
    by_strategy: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in runs:
        for x, y in r["curve"]:
            by_strategy[r["strategy"]][int(x)].append(float(y))
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ends = []
    for i, s in enumerate(STRATEGY_ORDER):
        pts = sorted(by_strategy[s].items())
        xs = [x for x, _ in pts]
        ys = [mean(v) for _, v in pts]
        ax.plot(xs, ys, color=PALETTE[i], linestyle=LINESTYLES[i],
                marker=MARKERS[i], markevery=6, markersize=3.5, linewidth=1.6,
                label=STRATEGY_LABEL[s])
        ends.append([s, xs[-1], ys[-1]])
    # rótulos diretos com separação mínima vertical (evita colisão nos fins de linha)
    min_gap = 0.016
    for j, (s, x, y) in enumerate(sorted(ends, key=lambda e: e[2])):
        y_label = y if j == 0 else max(y, prev + min_gap)  # noqa: F821
        prev = y_label
        ax.annotate(STRATEGY_LABEL[s], (x, y), xytext=(6, (y_label - y) * 400),
                    textcoords="offset points",
                    color="#333333", fontsize=8, va="center")
    ax.set_xlabel("rótulos adquiridos $|L|$")
    ax.set_ylabel("Macro F1 (teste)")
    ax.set_xlim(0, 3450)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    _save(fig, "fig_e1_curvas")


def fig_e4_ruido():
    """Degradação do Macro-F1 final com o ruído do oráculo (E4)."""
    runs = _load_sweeps()
    series = {"entropy": {}, "random": {}}
    for s in series:
        for eps in (0.0, 0.1, 0.2, 0.4):
            exp = "e1" if eps == 0.0 else "e4"
            vals = [r["final_macro_f1"] for r in runs
                    if r["exp"] == exp and r["strategy"] == s
                    and r["noise"] == eps and r["batch"] == 100]
            series[s][eps] = (mean(vals), stdev(vals))
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    for i, (s, label) in enumerate((("entropy", "entropia"), ("random", "aleatória"))):
        eps = sorted(series[s])
        ys = [series[s][e][0] for e in eps]
        sd = [series[s][e][1] for e in eps]
        color = PALETTE[i]
        ax.errorbar(eps, ys, yerr=sd, color=color, linestyle=LINESTYLES[i],
                    marker=MARKERS[i], markersize=4, linewidth=1.6,
                    capsize=2.5, label=label)
        ax.annotate(label, (eps[-1], ys[-1]), xytext=(5, 0),
                    textcoords="offset points", color="#333333",
                    fontsize=8, va="center")
    ax.set_xlabel("taxa de ruído do oráculo $\\varepsilon$")
    ax.set_ylabel("Macro F1 final (teste)")
    ax.set_xticks([0.0, 0.1, 0.2, 0.4])
    ax.set_xlim(-0.02, 0.50)
    ax.axvspan(0.17, 0.23, color="#bbbbbb", alpha=0.25, linewidth=0)
    ax.annotate("faixa dos LLMs\nreais (E0)", (0.20, ax.get_ylim()[0]),
                xytext=(0, 6), textcoords="offset points",
                ha="center", va="bottom", fontsize=7.5, color="#555555")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    _save(fig, "fig_e4_ruido")


def fig_e0_custo_acuracia():
    """Custo por mil rótulos × acurácia (S-rand) por oráculo do E0."""
    rows = []
    for rep in sorted((ROOT / "experiments/e0/results/rand").glob("report_*.json")):
        d = json.loads(rep.read_text())
        n = d.get("n_total", 0)
        if n < 900:  # ignora execuções parciais
            continue
        if "#free" in d["oracle_id"]:
            continue  # modo free é instrumento do RQ4, não modo de produção
        rows.append({
            "oracle": d["oracle_id"].split("@")[0],
            "acc": d["accuracy"],
            "cost": d.get("cost_per_1k_labels_usd", 0.0) or 0.0,
        })
    if not rows:
        print("fig_e0: nenhum report completo em results/rand — pulando")
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.set_xscale("symlog", linthresh=0.01)
    for r in rows:
        ax.scatter(r["cost"], r["acc"], s=42, color=PALETTE[0],
                   edgecolors="white", linewidths=1.2, zorder=3)
        ax.annotate(r["oracle"].split(":")[-1].replace("nvidia/", ""),
                    (r["cost"], r["acc"]), xytext=(5, 4),
                    textcoords="offset points", fontsize=7.5, color="#333333")
    ax.set_xlabel("custo (US\\$ por mil rótulos; escala simlog — 0 = gratuito)")
    ax.set_ylabel("acurácia na S-rand")
    ax.set_xlim(right=max(r["cost"] for r in rows) * 2.6)  # folga p/ rótulo à direita
    _save(fig, "fig_e0_custo_acuracia")


def fig_ciclo_curvas():
    """Ciclo E2E real (nemotron NIM): curva interna (validação) × externa (teste)."""
    res_dir = ROOT / "experiments/e5cycle/results"
    panels = []
    for name, title in (("pvbin", "PVBin"), ("sgd", "SGD logístico")):
        p = res_dir / f"cycle_{name}.json"
        if p.exists():
            panels.append((title, json.loads(p.read_text())))
    if not panels:
        print("fig_ciclo: sem resultados — pulando")
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6, 2.9),
                             sharey=True, sharex=True)
    axes = axes if isinstance(axes, (list, tuple)) or hasattr(axes, "__len__") else [axes]
    for ax, (title, d) in zip(axes, panels):
        for i, (key, label) in enumerate((("curve_val", "interna (validação)"),
                                          ("curve_test", "externa (teste)"))):
            xs = [x for x, _ in d[key]]
            ys = [y for _, y in d[key]]
            ax.plot(xs, ys, color=PALETTE[i], linestyle=LINESTYLES[i],
                    marker=MARKERS[i], markersize=3, linewidth=1.6, label=label)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("rótulos do oráculo $|L|$")
    axes[0].set_ylabel("Macro F1")
    axes[0].legend(loc="lower right", frameon=False, fontsize=7.5)
    _save(fig, "fig_ciclo_curvas")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["e1", "e4", "e0", "ciclo"], default=None)
    args = ap.parse_args()
    if args.only in (None, "e1"):
        fig_e1_curvas()
    if args.only in (None, "e4"):
        fig_e4_ruido()
    if args.only in (None, "e0"):
        fig_e0_custo_acuracia()
    if args.only in (None, "ciclo"):
        fig_ciclo_curvas()
