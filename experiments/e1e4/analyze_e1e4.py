"""E1/E4 — agregação dos sweeps (LCE e Macro-F1 final por célula).

E1: 5 estratégias × 8 sementes, ruído 0, lote 100; ablação de lote (50/200).
E4: {entropia, aleatória} × ε∈{0.1,0.2,0.4} × 8 sementes, lote 100.
Wilcoxon pareado por semente (scipy). Gera results/analysis.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev

from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent
runs = [json.loads(l) for l in (ROOT / "results/sweeps.jsonl").read_text().splitlines() if l.strip()]
baseline = json.loads((ROOT / "results/baseline.json").read_text())


def cell(exp=None, strategy=None, noise=None, batch=None):
    sel = [r for r in runs
           if (exp is None or r["exp"] == exp)
           and (strategy is None or r["strategy"] == strategy)
           and (noise is None or r["noise"] == noise)
           and (batch is None or r["batch"] == batch)]
    return sorted(sel, key=lambda r: r["seed"])


def agg(sel, key):
    vals = [r[key] for r in sel]
    return {"mean": round(mean(vals), 4),
            "sd": round(stdev(vals), 4) if len(vals) > 1 else 0.0,
            "n_seeds": len(vals)}


out = {"baseline_macro_f1_full_pool": round(baseline["macro_f1"], 4)}

# --- E1: estratégias, ruído 0, lote 100 ---
strategies = sorted({r["strategy"] for r in runs if r["exp"] == "e1"})
e1 = {}
rand_cell = cell("e1", "random", 0.0, 100)
for s in strategies:
    c = cell("e1", s, 0.0, 100)
    e1[s] = {"lce": agg(c, "lce"), "final_macro_f1": agg(c, "final_macro_f1")}
    if s != "random" and len(c) == len(rand_cell):
        for key in ("lce", "final_macro_f1"):
            a = [r[key] for r in c]
            b = [r[key] for r in rand_cell]
            stat, p = wilcoxon(a, b)
            e1[s][f"wilcoxon_vs_random_{key}_p"] = round(float(p), 6)
out["e1_strategies_noise0_b100"] = e1

# --- E1b: ablação de lote (entropia — estratégia do E4, executada em b=50/200) ---
e1b = {}
for b in (50, 100, 200):
    c = [r for r in cell(None, "entropy", 0.0, b) if r["exp"] in ("e1", "e1b")]
    if c:
        e1b[f"b{b}"] = {"lce": agg(c, "lce"), "final_macro_f1": agg(c, "final_macro_f1")}
out["e1b_batch_ablation"] = {"strategy": "entropy", "cells": e1b}

# --- E4: ruído ---
e4 = {}
ent0 = cell("e1", "entropy", 0.0, 100)
rnd0 = cell("e1", "random", 0.0, 100)
for eps in (0.1, 0.2, 0.4):
    row = {}
    for s in ("entropy", "random"):
        c = cell("e4", s, eps, 100)
        ref = ent0 if s == "entropy" else rnd0
        row[s] = {"lce": agg(c, "lce"), "final_macro_f1": agg(c, "final_macro_f1"),
                  "f1_retention_vs_eps0": round(
                      agg(c, "final_macro_f1")["mean"] / agg(ref, "final_macro_f1")["mean"], 4)}
    ce, cr = cell("e4", "entropy", eps, 100), cell("e4", "random", eps, 100)
    for key in ("lce", "final_macro_f1"):
        stat, p = wilcoxon([r[key] for r in ce], [r[key] for r in cr])
        row[f"wilcoxon_entropy_vs_random_{key}_p"] = round(float(p), 6)
    e4[f"eps{eps}"] = row
out["e4_noise"] = e4

path = ROOT / "results/analysis.json"
path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(json.dumps(out, indent=2, ensure_ascii=False))
