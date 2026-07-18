"""E6 — análise consolidada: seletores × classificadores, viés interna×externa.

Para cada braço (classificador × estratégia): F1/acc externos em marcos de
orçamento, ponto de 95% do teto externo (saturação), e viés de autoavaliação
(interna − externa) médio em três regiões da curva.
Gera results/analysis.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).parent / "results"
STRATEGIES = ("entropy", "random", "drisl", "drisl-c")
CLASSIFIERS = ("pvbin", "sgd")
MARKS = (2000, 5000, 10000, 20000, 50000)


def load(clf, strat):
    p = ROOT / f"popcurve_{clf}_{strat}.jsonl"
    if not p.exists():
        return None
    pts, seen = [], set()
    for line in p.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            if d["n_labels"] not in seen:
                seen.add(d["n_labels"])
                pts.append(d)
    return sorted(pts, key=lambda d: d["n_labels"])


def at(pts, n):
    cand = [d for d in pts if d["n_labels"] <= n]
    return cand[-1] if cand else None


def main():
    out = {}
    for clf in CLASSIFIERS:
        for strat in STRATEGIES:
            pts = load(clf, strat)
            if not pts:
                continue
            key = f"{clf}_{strat}"
            teto = max(d["f1_ext"] for d in pts)
            sat = next((d["n_labels"] for d in pts if d["f1_ext"] >= 0.95 * teto), None)
            terco = len(pts) // 3
            out[key] = {
                "n_final": pts[-1]["n_labels"],
                "f1_ext_marcos": {str(m): (at(pts, m) or {}).get("f1_ext") for m in MARKS},
                "acc_ext_final": pts[-1]["acc_ext"],
                "f1_ext_final": pts[-1]["f1_ext"],
                "f1_ext_teto": round(teto, 4),
                "saturacao_95pct_teto": sat,
                "vies_f1_int_menos_ext": {
                    "inicio": round(mean(d["f1_int"] - d["f1_ext"] for d in pts[:terco]), 4),
                    "meio": round(mean(d["f1_int"] - d["f1_ext"] for d in pts[terco:2 * terco]), 4),
                    "fim": round(mean(d["f1_int"] - d["f1_ext"] for d in pts[2 * terco:]), 4),
                },
                "vies_acc_int_menos_ext": {
                    "inicio": round(mean(d["acc_int"] - d["acc_ext"] for d in pts[:terco]), 4),
                    "fim": round(mean(d["acc_int"] - d["acc_ext"] for d in pts[2 * terco:]), 4),
                },
            }
    (ROOT / "analysis.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
