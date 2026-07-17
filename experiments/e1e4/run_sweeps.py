"""E1 (estratégias × lote) e E4 (robustez a ruído ε) — PVBin + oráculo simulado.

E1: 5 estratégias × 8 sementes, orçamento 3.000, lote 100, pool 20k (dedupe),
teste 5k fixo; + ablação de lote b∈{50,200} na melhor estratégia (entropy).
E4: ε∈{0.1, 0.2, 0.4} × {entropy, random} × 8 sementes, mesmo desenho
(ε=0 é o próprio E1).

Retomável (JSONL); LCE com baseline = supervisão completa do pool (medida uma vez).
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "experiments/p1"))

from replay_l0_sensitivity import dedupe_and_split  # noqa: E402

from activelearning.adapters.classifiers.pvbin import PVBinClassifier  # noqa: E402
from activelearning.adapters.datasets.retail_csv import load_rows_and_schema  # noqa: E402
from activelearning.adapters.oracles.simulated_oracle import SimulatedOracle  # noqa: E402
from activelearning.application.run_active_learning import run_active_learning  # noqa: E402
from activelearning.domain.instances import CategorySchema, Instance, Label  # noqa: E402

POOL_CAP, TEST_CAP = 20_000, 5_000
BUDGET, BATCH, INIT = 3_000, 100, 100
SEEDS = list(range(8))


def build():
    config = json.loads((_ROOT / "experiments/e0/config.json").read_text())
    rows, _ = load_rows_and_schema(config)
    pool_rows, test_rows = dedupe_and_split(rows)
    rng = random.Random(7)
    pool_rows = rng.sample(pool_rows, POOL_CAP)
    test_rows = rng.sample(test_rows, min(TEST_CAP, len(test_rows)))
    labels = {l for _, l in pool_rows} | {l for _, l in test_rows}
    schema = CategorySchema.from_raw(labels, include_rare=True)

    def mk(rws, p):
        out = []
        for i, (t, l) in enumerate(rws):
            g = schema.validate(l) or Label("_rare_")
            out.append(Instance(id=f"{p}-{i}", text=t, gold_label=g))
        return out

    return mk(pool_rows, "p"), mk(test_rows, "t"), schema


def main():
    pool, test, schema = build()
    out = _ROOT / "experiments/e1e4/results"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "sweeps.jsonl"
    done = set()
    if path.exists():
        for line in path.open():
            r = json.loads(line)
            done.add((r["exp"], r["strategy"], r["noise"], r["batch"], r["seed"]))
    fh = path.open("a", encoding="utf-8")

    base_path = out / "baseline.json"
    if base_path.exists():
        baseline = json.loads(base_path.read_text())["macro_f1"]
    else:
        clf = PVBinClassifier().fit(
            [i.text for i in pool], [i.gold_label.value for i in pool]
        )
        baseline = clf.score_macro_f1(
            [i.text for i in test], [i.gold_label.value for i in test]
        )
        base_path.write_text(json.dumps({"macro_f1": baseline, "n_pool": len(pool)}))
    print(f"baseline supervisão completa (pool {len(pool)}): MacroF1={baseline:.4f}", flush=True)

    def one(exp, strategy, noise, batch, seed):
        if (exp, strategy, noise, batch, seed) in done:
            return
        t0 = time.time()
        r = run_active_learning(
            pool, test, schema, PVBinClassifier,
            SimulatedOracle(noise=noise, seed=seed),
            strategy=strategy, budget=BUDGET, batch_size=batch,
            initial_size=INIT, seed=seed, baseline_performance=baseline,
        )
        rec = {
            "exp": exp, "strategy": strategy, "noise": noise, "batch": batch,
            "seed": seed, "lce": round(r.lce_macro_f1, 4),
            "final_macro_f1": round(r.final_macro_f1, 4),
            "final_accuracy": round(r.final_accuracy, 4),
            "curve": [[n, f] for n, f in zip(r.curve_macro_f1.l_sizes, r.curve_macro_f1.scores)],
            "elapsed_s": round(time.time() - t0, 1),
        }
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        print({k: rec[k] for k in ("exp", "strategy", "noise", "batch", "seed",
                                    "lce", "final_macro_f1", "elapsed_s")}, flush=True)

    for strategy in ("entropy", "least_confidence", "smallest_margin", "random", "hybrid"):
        for seed in SEEDS:
            one("e1", strategy, 0.0, BATCH, seed)
    for batch in (50, 200):
        for seed in SEEDS:
            one("e1b", "entropy", 0.0, batch, seed)
    for noise in (0.1, 0.2, 0.4):
        for strategy in ("entropy", "random"):
            for seed in SEEDS:
                one("e4", strategy, noise, BATCH, seed)
    fh.close()
    print("SWEEPS CONCLUÍDOS", flush=True)


if __name__ == "__main__":
    main()
