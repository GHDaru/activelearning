"""Calibração rápida de lote no NVIDIA NIM: b=20 × b=50, pareado.

Mesmos 200 primeiros itens da S-rand oficial (gabarito conhecido); mede
acurácia, inválidos, latência por rótulo e McNemar exato entre os lotes.
Gera experiments/e5cycle/results/calibration_b20_b50.json.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "experiments/e0"))

from run_e0 import build_samples, load_dotenv_if_present, load_rows_and_schema  # noqa: E402

load_dotenv_if_present()

from activelearning.adapters.oracles.factory import build_oracle  # noqa: E402

N = 200


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def main():
    config = json.loads((_ROOT / "experiments/e0/config.json").read_text())
    rows, schema = load_rows_and_schema(config)
    instances = build_samples(config, rows, schema)["rand"][:N]
    gold = {i.id: i.gold_label.value for i in instances}

    results = {}
    for b in (20, 50):
        oracle = build_oracle({"provider": "nvidia",
                               "model": "nvidia/nemotron-3-ultra-550b-a55b",
                               "mode": "json-prompt", "items_per_call": b})
        t0 = time.time()
        anns = oracle.annotate(instances, schema)
        wall = time.time() - t0
        correct = {a.instance_id: (a.label is not None and a.label.value == gold[a.instance_id])
                   for a in anns}
        invalid = sum(1 for a in anns if a.label is None)
        results[f"b{b}"] = {
            "n": N,
            "accuracy": round(sum(correct.values()) / N, 4),
            "invalid": invalid,
            "wall_seconds": round(wall, 1),
            "seconds_per_label": round(wall / N, 2),
            "_correct": correct,
        }
        print(f"b={b}: acc={results[f'b{b}']['accuracy']} inválidos={invalid} "
              f"{results[f'b{b}']['seconds_per_label']}s/rótulo", flush=True)

    c20, c50 = results["b20"]["_correct"], results["b50"]["_correct"]
    only20 = sum(1 for k in gold if c20[k] and not c50[k])
    only50 = sum(1 for k in gold if not c20[k] and c50[k])
    for r in results.values():
        r.pop("_correct")
    out = {
        **results,
        "mcnemar": {"only_b20_correct": only20, "only_b50_correct": only50,
                    "p_value": round(mcnemar_exact(only20, only50), 4)},
        "speedup_b50_vs_b20": round(
            results["b20"]["seconds_per_label"] / results["b50"]["seconds_per_label"], 2),
    }
    path = Path(__file__).parent / "results/calibration_b20_b50.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
