"""E0-P — análise pareada v3 × v4a × v4b (McNemar exato, mesmos 500 itens).

Gera experiments/e0p/results/analysis.json (artefato rastreável).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "experiments/e0"))

from run_e0 import build_samples, load_rows_and_schema  # noqa: E402

from activelearning.domain.instances import normalize_label  # noqa: E402

N = 500
V3_FILE = "annotations_openai_gpt-4o-mini_T0.0_b10.jsonl"
VAR_FILE = "annotations_openai_gpt-4o-mini_T0.0_b10#{v}.jsonl"


def load_preds(path: Path) -> dict[str, str]:
    preds = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            preds[r["instance_id"]] = normalize_label(r["label"]) if r["label"] else ""
    return preds


def mcnemar_exact(b: int, c: int) -> float:
    """McNemar binomial exato bicaudal sobre os pares discordantes."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n * 2
    return min(1.0, p)


def main():
    config = json.loads((_ROOT / "experiments/e0/config.json").read_text())
    rows, schema = load_rows_and_schema(config)
    samples = build_samples(config, rows, schema)
    out = {}
    for sample_name in ("rand", "strat"):
        instances = samples[sample_name][:N]
        gold = {i.id: i.gold_label.value for i in instances}
        ids = [i.id for i in instances]
        e0_dir = _ROOT / "experiments/e0/results" / sample_name
        e0p_dir = _ROOT / "experiments/e0p/results" / sample_name
        preds = {"v3": load_preds(e0_dir / V3_FILE)}
        for v in ("v4a", "v4b"):
            preds[v] = load_preds(e0p_dir / VAR_FILE.format(v=v))
        res = {}
        for v, p in preds.items():
            correct = [1 if p.get(i) == gold[i] else 0 for i in ids]
            res[v] = {"n": len(ids), "n_correct": sum(correct),
                      "accuracy": round(sum(correct) / len(ids), 4)}
        for a, bv in (("v3", "v4a"), ("v3", "v4b"), ("v4a", "v4b")):
            b = sum(1 for i in ids
                    if preds[a].get(i) == gold[i] and preds[bv].get(i) != gold[i])
            c = sum(1 for i in ids
                    if preds[a].get(i) != gold[i] and preds[bv].get(i) == gold[i])
            res[f"mcnemar_{a}_vs_{bv}"] = {
                "only_a_correct": b, "only_b_correct": c,
                "p_value": round(mcnemar_exact(b, c), 6)}
        out[sample_name] = res
    path = _ROOT / "experiments/e0p/results/analysis.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
