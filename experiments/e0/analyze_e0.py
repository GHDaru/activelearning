"""Análise do E0: tabela consolidada com IC de Wilson + McNemar pareado.

Lê os JSONL de anotações em experiments/e0/results/{rand,strat}/ e produz:
- e0_table.json : por oráculo×amostra: acc + IC95, macro-F1, invalid, custo
- e0_mcnemar.json : p-valores par-a-par entre oráculos na S-rand (pareado)

Uso: uv run python experiments/e0/analyze_e0.py [--results experiments/e0/results]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from activelearning.domain.metrics import mcnemar_test, wilson_interval  # noqa: E402
from run_e0 import build_samples, load_dotenv_if_present, load_rows_and_schema  # noqa: E402

load_dotenv_if_present()


def correctness_by_oracle(sample_dir: Path, gold: dict[str, str]) -> dict[str, dict[str, bool]]:
    """{oracle_id: {instance_id: acertou?}} a partir dos JSONL da amostra."""
    result: dict[str, dict[str, bool]] = {}
    for path in sorted(sample_dir.glob("annotations_*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                record = json.loads(line)
                oracle_id = record["oracle_id"]
                iid = record["instance_id"]
                if iid not in gold:
                    continue
                result.setdefault(oracle_id, {})[iid] = record.get("label") == gold[iid]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=_ROOT / "experiments/e0/results")
    parser.add_argument("--config", type=Path, default=_ROOT / "experiments/e0/config.json")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows, schema = load_rows_and_schema(config)
    samples = build_samples(config, rows, schema)

    table = []
    for sample_name, instances in samples.items():
        gold = {i.id: i.gold_label.value for i in instances}
        sample_dir = args.results / sample_name
        if not sample_dir.exists():
            continue
        for report_path in sorted(sample_dir.glob("report_*.json")):
            summary = json.loads(report_path.read_text(encoding="utf-8"))
            low, high = wilson_interval(
                round(summary["accuracy"] * summary["n_total"]), summary["n_total"]
            )
            table.append(
                {
                    "sample": sample_name,
                    **summary,
                    "acc_wilson_low": round(low, 4),
                    "acc_wilson_high": round(high, 4),
                }
            )

    # McNemar pareado dentro de cada amostra (rand e strat).
    # Gold indexado pela base inteira (id = e0-{índice da linha filtrada}, estável):
    # anotações de execuções com amostragem anterior continuam pareáveis mesmo que
    # a amostra regenerada pela config atual seja outra.
    from activelearning.domain.instances import Label  # noqa: E402

    gold_all = {
        f"e0-{i}": (schema.validate(label) or Label("_rare_")).value
        for i, (_, label) in enumerate(rows)
    }
    mcnemar = []
    for sample_name in samples:
        sample_dir = args.results / sample_name
        if not sample_dir.exists():
            continue
        correctness = correctness_by_oracle(sample_dir, gold_all)
        for a, b in itertools.combinations(sorted(correctness), 2):
            common = set(correctness[a]) & set(correctness[b])
            if not common:
                continue
            n01 = sum(1 for i in common if correctness[a][i] and not correctness[b][i])
            n10 = sum(1 for i in common if not correctness[a][i] and correctness[b][i])
            mcnemar.append(
                {
                    "sample": sample_name,
                    "oracle_a": a,
                    "oracle_b": b,
                    "n_paired": len(common),
                    "a_right_b_wrong": n01,
                    "a_wrong_b_right": n10,
                    "p_value": round(mcnemar_test(n01, n10), 6),
                }
            )

    (args.results / "e0_table.json").write_text(
        json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.results / "e0_mcnemar.json").write_text(
        json.dumps(mcnemar, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(table, ensure_ascii=False, indent=2))
    print(f"\nMcNemar ({len(mcnemar)} pares) -> {args.results / 'e0_mcnemar.json'}")


if __name__ == "__main__":
    main()
