"""E0-P — ablação de prompt no modelo fraco (gpt-4o-mini, enum, b=10).

Variantes {v4a, v4b} sobre os MESMOS 500 primeiros itens da S-rand e da
S-strat oficiais; o v3 reutiliza as anotações oficiais existentes (mesmo
instrumento). Comparação por McNemar pareado (analyze_e0p.py).
Decisão de desenho anti-vazamento: D-004 (tesedaru/docs/decisoes.md).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "experiments/e0"))

from run_e0 import build_samples, load_dotenv_if_present, load_rows_and_schema  # noqa: E402

load_dotenv_if_present()

from activelearning.adapters.oracles.openai_oracle import OpenAIOracle  # noqa: E402
from activelearning.application.evaluate_oracle import EvaluateOracle  # noqa: E402

N = 500


def main():
    config = json.loads((_ROOT / "experiments/e0/config.json").read_text())
    rows, schema = load_rows_and_schema(config)
    samples = build_samples(config, rows, schema)
    out_root = _ROOT / "experiments/e0p/results"
    for variant in ("v4a", "v4b"):
        for sample_name in ("rand", "strat"):
            instances = samples[sample_name][:N]
            oracle = OpenAIOracle(model="gpt-4o-mini", mode="enum",
                                  items_per_call=10, prompt_variant=variant)
            out = out_root / sample_name
            out.mkdir(parents=True, exist_ok=True)
            print(f"=== {variant} {sample_name} n={len(instances)} ===", flush=True)
            report = EvaluateOracle(oracle=oracle, output_dir=out).run(instances, schema)
            print(json.dumps(report.to_summary(), ensure_ascii=False)[:400], flush=True)


if __name__ == "__main__":
    main()
