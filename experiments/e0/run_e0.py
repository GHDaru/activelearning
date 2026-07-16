"""E0 — Avaliação de oráculos LLM com saída restrita por enum.

Uso:
    uv run python experiments/e0/run_e0.py --config experiments/e0/config.json

Requer as variáveis de ambiente das APIs configuradas (OPENAI_API_KEY,
GEMINI_API_KEY) e/ou um servidor Ollama local para os modelos locais.
O dataset é o mesmo do legado (CSV com colunas nm_item, nm_product).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from activelearning.application.evaluate_oracle import EvaluateOracle  # noqa: E402
from activelearning.domain.instances import CategorySchema, Instance, Label  # noqa: E402


def load_instances(config: dict) -> tuple[list[Instance], CategorySchema]:
    import csv
    import random

    data_cfg = config["dataset"]
    path = Path(data_cfg["csv_path"])
    text_col, label_col = data_cfg["text_column"], data_cfg["label_column"]
    min_per_class = int(data_cfg.get("min_samples_per_class", 5))

    rows: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            text = (row.get(text_col) or "").strip()
            label = (row.get(label_col) or "").strip()
            if text and label:
                rows.append((text, label))

    counts: dict[str, int] = {}
    for _, label in rows:
        counts[label] = counts.get(label, 0) + 1
    frequent = {label for label, n in counts.items() if n >= min_per_class}
    schema = CategorySchema.from_raw(frequent, include_rare=True)

    rng = random.Random(int(config.get("seed", 42)))
    sample_size = int(config["sample_size"])
    indices = rng.sample(range(len(rows)), min(sample_size, len(rows)))
    instances = []
    for i in indices:
        text, label = rows[i]
        gold = schema.validate(label) or Label("_rare_")
        instances.append(Instance(id=f"e0-{i}", text=text, gold_label=gold))
    return instances, schema


def build_oracle(spec: dict):
    kind = spec["provider"]
    if kind == "openai":
        from activelearning.adapters.oracles.openai_oracle import OpenAIOracle

        return OpenAIOracle(model=spec["model"], temperature=spec.get("temperature", 0.0))
    if kind == "gemini":
        from activelearning.adapters.oracles.gemini_oracle import GeminiOracle

        return GeminiOracle(model=spec["model"], temperature=spec.get("temperature", 0.0))
    if kind == "ollama":
        from activelearning.adapters.oracles.ollama_oracle import OllamaOracle

        return OllamaOracle(model=spec["model"], temperature=spec.get("temperature", 0.0))
    raise ValueError(f"Provider desconhecido: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))

    instances, schema = load_instances(config)
    output_dir = Path(config.get("output_dir", "experiments/e0/results"))
    print(f"E0: {len(instances)} instâncias, {len(schema)} categorias no schema (enum).")

    summaries = []
    for spec in config["oracles"]:
        print(f"\n=== {spec['provider']}:{spec['model']} ===")
        try:
            oracle = build_oracle(spec)
        except (RuntimeError, ImportError) as exc:
            print(f"  PULADO: {exc}")
            continue
        use_case = EvaluateOracle(oracle, output_dir)
        report = use_case.run(instances, schema, batch_size=int(config.get("batch_size", 25)))
        summary = report.to_summary()
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    summary_path = output_dir / "e0_summary.json"
    summary_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nResumo consolidado: {summary_path}")


if __name__ == "__main__":
    main()
