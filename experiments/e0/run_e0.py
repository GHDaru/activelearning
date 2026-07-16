"""E0 — Avaliação de oráculos LLM (desenho fatorial; ver specs/001).

Amostras pareadas (todos os modelos rotulam as MESMAS instâncias):
- S-rand : aleatória simples (acurácia de produção, custo, McNemar, RQ4)
- S-strat: estratificada k por classe (macro-F1 e confusão confiáveis)

Uso:
    uv run python experiments/e0/run_e0.py --config experiments/e0/config.json

Chaves de API via .env na raiz (OPENAI_API_KEY, GEMINI_API_KEY, MAAS_API_KEY);
providers sem chave são pulados. Execução retomável.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))


def load_dotenv_if_present() -> None:
    """Carrega variáveis de um .env na raiz do projeto (não versionado)."""
    env_path = _ROOT / ".env"
    if not env_path.exists():
        return
    import os

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv_if_present()

from activelearning.application.evaluate_oracle import EvaluateOracle  # noqa: E402
from activelearning.domain.instances import CategorySchema, Instance, Label  # noqa: E402


def load_rows_and_schema(config: dict):
    import csv

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
    return rows, schema


def _to_instance(rows, i: int, schema: CategorySchema) -> Instance:
    text, label = rows[i]
    gold = schema.validate(label) or Label("_rare_")
    return Instance(id=f"e0-{i}", text=text, gold_label=gold)


def build_samples(config: dict, rows, schema: CategorySchema) -> dict[str, list[Instance]]:
    """S-rand (aleatória simples) e S-strat (k por classe), ambas com seed fixa."""
    import random

    seed = int(config.get("seed", 42))
    samples: dict[str, list[Instance]] = {}

    rand_size = int(config["samples"]["random_size"])
    rng = random.Random(seed)
    rand_indices = rng.sample(range(len(rows)), min(rand_size, len(rows)))
    samples["rand"] = [_to_instance(rows, i, schema) for i in rand_indices]

    k = int(config["samples"]["stratified_per_class"])
    by_class: dict[str, list[int]] = {}
    for i, (_, label) in enumerate(rows):
        normalized = schema.validate(label)
        key = normalized.value if normalized else "_rare_"
        by_class.setdefault(key, []).append(i)
    rng = random.Random(seed)
    strat_indices: list[int] = []
    for key in sorted(by_class):
        pool = by_class[key]
        strat_indices.extend(rng.sample(pool, min(k, len(pool))))
    samples["strat"] = [_to_instance(rows, i, schema) for i in strat_indices]
    return samples


def build_oracle(spec: dict):
    kind = spec["provider"]
    constrained = bool(spec.get("constrained", True))
    temperature = float(spec.get("temperature", 0.0))
    pricing = spec.get("pricing_usd_per_mtok")
    pricing = tuple(pricing) if pricing else None

    if kind == "openai":
        from activelearning.adapters.oracles.openai_oracle import OpenAIOracle

        return OpenAIOracle(
            model=spec["model"], temperature=temperature, constrained=constrained
        )
    if kind == "huawei-maas":
        from activelearning.adapters.oracles.openai_compatible import HuaweiMaasOracle

        return HuaweiMaasOracle(
            model=spec["model"],
            base_url=spec.get("base_url"),
            temperature=temperature,
            constrained=constrained,
            pricing_usd_per_mtok=pricing,
            disable_thinking=bool(spec.get("disable_thinking", True)),
        )
    if kind == "openai-compatible":
        from activelearning.adapters.oracles.openai_compatible import OpenAICompatibleOracle

        return OpenAICompatibleOracle(
            model=spec["model"],
            provider_name=spec.get("name", "openai-compatible"),
            base_url=spec["base_url"],
            temperature=temperature,
            api_key_env=spec.get("api_key_env", "OPENAI_API_KEY"),
            constrained=constrained,
            pricing_usd_per_mtok=pricing,
            use_prompt_cache_key=False,
        )
    if kind == "gemini":
        from activelearning.adapters.oracles.gemini_oracle import GeminiOracle

        return GeminiOracle(model=spec["model"], temperature=temperature)
    if kind == "ollama":
        from activelearning.adapters.oracles.ollama_oracle import OllamaOracle

        return OllamaOracle(model=spec["model"], temperature=temperature)
    raise ValueError(f"Provider desconhecido: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))

    rows, schema = load_rows_and_schema(config)
    samples = build_samples(config, rows, schema)
    output_root = Path(config.get("output_dir", "experiments/e0/results"))
    print(
        f"E0: schema com {len(schema)} categorias | "
        f"S-rand={len(samples['rand'])} | S-strat={len(samples['strat'])}"
    )

    summaries = []
    for spec in config["oracles"]:
        mode = "enum" if spec.get("constrained", True) else "free"
        wanted = spec.get("samples", ["rand", "strat"])
        print(f"\n=== {spec['provider']}:{spec['model']} [{mode}] amostras={wanted} ===")
        try:
            oracle = build_oracle(spec)
        except (RuntimeError, ImportError) as exc:
            print(f"  PULADO: {exc}")
            continue
        for sample_name in wanted:
            instances = samples[sample_name]
            use_case = EvaluateOracle(oracle, output_root / sample_name)
            report = use_case.run(
                instances, schema, batch_size=int(config.get("batch_size", 25))
            )
            summary = {"sample": sample_name, **report.to_summary()}
            summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2))

    summary_path = output_root / "e0_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nResumo consolidado: {summary_path}")
    print("Análise pareada (McNemar/Wilson): uv run python experiments/e0/analyze_e0.py")


if __name__ == "__main__":
    main()
