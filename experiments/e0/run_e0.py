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


# Implementações canônicas movidas para o pacote (compartilhadas com a API do
# FlowBuilder); reexportadas aqui para compatibilidade (analyze_e0 importa daqui).
from activelearning.adapters.datasets.retail_csv import (  # noqa: E402,F401
    build_samples,
    load_rows_and_schema,
    to_instance as _to_instance,
)
from activelearning.adapters.oracles.factory import build_oracle  # noqa: E402,F401


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
        mode = spec.get("mode", "enum")
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
