"""E5-ciclo — FALCO ponta a ponta com oráculo LLM real e gratuito (NVIDIA NIM).

Ciclo completo por classificador ∈ {pvbin, sgd}: DRI-SL (cold start sem
rótulos) → Fase 2 por entropia com transição por estagnação NA VALIDAÇÃO →
oráculo nemotron-3-ultra (custo monetário zero). Registra a curva INTERNA
(validação, usada nas decisões) e a EXTERNA (teste intocado, só relato).

Uso: python experiments/e5cycle/run_cycle.py [--classifier pvbin|sgd|both]
Parâmetros de desenho no bloco CONFIG.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "experiments/e0"))

from run_e0 import load_dotenv_if_present  # noqa: E402

load_dotenv_if_present()

from activelearning.adapters.classifiers.pvbin import PVBinClassifier  # noqa: E402
from activelearning.adapters.classifiers.sgd_text import SgdTextClassifier  # noqa: E402
from activelearning.adapters.oracles.factory import build_oracle  # noqa: E402
from activelearning.adapters.strategies.drisl import TfidfSvdEncoder, drisl_select  # noqa: E402
from activelearning.domain.instances import CategorySchema, Instance, normalize_label  # noqa: E402
from activelearning.application.run_falco import run_falco  # noqa: E402

CONFIG = {
    "seed": 42,
    "pool_size": 5000,
    "val_size": 1000,
    "test_size": 1000,
    "budget": 1000,
    "batch_size": 100,
    "min_per_class": 2,
    "oracle": {"provider": "nvidia", "model": "nvidia/nemotron-3-ultra-550b-a55b",
                "mode": "json-prompt", "items_per_call": 10},
}

CLASSIFIERS = {"pvbin": PVBinClassifier, "sgd": SgdTextClassifier}


def load_splits(cfg):
    rows = []
    with (_ROOT / "data/dataset.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append((r["nm_item"], normalize_label(r["nm_product"])))
    counts = Counter(l for _, l in rows)
    rows = [(t, l) for t, l in rows if counts[l] >= cfg["min_per_class"]]
    # dedup por texto (evita mesma descrição em pool e teste)
    seen, dedup = set(), []
    for t, l in rows:
        k = t.strip().lower()
        if k not in seen:
            seen.add(k)
            dedup.append((t, l))
    rng = random.Random(cfg["seed"])
    rng.shuffle(dedup)
    n_pool, n_val, n_test = cfg["pool_size"], cfg["val_size"], cfg["test_size"]
    def to_instances(part, prefix):
        from activelearning.domain.instances import Label
        return [Instance(id=f"{prefix}-{i}", text=t, gold_label=Label(l))
                for i, (t, l) in enumerate(part)]
    pool = to_instances(dedup[:n_pool], "pool")
    val = to_instances(dedup[n_pool:n_pool + n_val], "val")
    test = to_instances(dedup[n_pool + n_val:n_pool + n_val + n_test], "test")
    from activelearning.domain.instances import Label
    schema = CategorySchema([Label(l) for l in sorted({l for _, l in dedup})])
    return pool, val, test, schema


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier", choices=["pvbin", "sgd", "both"], default="both")
    ap.add_argument("--simulated", action="store_true",
                    help="smoke offline: troca o oráculo por SimulatedOracle(ruído 0,2)")
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--pool-size", type=int, default=None)
    ap.add_argument("--val-size", type=int, default=None)
    ap.add_argument("--test-size", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--items-per-call", type=int, default=None)
    ap.add_argument("--cache", type=str, default=None,
                    help="JSONL de cache de anotações compartilhado entre ciclos")
    ap.add_argument("--tag", type=str, default="",
                    help="sufixo dos arquivos de saída (ex.: _b30k)")
    args = ap.parse_args()
    cfg = dict(CONFIG)
    for k in ("budget", "pool_size", "val_size", "test_size", "batch_size"):
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    if args.items_per_call is not None:
        cfg["oracle"] = {**cfg["oracle"], "items_per_call": args.items_per_call}
    if args.simulated:
        cfg["oracle"] = {"provider": "simulated", "noise": 0.2, "seed": cfg["seed"]}
    pool, val, test, schema = load_splits(cfg)
    print(f"splits: pool={len(pool)} val={len(val)} test={len(test)} "
          f"classes(schema)={len(schema)}", flush=True)

    encoder = TfidfSvdEncoder()
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)

    names = ["pvbin", "sgd"] if args.classifier == "both" else [args.classifier]
    for name in names:
        factory = CLASSIFIERS[name]
        oracle = build_oracle(cfg["oracle"])
        if args.cache:
            from activelearning.adapters.oracles.cached import CachedOracle
            oracle = CachedOracle(oracle, Path(args.cache))
        print(f"\n=== ciclo FALCO · classificador={name} · oráculo={cfg['oracle'].get('model', cfg['oracle']['provider'])} ===",
              flush=True)
        t0 = time.time()
        res = run_falco(
            pool=pool, validation=val, test=test, schema=schema,
            classifier_factory=lambda: factory(),
            oracle_initial=oracle,
            drisl_selector=lambda texts, k: drisl_select(
                texts, k, encoder, seed=cfg["seed"]).indices,
            budget=cfg["budget"], batch_size=cfg["batch_size"], seed=cfg["seed"],
            output_path=out / f"cycle_{name}{args.tag}_records.jsonl",
        )
        payload = {
            "classifier": name, "config": cfg, **res.summary(),
            "wall_seconds": round(time.time() - t0, 1),
            "curve_test": list(zip(res.curve_macro_f1.l_sizes, res.curve_macro_f1.scores)),
            "curve_val": res.curve_val_macro_f1,
        }
        if args.cache and hasattr(oracle, "hits"):
            payload["oracle_cache"] = {"hits": oracle.hits, "calls_inner": oracle.calls_inner}
        (out / f"cycle_{name}{args.tag}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False))
        print(json.dumps({k: payload[k] for k in
                          ("classifier", "final_macro_f1", "lce_macro_f1",
                           "n_labeled", "invalid_labels", "phase_boundaries",
                           "wall_seconds")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
