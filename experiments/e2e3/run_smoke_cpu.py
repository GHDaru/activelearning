"""Smoke test do BERTimbau em CPU (pré-execução do E2/E3 na estação GPU).

Objetivo: validar a cadeia completa (download do modelo, tokenização, ajuste
fino, predição, Macro-F1) num subconjunto minúsculo e MEDIR o custo de parede
em CPU, para extrapolar a viabilidade local vs. GPU. Não produz números para a
tese — apenas verificação de instrumental (registrado no diário/checklist H).

Uso: python experiments/e2e3/run_smoke_cpu.py [--classes 20] [--per-class 15]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

from activelearning.adapters.classifiers.bertimbau import BertimbauClassifier  # noqa: E402


def load_subset(csv_path: Path, n_classes: int, per_class: int, per_class_eval: int, seed: int):
    rows = []
    with csv_path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append((r["nm_item"], r["nm_product"]))
    rng = random.Random(seed)
    rng.shuffle(rows)
    counts = Counter(l for _, l in rows)
    # classes mais frequentes -> garante exemplos suficientes p/ treino+eval
    top = [c for c, _ in counts.most_common(n_classes)]
    by_class = defaultdict(list)
    for t, l in rows:
        if l in top and len(by_class[l]) < per_class + per_class_eval:
            by_class[l].append(t)
    train_x, train_y, eval_x, eval_y = [], [], [], []
    for c in top:
        docs = by_class[c]
        train_x += docs[:per_class]
        train_y += [c] * len(docs[:per_class])
        eval_x += docs[per_class : per_class + per_class_eval]
        eval_y += [c] * len(docs[per_class : per_class + per_class_eval])
    return train_x, train_y, eval_x, eval_y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", type=int, default=20)
    ap.add_argument("--per-class", type=int, default=15)
    ap.add_argument("--per-class-eval", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    train_x, train_y, eval_x, eval_y = load_subset(
        _ROOT / "data/dataset.csv", args.classes, args.per_class,
        args.per_class_eval, args.seed,
    )
    print(f"subconjunto: {len(train_x)} treino / {len(eval_x)} aval / "
          f"{args.classes} classes  [{time.time()-t0:.1f}s]", flush=True)

    clf = BertimbauClassifier(
        epochs=args.epochs, batch_size=args.batch_size,
        max_length=args.max_length, seed=args.seed, progress=True,
    )
    t1 = time.time()
    clf.fit(train_x, train_y)
    t_fit = time.time() - t1
    t2 = time.time()
    pred = clf.predict(eval_x)
    t_pred = time.time() - t2

    report = {
        "n_train": len(train_x), "n_eval": len(eval_x),
        "n_classes": args.classes, "epochs": args.epochs,
        "batch_size": args.batch_size, "max_length": args.max_length,
        "seed": args.seed,
        "accuracy": round(accuracy_score(eval_y, pred), 4),
        "macro_f1": round(f1_score(eval_y, pred, average="macro"), 4),
        "fit_seconds": round(t_fit, 1),
        "predict_seconds": round(t_pred, 1),
        "device": "cpu",
    }
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "smoke_cpu.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
