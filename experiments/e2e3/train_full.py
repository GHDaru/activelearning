"""Treino completo do BERTimbau na base inteira (teto supervisionado local).

Mede a capacidade preditiva com supervisão completa: ajuste fino sobre a
partição de treino e avaliação em teste estratificado intocado. Pensado para a
estação com GPU (RTX 3090); roda em CPU, mas ~30-60x mais lento.

Uso típico (GPU):
  python experiments/e2e3/train_full.py --epochs 3 --batch-size 64
Subconjunto rápido de validação da máquina (GPU, ~2 min):
  python experiments/e2e3/train_full.py --limit 20000 --epochs 1
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from activelearning.adapters.classifiers.bertimbau import BertimbauClassifier  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=_ROOT / "data/dataset.csv")
    ap.add_argument("--test-size", type=float, default=0.1,
                    help="fração de teste (estratificada quando possível)")
    ap.add_argument("--limit", type=int, default=0,
                    help="limita o total de linhas (0 = base inteira)")
    ap.add_argument("--min-per-class", type=int, default=2,
                    help="descarta classes com menos instâncias que isso "
                         "(estratificação exige >=2)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None,
                    help="cuda | cpu (padrão: autodetecção)")
    args = ap.parse_args()

    t0 = time.time()
    with args.csv.open(encoding="utf-8") as fh:
        rows = [(r["nm_item"], r["nm_product"]) for r in csv.DictReader(fh)]
    if args.limit:
        rows = rows[: args.limit]
    counts = Counter(l for _, l in rows)
    rows = [(t, l) for t, l in rows if counts[l] >= args.min_per_class]
    texts = [t for t, _ in rows]
    labels = [l for _, l in rows]
    tr_x, te_x, tr_y, te_y = train_test_split(
        texts, labels, test_size=args.test_size,
        stratify=labels, random_state=args.seed,
    )
    print(f"dados: {len(tr_x)} treino / {len(te_x)} teste / "
          f"{len(set(labels))} classes  [{time.time()-t0:.1f}s]", flush=True)

    clf = BertimbauClassifier(
        epochs=args.epochs, batch_size=args.batch_size,
        max_length=args.max_length, learning_rate=args.lr,
        seed=args.seed, device=args.device, progress=True,
    )
    t1 = time.time()
    clf.fit(tr_x, tr_y)
    t_fit = time.time() - t1
    t2 = time.time()
    pred = clf.predict(te_x)
    t_pred = time.time() - t2

    report = {
        "n_train": len(tr_x), "n_test": len(te_x),
        "n_classes": len(set(labels)),
        "epochs": args.epochs, "batch_size": args.batch_size,
        "max_length": args.max_length, "lr": args.lr, "seed": args.seed,
        "accuracy": round(accuracy_score(te_y, pred), 4),
        "macro_f1": round(f1_score(te_y, pred, average="macro"), 4),
        "fit_seconds": round(t_fit, 1),
        "predict_seconds": round(t_pred, 1),
        "device": args.device or "auto",
    }
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    tag = f"full_{len(tr_x)}tr_{args.epochs}ep_s{args.seed}"
    (out / f"{tag}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
