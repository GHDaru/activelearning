"""P1-replay — sensibilidade do desempenho à composição/tamanho de L0 (PVBin).

Reexecução independente do experimento da tese (original: 47 tamanhos × 30
repetições) em grade reduzida com racional registrado (decisão D-002 do
tesedaru): 15 tamanhos log-espaçados × 10 repetições.

Protocolo: base corrigida → deduplicação por texto normalizado → split
estratificado teste (20%, cap 20k por custo de avaliação) / pool; para cada
tamanho I e repetição r, treina PVBin em L0 ~ amostra aleatória (semente
100*r) do pool e mede Acurácia e Macro F1 no teste FIXO.

Saída: experiments/p1/results/replay_l0.jsonl (1 linha por execução).
Uso: uv run python experiments/p1/replay_l0_sensitivity.py
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from activelearning.adapters.classifiers.pvbin import PVBinClassifier  # noqa: E402
from activelearning.adapters.datasets.retail_csv import load_rows_and_schema  # noqa: E402
from activelearning.domain.instances import normalize_label  # noqa: E402

SIZES = [10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000,
         20_000, 50_000, 100_000, 150_000, 200_000]
N_REPS = 10
TEST_FRACTION = 0.2
TEST_CAP = 20_000
SEED = 42


def dedupe_and_split(rows):
    """Deduplica por texto normalizado e faz split estratificado teste/pool."""
    seen: dict[str, int] = {}
    unique_rows = []
    for text, label in rows:
        key = normalize_label(text)
        if key not in seen:
            seen[key] = 1
            unique_rows.append((text, label))
    by_class = defaultdict(list)
    for i, (_, label) in enumerate(unique_rows):
        by_class[label].append(i)
    rng = random.Random(SEED)
    test_idx: set[int] = set()
    for label, idxs in sorted(by_class.items()):
        k = max(1, round(len(idxs) * TEST_FRACTION)) if len(idxs) > 1 else 0
        test_idx.update(rng.sample(idxs, k))
    test_idx = set(rng.sample(sorted(test_idx), min(TEST_CAP, len(test_idx))))
    test = [unique_rows[i] for i in sorted(test_idx)]
    pool = [unique_rows[i] for i in range(len(unique_rows)) if i not in test_idx]
    return pool, test


def main() -> None:
    config = json.loads((_ROOT / "experiments/e0/config.json").read_text())
    rows, _ = load_rows_and_schema(config)
    pool, test = dedupe_and_split(rows)
    test_texts = [t for t, _ in test]
    test_gold = [l for _, l in test]
    print(f"pool={len(pool):,} teste={len(test):,} (dedup + estratificado, semente {SEED})")

    out = _ROOT / "experiments/p1/results"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "replay_l0.jsonl"
    done = set()
    if path.exists():  # retomável
        for line in path.open():
            r = json.loads(line)
            done.add((r["size"], r["rep"]))
    fh = path.open("a", encoding="utf-8")

    for size in SIZES:
        if size > len(pool):
            continue
        for rep in range(N_REPS):
            if (size, rep) in done:
                continue
            rng = random.Random(100 * rep + size)
            idx = rng.sample(range(len(pool)), size)
            texts = [pool[i][0] for i in idx]
            labels = [pool[i][1] for i in idx]
            t0 = time.time()
            clf = PVBinClassifier().fit(texts, labels)
            acc = clf.score_accuracy(test_texts, test_gold)
            f1 = clf.score_macro_f1(test_texts, test_gold)
            rec = {"size": size, "rep": rep, "accuracy": round(acc, 4),
                   "macro_f1": round(f1, 4), "n_classes_l0": len(set(labels)),
                   "elapsed_s": round(time.time() - t0, 1)}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(rec)
    fh.close()
    print("P1-replay concluído:", path)


if __name__ == "__main__":
    main()
