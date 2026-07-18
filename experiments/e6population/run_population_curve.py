"""E6 — curvas interna (autoavaliação) × externa (população) com oráculo perfeito.

Desenho do autor (17/07/2026): divide-se a base deduplicada em POOL (x%, aqui
50.000 — candidatos a rotulagem pelo oráculo perfeito = gabarito) e POPULAÇÃO
(todo o restante, reservado). A cada passo:
  1. o classificador corrente seleciona um lote do pool (entropia) e o
     "rotula" (gold);
  2. o conjunto rotulado acumulado é dividido de forma convencional em
     treino/teste interno (80/20, estratificado quando possível, semente fixa);
  3. treina-se no treino interno; mede-se acurácia e Macro-F1 no TESTE
     INTERNO (o que o classificador "enxerga" de si) e na POPULAÇÃO
     (desempenho real de implantação) com o MESMO modelo.

A distância entre as curvas quantifica o viés de autoavaliação sob amostragem
ativa (o teste interno herda a distribuição enviesada da seleção por
incerteza) — fenômeno citado no Cap. 3 da tese; aqui, medido.

Uso: python run_population_curve.py [--classifier pvbin|sgd|both]
     [--budget 50000] [--batch 500] [--smoke]
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

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from activelearning.adapters.classifiers.pvbin import PVBinClassifier  # noqa: E402
from activelearning.adapters.classifiers.sgd_text import SgdTextClassifier  # noqa: E402
from activelearning.adapters.strategies.drisl import TfidfSvdEncoder, drisl_select  # noqa: E402
from activelearning.domain.instances import normalize_label  # noqa: E402


class PrecomputedEncoder:
    """Cacheia embeddings por texto — o DRI-SL roda a cada lote sem re-encodar."""

    def __init__(self, texts):
        self._base = TfidfSvdEncoder()
        emb = self._base(texts)
        self._cache = {t: emb[i] for i, t in enumerate(texts)}

    def __call__(self, texts):
        return np.vstack([self._cache[t] for t in texts])

CLASSIFIERS = {"pvbin": PVBinClassifier, "sgd": SgdTextClassifier}
SEED = 42


def load_base(min_per_class=2):
    rows = []
    with (_ROOT / "data/dataset.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append((r["nm_item"], normalize_label(r["nm_product"])))
    counts = Counter(l for _, l in rows)
    rows = [(t, l) for t, l in rows if counts[l] >= min_per_class]
    seen, dedup = set(), []
    for t, l in rows:
        k = t.strip().lower()
        if k not in seen:
            seen.add(k)
            dedup.append((t, l))
    rng = random.Random(SEED)
    rng.shuffle(dedup)
    return dedup


def internal_split(texts, labels, seed=SEED):
    """80/20 estratificado quando possível; degrada p/ aleatório se preciso."""
    counts = Counter(labels)
    strat = labels if min(counts.values()) >= 2 else None
    try:
        return train_test_split(texts, labels, test_size=0.2,
                                stratify=strat, random_state=seed)
    except ValueError:
        return train_test_split(texts, labels, test_size=0.2, random_state=seed)


def run(classifier_name: str, pool, population, budget: int, batch: int,
        out_dir: Path, strategy: str = "entropy", encoder=None):
    factory = CLASSIFIERS[classifier_name]
    pool_texts = [t for t, _ in pool]
    pool_labels = [l for _, l in pool]
    pop_texts = [t for t, _ in population]
    pop_labels = [l for _, l in population]

    rng = np.random.default_rng(SEED)
    out_path = out_dir / f"popcurve_{classifier_name}_{strategy}.jsonl"
    state_path = out_dir / f"popcurve_{classifier_name}_{strategy}_state.json"
    if state_path.exists():  # retomada pós-reinício: recarrega a trajetória
        labeled_idx = json.loads(state_path.read_text())["labeled_idx"]
        print(f"[{classifier_name}/{strategy}] RETOMANDO em |L|={len(labeled_idx)}",
              flush=True)
    else:
        labeled_idx = [int(x) for x in rng.choice(len(pool), size=batch, replace=False)]
        out_path.write_text("")
    labeled_set = set(labeled_idx)
    unlabeled = [i for i in range(len(pool)) if i not in labeled_set]

    curve = []
    t_start = time.time()
    while True:
        lx = [pool_texts[i] for i in labeled_idx]
        ly = [pool_labels[i] for i in labeled_idx]
        tr_x, te_x, tr_y, te_y = internal_split(lx, ly)
        clf = factory()
        clf.fit(tr_x, tr_y)
        pred_int = clf.predict(te_x)
        pred_ext = []
        for i in range(0, len(pop_texts), 20000):  # blocos: PVBin gera escores densos
            pred_ext.extend(clf.predict(pop_texts[i:i + 20000]))
        point = {
            "n_labels": len(labeled_idx),
            "acc_int": round(accuracy_score(te_y, pred_int), 4),
            "f1_int": round(f1_score(te_y, pred_int, average="macro"), 4),
            "acc_ext": round(accuracy_score(pop_labels, pred_ext), 4),
            "f1_ext": round(f1_score(pop_labels, pred_ext, average="macro"), 4),
            "elapsed_s": round(time.time() - t_start, 1),
        }
        curve.append(point)
        with out_path.open("a") as fh:
            fh.write(json.dumps(point) + "\n")
        print(f"[{classifier_name}/{strategy}] |L|={point['n_labels']} "
              f"int acc/F1={point['acc_int']}/{point['f1_int']} "
              f"ext acc/F1={point['acc_ext']}/{point['f1_ext']}", flush=True)
        if len(labeled_idx) >= budget or not unlabeled:
            break
        take = min(batch, budget - len(labeled_idx), len(unlabeled))
        if strategy == "entropy":
            proba = clf.predict_proba([pool_texts[i] for i in unlabeled])
            with np.errstate(divide="ignore", invalid="ignore"):
                ent = -np.nansum(proba * np.log(np.clip(proba, 1e-12, 1)), axis=1)
            order = np.argsort(-ent)[:take]
            chosen = [unlabeled[j] for j in order]
        elif strategy == "random":
            chosen = list(rng.choice(unlabeled, size=take, replace=False))
        elif strategy == "drisl":
            texts_u = [pool_texts[i] for i in unlabeled]
            sel = drisl_select(texts_u, take, encoder, seed=SEED)
            chosen = [unlabeled[j] for j in sel.indices]
        elif strategy == "drisl-c":
            # variante guiada pelo classificador: grupos = classes previstas
            from activelearning.adapters.strategies.drisl import drisl_select_by_groups
            texts_u = [pool_texts[i] for i in unlabeled]
            pred_u = []
            for k in range(0, len(texts_u), 20000):
                pred_u.extend(clf.predict(texts_u[k:k + 20000]))
            sel = drisl_select_by_groups(texts_u, take, pred_u)
            chosen = [unlabeled[j] for j in sel.indices]
        else:
            raise ValueError(strategy)
        chosen_set = set(chosen)
        labeled_idx.extend(int(c) for c in chosen)
        unlabeled = [i for i in unlabeled if i not in chosen_set]
        state_path.write_text(json.dumps({"labeled_idx": labeled_idx}))

    summary = {"classifier": classifier_name, "strategy": strategy,
               "budget": budget, "batch": batch,
               "pool": len(pool), "population": len(population),
               "final": curve[-1], "wall_seconds": round(time.time() - t_start, 1)}
    (out_dir / f"popcurve_{classifier_name}_{strategy}_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier", choices=["pvbin", "sgd", "both"], default="both")
    ap.add_argument("--budget", type=int, default=50000)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--pool-size", type=int, default=50000)
    ap.add_argument("--strategy", choices=["entropy", "random", "drisl", "drisl-c"],
                    default="entropy")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.budget, args.batch, args.pool_size = 2000, 500, 8000

    dedup = load_base()
    pool = dedup[: args.pool_size]
    population = dedup[args.pool_size:]
    print(f"pool={len(pool)} população={len(population)} "
          f"classes={len(set(l for _, l in dedup))}", flush=True)
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    names = ["pvbin", "sgd"] if args.classifier == "both" else [args.classifier]
    encoder = None
    if args.strategy == "drisl":
        print("pré-computando embeddings do pool p/ DRI-SL...", flush=True)
        encoder = PrecomputedEncoder([t for t, _ in pool])
    for n in names:
        run(n, pool, population, args.budget, args.batch, out_dir,
            strategy=args.strategy, encoder=encoder)


if __name__ == "__main__":
    main()
