"""E3' — validação do FALCO com o classificador forte (BERTimbau) fora do laço.

Desenho aprovado em 18/07/2026 (corte do E3 original): quem roda o laço é o
classificador leve; o BERTimbau é treinado UMA vez por braço, sobre conjuntos
já coletados. A régua de "supervisão completa" é o POOL de 50.000 (o mesmo do
E6 e do ciclo real E5 — mesma semente, mesmo embaralhamento), porque o
aprendizado ativo só enxerga o pool: compará-lo com a base inteira confundiria
valor da seleção com valor de ter mais dados.

Braços (cada um = um ajuste fino):
  A  itens anotados pelo pipeline real (ciclos E5 com oráculo NIM), rótulos do
     ORÁCULO — o que o pipeline completo entrega ao modelo forte;
  B  mesmos itens de A, rótulos GOLD — isola o custo do ruído do oráculo (A−B);
  C  |A| itens aleatórios do pool, GOLD — isola o valor da seleção (B−C);
  D  pool inteiro (50k), GOLD — a régua (teto do pool);
  E  prefixo de 15k da trajetória de entropia do E6 (SGD), GOLD — orçamento
     maior ajudaria?

Hipótese: F1(A) >= 0,95 x F1(D) com |A|/50k dos rótulos.

Avaliação: POPULAÇÃO RESERVADA = dedup[54000:] (exclui pool e os val/teste de
2k+2k usados pelo ciclo real nas decisões de parada). Amostra estratificada
fixa entre braços (--eval-limit, 0 = população inteira); acurácia com IC de
Wilson 95%; Macro F1; predições salvas por braço para pareamento (McNemar).

Retomada: braço com results/e3prime_<braço>.json existente é pulado.

Uso:
  GPU : python experiments/e2e3/run_e3prime.py --arms A,B,C,D,E \
            --batch-size 128 --eval-limit 0
  CPU : python experiments/e2e3/run_e3prime.py --arms A,B,C \
            --batch-size 16 --eval-limit 20000
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

from activelearning.adapters.classifiers.bertimbau import BertimbauClassifier  # noqa: E402
from activelearning.domain.instances import normalize_label  # noqa: E402

SEED = 42
POOL_SIZE = 50_000
CYCLE_HOLDOUT = 4_000          # val 2k + teste 2k do ciclo real (dedup[50000:54000])
CACHE = _ROOT / "experiments/e5cycle/results/annotation_cache_nemotron.jsonl"
E6_ENTROPY_STATE = _ROOT / "experiments/e6population/results/popcurve_sgd_entropy_state.json"
OUT = Path(__file__).parent / "results"


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(center - half, 4), round(center + half, 4))


def load_base(min_per_class: int = 2):
    """Idêntico ao E6/E5: filtro por classe, dedup por texto, shuffle semente 42."""
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


def eval_sample(population, limit: int):
    """Amostra estratificada determinística (mesma para todos os braços)."""
    if not limit or limit >= len(population):
        return list(range(len(population)))
    by_class = defaultdict(list)
    for i, (_, l) in enumerate(population):
        by_class[l].append(i)
    rng = random.Random(SEED)
    frac = limit / len(population)
    chosen = []
    for l in sorted(by_class):
        idx = by_class[l]
        k = max(1, round(len(idx) * frac))
        chosen.extend(rng.sample(idx, min(k, len(idx))))
    return sorted(chosen)


def build_arms(pool, valid_labels, arm_names):
    """Retorna {braço: (texts, labels, meta)} apenas dos braços pedidos."""
    arms = {}
    need_cache = {"A", "B", "C"} & set(arm_names)
    if need_cache:
        oracle_by_idx = {}
        with CACHE.open(encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                i = int(d["instance_id"].split("-")[1])
                # label None = resposta inválida do oráculo, cacheada como tal
                oracle_by_idx[i] = (normalize_label(d["label"])
                                    if d.get("label") else None)
        valid = {i: lab for i, lab in oracle_by_idx.items() if lab in valid_labels}
        idx_sorted = sorted(valid)
        meta_a = {"n_cache": len(oracle_by_idx), "n_validos": len(valid)}
        if "A" in arm_names:
            arms["A"] = ([pool[i][0] for i in idx_sorted],
                         [valid[i] for i in idx_sorted],
                         {**meta_a, "fonte": "rótulos do oráculo real (NIM)"})
        if "B" in arm_names:
            arms["B"] = ([pool[i][0] for i in idx_sorted],
                         [pool[i][1] for i in idx_sorted],
                         {**meta_a, "fonte": "mesmos itens de A, rótulos gold"})
        if "C" in arm_names:
            rng = random.Random(SEED)
            ridx = sorted(rng.sample(range(len(pool)), len(idx_sorted)))
            arms["C"] = ([pool[i][0] for i in ridx],
                         [pool[i][1] for i in ridx],
                         {"fonte": "aleatório do pool, gold, |A| itens"})
    if "D" in arm_names:
        arms["D"] = ([t for t, _ in pool], [l for _, l in pool],
                     {"fonte": "pool inteiro, gold (régua)"})
    traj = None
    for name in arm_names:
        if name == "E" or (name.startswith("E") and name[1:].isdigit()):
            if traj is None:
                traj = json.loads(E6_ENTROPY_STATE.read_text())["labeled_idx"]
            k = 15_000 if name == "E" else int(name[1:]) * 1_000
            prefix = traj[:k]
            arms[name] = ([pool[i][0] for i in prefix],
                          [pool[i][1] for i in prefix],
                          {"fonte": f"prefixo {k} da trajetória de entropia do E6 (SGD), gold"})
    return arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A,B,C,D,E")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=32)
    ap.add_argument("--eval-limit", type=int, default=20_000,
                    help="tamanho da amostra de avaliação na população (0 = inteira)")
    ap.add_argument("--force", action="store_true", help="reexecuta braços já concluídos")
    args = ap.parse_args()
    arm_names = [a.strip().upper() for a in args.arms.split(",") if a.strip()]
    # braços E<k> = prefixo de k mil rótulos da trajetória de entropia
    # (varredura de orçamento do E3' corrigido: onde F1 cruza 0,95*F1(D))

    dedup = load_base()
    pool = dedup[:POOL_SIZE]
    population = dedup[POOL_SIZE + CYCLE_HOLDOUT:]
    valid_labels = {l for _, l in dedup}
    print(f"pool={len(pool)} população={len(population)} classes={len(valid_labels)}",
          flush=True)

    sample_idx = eval_sample(population, args.eval_limit)
    ev_texts = [population[i][0] for i in sample_idx]
    ev_gold = [population[i][1] for i in sample_idx]
    print(f"avaliação: {len(sample_idx)} instâncias "
          f"({len(set(ev_gold))} classes na amostra)", flush=True)

    OUT.mkdir(exist_ok=True)
    arms = build_arms(pool, valid_labels, arm_names)
    for name in arm_names:
        res_path = OUT / f"e3prime_{name}.json"
        if res_path.exists() and not args.force:
            print(f"[{name}] já concluído — pulando (use --force p/ repetir)", flush=True)
            continue
        texts, labels, meta = arms[name]
        print(f"\n=== braço {name}: {meta['fonte']} · n={len(texts)} "
              f"classes={len(set(labels))} ===", flush=True)
        clf = BertimbauClassifier(epochs=args.epochs, batch_size=args.batch_size,
                                  max_length=args.max_length, seed=SEED, progress=True)
        t0 = time.time()
        clf.fit(texts, labels)
        fit_s = time.time() - t0
        t0 = time.time()
        pred = []
        for i in range(0, len(ev_texts), 5_000):
            pred.extend(clf.predict(ev_texts[i:i + 5_000]))
            print(f"  avaliação: {min(i + 5_000, len(ev_texts))}/{len(ev_texts)}",
                  flush=True)
        pred_s = time.time() - t0
        acc = accuracy_score(ev_gold, pred)
        result = {
            "arm": name, **meta, "n_train": len(texts),
            "n_train_classes": len(set(labels)),
            "epochs": args.epochs, "batch_size": args.batch_size,
            "max_length": args.max_length, "seed": SEED,
            "eval_n": len(ev_texts), "eval_limit": args.eval_limit,
            "accuracy": round(acc, 4),
            "accuracy_wilson95": wilson_ci(round(acc * len(ev_texts)), len(ev_texts)),
            "macro_f1": round(f1_score(ev_gold, pred, average="macro"), 4),
            "fit_seconds": round(fit_s, 1), "predict_seconds": round(pred_s, 1),
        }
        (OUT / f"e3prime_{name}_pred.json").write_text(json.dumps(
            {"sample_idx": sample_idx, "pred": pred}))
        res_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(json.dumps(result, ensure_ascii=False), flush=True)

    done = {p.stem.split("_")[1]: json.loads(p.read_text())
            for p in OUT.glob("e3prime_*.json")
            if p.stem.count("_") == 1 and not p.stem.endswith("pred")}
    if "A" in done and "D" in done:
        fa, fd = done["A"]["macro_f1"], done["D"]["macro_f1"]
        frac = done["A"]["n_train"] / done["D"]["n_train"]
        print(f"\nHIPÓTESE: F1(A)={fa} vs 0,95×F1(D)={0.95 * fd:.4f} "
              f"com {frac:.1%} dos rótulos -> "
              f"{'SUSTENTADA' if fa >= 0.95 * fd else 'NÃO sustentada'}", flush=True)
    if "D" in done:
        fd = done["D"]["macro_f1"]
        sweep = sorted((d["n_train"], d["macro_f1"], d["accuracy"], a)
                       for a, d in done.items()
                       if a.startswith("E") or a == "D")
        if len(sweep) > 1:
            print("VARREDURA (n, F1, acc, braço) vs critério "
                  f"F1>={0.95 * fd:.4f} / acc>={0.95 * done['D']['accuracy']:.4f}:",
                  flush=True)
            for n, f1, acc, a in sweep:
                ok_f1 = "OK" if f1 >= 0.95 * fd else "--"
                ok_ac = "OK" if acc >= 0.95 * done["D"]["accuracy"] else "--"
                print(f"  {a:>4} n={n:>6} F1={f1:.4f} [{ok_f1}] "
                      f"acc={acc:.4f} [{ok_ac}] ({n / done['D']['n_train']:.0%})",
                      flush=True)


if __name__ == "__main__":
    main()
