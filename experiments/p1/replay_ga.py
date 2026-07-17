"""AG-replay — envelope de desempenho por otimização evolutiva de L0 (reduzido).

Verificação independente do MECANISMO do P1-AG: um algoritmo genético sobre a
composição de L0 encontra conjuntos muito acima da média aleatória (envelope
superior). Escala reduzida com racional na decisão D-002; protocolo
anticircularidade da correção A3: aptidão medida em partição de AFERIÇÃO,
indivíduo final REAVALIADO no teste intocado.

Configuração: tamanhos {50, 500} × cenários {max_acc, max_f1};
N_pop=30, 40 gerações, torneio k=3, cruzamento 1 ponto p_c=0.8 com reparo de
unicidade, mutação p_m=0.1 (1% dos genes), elitismo 10%.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from activelearning.adapters.classifiers.pvbin import PVBinClassifier  # noqa: E402
from replay_l0_sensitivity import dedupe_and_split  # noqa: E402

from activelearning.adapters.datasets.retail_csv import load_rows_and_schema  # noqa: E402

POP, GENS, TOURN, PC, PM, ELITE_FRAC = 30, 40, 3, 0.8, 0.1, 0.1
SIZES = [50, 500]
SCENARIOS = ["max_acc", "max_f1"]
FITNESS_SET = 5000
SEED = 42


def evaluate(pool, idx, texts_eval, gold_eval, metric):
    clf = PVBinClassifier().fit([pool[i][0] for i in idx], [pool[i][1] for i in idx])
    if metric == "max_acc":
        return clf.score_accuracy(texts_eval, gold_eval)
    return clf.score_macro_f1(texts_eval, gold_eval)


def main() -> None:
    config = json.loads((_ROOT / "experiments/e0/config.json").read_text())
    rows, _ = load_rows_and_schema(config)
    pool, test = dedupe_and_split(rows)
    rng = random.Random(SEED)

    # partição de aferição disjunta do teste (sai do pool; não volta para sorteio)
    fit_idx = rng.sample(range(len(pool)), FITNESS_SET)
    fit_texts = [pool[i][0] for i in fit_idx]
    fit_gold = [pool[i][1] for i in fit_idx]
    fit_set = set(fit_idx)
    candidates = [i for i in range(len(pool)) if i not in fit_set]
    test_texts = [t for t, _ in test]
    test_gold = [l for _, l in test]

    out = _ROOT / "experiments/p1/results"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "replay_ga.jsonl"
    fh = path.open("a", encoding="utf-8")

    for size in SIZES:
        for scenario in SCENARIOS:
            t0 = time.time()
            grng = random.Random(1000 + size)
            pop = [grng.sample(candidates, size) for _ in range(POP)]
            fitness = [evaluate(pool, ind, fit_texts, fit_gold, scenario) for ind in pop]
            history = []
            n_elite = max(1, int(POP * ELITE_FRAC))
            for gen in range(GENS):
                order = sorted(range(POP), key=lambda i: -fitness[i])
                new_pop = [pop[i] for i in order[:n_elite]]
                while len(new_pop) < POP:
                    def pick():
                        contenders = grng.sample(range(POP), TOURN)
                        return pop[max(contenders, key=lambda i: fitness[i])]
                    a, b = pick(), pick()
                    if grng.random() < PC:
                        cut = grng.randrange(1, size)
                        child = a[:cut] + [g for g in b if g not in set(a[:cut])]
                        child = child[:size]
                        while len(child) < size:  # reparo de unicidade
                            g = grng.choice(candidates)
                            if g not in set(child):
                                child.append(g)
                    else:
                        child = list(a)
                    if grng.random() < PM:
                        m = max(1, size // 100)
                        pos = grng.sample(range(size), m)
                        used = set(child)
                        for p in pos:
                            g = grng.choice(candidates)
                            while g in used:
                                g = grng.choice(candidates)
                            used.discard(child[p]); used.add(g)
                            child[p] = g
                    new_pop.append(child)
                pop = new_pop
                fitness = [evaluate(pool, ind, fit_texts, fit_gold, scenario) for ind in pop]
                best = max(fitness)
                history.append(round(best, 4))
                if gen % 10 == 0:
                    print(f"[{scenario} I={size}] ger {gen}: melhor aptidão {best:.4f}", flush=True)

            best_ind = pop[max(range(POP), key=lambda i: fitness[i])]
            # ANTICIRCULARIDADE: reavaliação no teste intocado
            test_value = evaluate(pool, best_ind, test_texts, test_gold, scenario)
            rec = {
                "scenario": scenario, "size": size,
                "fitness_best": max(fitness),
                "test_reeval": round(test_value, 4),
                "history": history,
                "elapsed_s": round(time.time() - t0, 1),
                "params": {"pop": POP, "gens": GENS, "pc": PC, "pm": PM,
                            "tourn": TOURN, "elite": n_elite, "fitness_set": FITNESS_SET},
            }
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            print("CONCLUÍDO:", {k: rec[k] for k in ("scenario", "size", "fitness_best", "test_reeval", "elapsed_s")}, flush=True)
    fh.close()


if __name__ == "__main__":
    main()
