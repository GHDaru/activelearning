"""Estatística pareada do E3' sobre predições persistidas (McNemar + bootstrap).

Consome os artefatos e3prime_<braço>_s<semente>_pred.json gravados por
run_e3prime.py (mesma amostra de avaliação em todos os braços — o script
verifica) e o gabarito reconstruído por run_e3prime.load_base (dados fixos em
DATA_SEED=42). Produz dois artefatos no mesmo diretório:

  mcnemar_s<semente>.json      — McNemar pareado por par de braços: tabela de
      discordância (b = só o 1º acerta, c = só o 2º acerta), teste exato
      binomial bicaudal e qui-quadrado com correção de continuidade.
  bootstrap_f1_s<semente>.json — IC percentil 95% (bootstrap pareado, mesmas
      reamostragens para todos os braços) da diferença de Macro F1 por par,
      mais o IC de Macro F1 de cada braço envolvido.

Macro F1 segue a convenção do run_e3prime (sklearn): conjunto de rótulos de
cada braço fixado em união(gabarito, predições do braço) na amostra completa;
classe ausente numa reamostragem contribui F1=0 (zero_division=0). No ponto
(sem reamostragem) o valor reproduz exatamente o macro_f1 reportado em
e3prime_<braço>_s<semente>.json — o script aborta se não reproduzir.

Uso:
  uv run --with scikit-learn python experiments/e2e3/stats_e3prime.py \
      --seed 42 --pairs A-B,B-C,E35-D --n-boot 10000 --boot-seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_HERE))

from run_e3prime import CYCLE_HOLDOUT, DATA_SEED, POOL_SIZE, load_base  # noqa: E402


def mcnemar_exact(b: int, c: int) -> dict:
    """Teste de McNemar: exato binomial bicaudal + qui² com correção."""
    from scipy.stats import binomtest, chi2

    n_disc = b + c
    if n_disc == 0:
        return {"b": b, "c": c, "discordantes": 0, "p_exato": 1.0,
                "chi2_correcao": 0.0, "p_chi2": 1.0}
    p_exact = binomtest(min(b, c), n_disc, 0.5, alternative="two-sided").pvalue
    stat = (abs(b - c) - 1) ** 2 / n_disc
    return {
        "b": b, "c": c, "discordantes": n_disc,
        "p_exato": float(p_exact),
        "chi2_correcao": round(float(stat), 4),
        "p_chi2": float(chi2.sf(stat, df=1)),
    }


def macro_f1_from_codes(g: np.ndarray, p: np.ndarray, label_codes: np.ndarray,
                        n_classes: int) -> float:
    tp = np.bincount(g[g == p], minlength=n_classes).astype(np.float64)
    gc = np.bincount(g, minlength=n_classes).astype(np.float64)
    pc = np.bincount(p, minlength=n_classes).astype(np.float64)
    denom = gc + pc
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denom > 0, 2.0 * tp / denom, 0.0)
    return float(f1[label_codes].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42,
                    help="semente de TREINO dos artefatos a consumir (_s<semente>)")
    ap.add_argument("--pairs", default="A-B,B-C,E35-D",
                    help="pares X-Y; convenção: delta = métrica(X) − métrica(Y)")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--boot-seed", type=int, default=42,
                    help="semente do gerador das reamostragens (reprodutibilidade)")
    ap.add_argument("--out-dir", default=str(_HERE / "results"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    pairs = [tuple(p.split("-")) for p in args.pairs.split(",") if p.strip()]
    arms = sorted({a for pair in pairs for a in pair})

    dedup = load_base()
    population = dedup[POOL_SIZE + CYCLE_HOLDOUT:]

    preds, reported = {}, {}
    sample_idx = None
    for a in arms:
        d = json.loads((out_dir / f"e3prime_{a}_s{args.seed}_pred.json").read_text())
        if sample_idx is None:
            sample_idx = d["sample_idx"]
        elif d["sample_idx"] != sample_idx:
            raise SystemExit(f"sample_idx do braço {a} difere — braços não pareáveis")
        preds[a] = d["pred"]
        reported[a] = json.loads(
            (out_dir / f"e3prime_{a}_s{args.seed}.json").read_text())
    gold = [population[i][1] for i in sample_idx]
    n = len(gold)

    # codificação inteira num universo comum de classes
    classes = sorted(set(gold) | {l for p in preds.values() for l in p})
    code = {c: i for i, c in enumerate(classes)}
    k = len(classes)
    g = np.fromiter((code[x] for x in gold), dtype=np.int32, count=n)
    p_codes = {a: np.fromiter((code[x] for x in preds[a]), dtype=np.int32, count=n)
               for a in arms}
    # conjunto de rótulos por braço = união(gabarito, predições) — convenção sklearn
    label_codes = {a: np.array(sorted({code[x] for x in gold} |
                                      {code[x] for x in preds[a]}), dtype=np.int32)
                   for a in arms}

    # ponto: reproduzir exatamente o macro_f1 e a acurácia reportados
    point_f1, point_acc = {}, {}
    for a in arms:
        point_f1[a] = macro_f1_from_codes(g, p_codes[a], label_codes[a], k)
        point_acc[a] = float((g == p_codes[a]).mean())
        if round(point_f1[a], 4) != reported[a]["macro_f1"]:
            raise SystemExit(f"macro F1 recomputado do braço {a} "
                             f"({point_f1[a]:.4f}) difere do reportado "
                             f"({reported[a]['macro_f1']}) — investigar antes de seguir")
        if round(point_acc[a], 4) != reported[a]["accuracy"]:
            raise SystemExit(f"acurácia recomputada do braço {a} difere do reportado")
    print(f"ponto reproduzido nos braços {arms} (n={n}, classes={k})", flush=True)

    # ---------- McNemar ----------
    mcnemar = {"seed": args.seed, "data_seed": DATA_SEED, "eval_n": n,
               "fonte_predicoes": f"e3prime_<braço>_s{args.seed}_pred.json",
               "definicao": "b = só X acerta, c = só Y acerta (par X-Y); "
                            "p_exato = binomial bicaudal em min(b,c)~Bin(b+c, 0,5); "
                            "chi2 com correção de continuidade",
               "pares": {}}
    for x, y in pairs:
        cx, cy = g == p_codes[x], g == p_codes[y]
        b = int(np.sum(cx & ~cy))
        c = int(np.sum(~cx & cy))
        res = mcnemar_exact(b, c)
        mcnemar["pares"][f"{x}-{y}"] = {
            "acuracia": {x: round(point_acc[x], 4), y: round(point_acc[y], 4)},
            "delta_acuracia": round(point_acc[x] - point_acc[y], 4),
            "concordantes_ambos_acertam": int(np.sum(cx & cy)),
            "concordantes_ambos_erram": int(np.sum(~cx & ~cy)),
            **res,
        }
    (out_dir / f"mcnemar_s{args.seed}.json").write_text(
        json.dumps(mcnemar, indent=2, ensure_ascii=False) + "\n")
    print(f"mcnemar_s{args.seed}.json gravado", flush=True)

    # ---------- bootstrap pareado ----------
    rng = np.random.default_rng(args.boot_seed)
    boot_f1 = {a: np.empty(args.n_boot) for a in arms}
    t0 = time.time()
    for r in range(args.n_boot):
        idx = rng.integers(0, n, n)
        gr = g[idx]
        for a in arms:
            boot_f1[a][r] = macro_f1_from_codes(gr, p_codes[a][idx],
                                                label_codes[a], k)
        if (r + 1) % 2_000 == 0:
            print(f"  bootstrap {r + 1}/{args.n_boot} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    def ci(v: np.ndarray) -> list[float]:
        lo, hi = np.percentile(v, [2.5, 97.5])
        return [round(float(lo), 4), round(float(hi), 4)]

    boot = {"seed": args.seed, "data_seed": DATA_SEED, "eval_n": n,
            "n_boot": args.n_boot, "boot_seed": args.boot_seed,
            "metodo": "bootstrap pareado por instância (mesmas reamostragens em "
                      "todos os braços), IC percentil 95%; Macro F1 por braço com "
                      "rótulos fixados em união(gabarito, predições) da amostra "
                      "completa, classe ausente na reamostragem conta F1=0",
            "macro_f1_braco": {a: {"ponto": round(point_f1[a], 4),
                                   "ic95": ci(boot_f1[a])} for a in arms},
            "pares": {}}
    for x, y in pairs:
        d = boot_f1[x] - boot_f1[y]
        boot["pares"][f"{x}-{y}"] = {
            "delta_ponto": round(point_f1[x] - point_f1[y], 4),
            "delta_ic95": ci(d),
            "delta_media_boot": round(float(d.mean()), 4),
            "frac_replicas_delta_menor_igual_0": round(float((d <= 0).mean()), 4),
        }
    (out_dir / f"bootstrap_f1_s{args.seed}.json").write_text(
        json.dumps(boot, indent=2, ensure_ascii=False) + "\n")
    print(f"bootstrap_f1_s{args.seed}.json gravado "
          f"({time.time() - t0:.0f}s de bootstrap)", flush=True)


if __name__ == "__main__":
    main()
