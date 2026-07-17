"""Impacto do ruído de gabarito (conflitos de rótulo) na medição dos oráculos.

Um "conflito" é uma descrição normalizada que aparece na base com 2+ rótulos-ouro
distintos. Para cada oráculo já anotado no E0, reavalia a acurácia sob três regimes:

  original   : acerto = rótulo predito == gold da linha amostrada (medição atual);
  sem-ruido  : instâncias conflitantes EXCLUÍDAS da amostra;
  multi-gold : acerto = rótulo predito pertence ao CONJUNTO de golds observados
               para aquela descrição na base inteira.

A diferença (multi-gold - original) é o teto de erro espúrio atribuível ao
gabarito; (sem-ruido) mostra o efeito de simplesmente remover as instâncias.

Uso: uv run python experiments/e0/analyze_noise_impact.py [--results DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from activelearning.adapters.datasets.retail_csv import load_rows_and_schema  # noqa: E402
from activelearning.domain.instances import Label, normalize_label  # noqa: E402
from activelearning.domain.metrics import wilson_interval  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=_ROOT / "experiments/e0/results")
    parser.add_argument("--config", type=Path, default=_ROOT / "experiments/e0/config.json")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows, schema = load_rows_and_schema(config)

    # --- Censo do ruído na base -------------------------------------------------
    golds_by_text: dict[str, set[str]] = defaultdict(set)
    for text, label in rows:
        normalized = schema.validate(label)
        golds_by_text[normalize_label(text)].add(
            normalized.value if normalized else "_rare_"
        )
    conflicted_texts = {t for t, gs in golds_by_text.items() if len(gs) > 1}
    n_conf_rows = sum(
        1 for text, _ in rows if normalize_label(text) in conflicted_texts
    )
    sizes = Counter(len(golds_by_text[t]) for t in conflicted_texts)
    print("=== Censo do ruído (base corrigida) ===")
    print(f"descrições distintas: {len(golds_by_text):,} | conflitantes: "
          f"{len(conflicted_texts):,} | linhas afetadas: {n_conf_rows:,} "
          f"({n_conf_rows/len(rows)*100:.2f}%)")
    print("golds por descrição conflitante:",
          dict(sorted(sizes.items())))
    exemplos = sorted(conflicted_texts)[:5]
    for t in exemplos:
        print(f"  ex.: {t!r} -> {sorted(golds_by_text[t])}")

    # gold por instância (id estável e0-{i}) e texto por instância
    gold_by_id: dict[str, str] = {}
    text_by_id: dict[str, str] = {}
    for i, (text, label) in enumerate(rows):
        normalized = schema.validate(label)
        gold_by_id[f"e0-{i}"] = normalized.value if normalized else "_rare_"
        text_by_id[f"e0-{i}"] = normalize_label(text)

    # --- Reavaliação por oráculo -------------------------------------------------
    table = []
    for sample_name in ("rand", "strat"):
        sample_dir = args.results / sample_name
        if not sample_dir.exists():
            continue
        for path in sorted(sample_dir.glob("annotations_*.jsonl")):
            n = n_ok = n_conf = n_ok_multi = n_ok_clean = n_clean = 0
            oracle_id = None
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    rec = json.loads(line)
                    iid = rec["instance_id"]
                    if iid not in gold_by_id:
                        continue
                    oracle_id = rec["oracle_id"]
                    pred = rec.get("label")
                    gold = gold_by_id[iid]
                    text = text_by_id[iid]
                    is_conf = text in conflicted_texts
                    ok = pred == gold
                    ok_multi = pred in golds_by_text[text] if pred else False
                    n += 1
                    n_ok += ok
                    n_ok_multi += ok_multi
                    if is_conf:
                        n_conf += 1
                    else:
                        n_clean += 1
                        n_ok_clean += ok
            if not n or oracle_id is None:
                continue
            acc, acc_clean, acc_multi = n_ok / n, (n_ok_clean / n_clean if n_clean else 0), n_ok_multi / n
            lo, hi = wilson_interval(n_ok, n)
            lo_c, hi_c = wilson_interval(n_ok_clean, n_clean) if n_clean else (0, 0)
            table.append({
                "sample": sample_name, "oracle_id": oracle_id, "n": n,
                "n_conflito": n_conf, "pct_conflito": round(n_conf / n * 100, 2),
                "acc_original": round(acc, 4),
                "ic_original": [round(lo, 4), round(hi, 4)],
                "acc_sem_ruido": round(acc_clean, 4),
                "ic_sem_ruido": [round(lo_c, 4), round(hi_c, 4)],
                "acc_multigold": round(acc_multi, 4),
                "delta_sem_ruido_pp": round((acc_clean - acc) * 100, 2),
                "delta_multigold_pp": round((acc_multi - acc) * 100, 2),
            })

    out = args.results / "noise_impact.json"
    out.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== Impacto por oráculo (gravado em {out}) ===")
    print(f"{'amostra':6} {'oráculo':45} {'n':>5} {'conf%':>6} "
          f"{'acc':>6} {'s/ruído':>8} {'multi':>6} {'Δs/r':>6} {'Δmulti':>7}")
    for r in table:
        print(f"{r['sample']:6} {r['oracle_id']:45} {r['n']:>5} "
              f"{r['pct_conflito']:>6.2f} {r['acc_original']:>6.3f} "
              f"{r['acc_sem_ruido']:>8.3f} {r['acc_multigold']:>6.3f} "
              f"{r['delta_sem_ruido_pp']:>+6.2f} {r['delta_multigold_pp']:>+7.2f}")


if __name__ == "__main__":
    main()
