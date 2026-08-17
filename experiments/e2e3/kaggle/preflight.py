"""Pré-voo do E3′: confirma, SEM tocar a GPU, que os braços pedidos montam.

Vale o minuto que custa: montar os conjuntos lê `data/dataset.csv`, a
trajetória de entropia do E6 e o cache de anotações do oráculo. Se algum
faltar, é melhor descobrir agora do que depois de a sessão de GPU já estar
rodando — sobretudo no Kaggle, onde a cota semanal de GPU é limitada.

Não treina nada e não escreve nada: só carrega os dados e monta os conjuntos.

Uso (da raiz do repositório):
    python experiments/e2e3/kaggle/preflight.py
    python experiments/e2e3/kaggle/preflight.py --arms A,B,C,E,D --seed 7

Código de saída 0 = todos os braços pedidos montam; 1 = algum não monta (a
mensagem diz qual arquivo falta).
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
RUNNER = RAIZ / "experiments/e2e3/run_e3prime.py"


def carregar_runner():
    spec = importlib.util.spec_from_file_location("run_e3prime", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", default="A,B,C,E,D,E20,E25,E30,E35")
    ap.add_argument("--seed", type=int, default=7, help="semente de TREINO")
    ap.add_argument("--eval-limit", type=int, default=0)
    args = ap.parse_args()
    pedidos = [a.strip().upper() for a in args.arms.split(",") if a.strip()]

    m = carregar_runner()
    dedup = m.load_base()
    pool = dedup[: m.POOL_SIZE]
    populacao = dedup[m.POOL_SIZE + m.CYCLE_HOLDOUT :]
    rotulos = {l for _, l in dedup}
    print(f"dedup={len(dedup)} pool={len(pool)} população={len(populacao)} "
          f"classes={len(rotulos)}")
    print(f"avaliação (--eval-limit {args.eval_limit}) = "
          f"{len(m.eval_sample(populacao, args.eval_limit))} instâncias")

    # um braço por vez: um que falta não pode esconder os que estão bons
    ok, falharam = [], []
    for nome in pedidos:
        try:
            texts, labels, _ = m.build_arms(pool, rotulos, [nome], args.seed)[nome]
        except FileNotFoundError as exc:
            falharam.append((nome, f"arquivo ausente: {exc.filename}"))
            continue
        except Exception as exc:  # noqa: BLE001 — o pré-voo relata, não decide
            falharam.append((nome, f"{type(exc).__name__}: {exc}"))
            continue
        if not texts or len(texts) != len(labels):
            falharam.append((nome, f"conjunto inválido (n={len(texts)})"))
            continue
        ok.append(nome)
        print(f"  braço {nome:>4}: n={len(texts):>6} classes={len(set(labels)):>4}")

    for nome, motivo in falharam:
        print(f"  braço {nome:>4}: NÃO MONTA — {motivo}")

    print(f"\nmontam: {','.join(ok) or 'nenhum'}")
    if falharam:
        print(f"não montam: {','.join(n for n, _ in falharam)}")
        print("\nOs braços A, B e C dependem de "
              "experiments/e5cycle/results/annotation_cache_nemotron.jsonl, que o "
              ".gitignore mantém fora do repositório. Traga-o antes de gastar GPU "
              "— ou rode só os braços que montam.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
