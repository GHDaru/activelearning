"""Re-avaliação do E6 em 177.490 — tarefa 20260822-1915 (uniformização).

NÃO re-roda o seletor. Para cada braço já rodado (curva publicada em
`popcurve_<clf>_<estrategia><tag>.jsonl`), reaproveita a seleção CONGELADA
salva em `*_state.json` (`labeled_idx`, os 50.000 índices do pool na ordem
em que o oráculo perfeito rotulou) e, para cada checkpoint |L| que já existe
na curva original, retreina o classificador exatamente como da vez anterior
(mesmo prefixo de `labeled_idx`, mesma `internal_split` com o mesmo
`run_seed`) e reavalia SÓ as métricas externas — nos DOIS denominadores
(177.490 e 181.490 inteiro) no mesmo passe, por sugestão do revisor1: já
que o modelo está treinado e 177.490 é subconjunto de 181.490, o custo
marginal do segundo é ~zero e evita que a próxima troca de denominador
custe outras 10h.

`acc_int`/`f1_int` são TRANSPORTADOS da curva original sem recálculo — o
pool não muda, recalcular só injetaria flutuação gratuita numa métrica que
ninguém pediu para mudar.

Persistência de predições por instância: só no checkpoint FINAL (modelo
treinado no orçamento completo, |L|=50.000) — é o ponto que sustenta o
"teto" de F1 relatado na tese e o mais provável de precisar de outro corte
populacional no futuro; persistir todos os ~100 checkpoints x ~180k
instâncias por curva não é o que o pedido original parece pedir e infla o
armazenamento em ordens de grandeza sem uso conhecido. Documentado aqui
para o revisor1 corrigir se a leitura dele for outra.

Saída (ao lado da curva antiga, nada sobrescrito):
  popcurve_<clf>_<estrategia><tag>_pop177490.jsonl   — curva reavaliada
  popcurve_<clf>_<estrategia><tag>_pop177490_final_pred.jsonl — predições
    por instância do checkpoint final, nos dois denominadores

Uso: python reavaliar_177490.py --branch sgd:entropy [--tag _s43] [--smoke]
     python reavaliar_177490.py --all-tab-e6      # as 10 células principais
     python reavaliar_177490.py --all-seeded       # as 32 curvas com semente
     python reavaliar_177490.py --all-tab-e6 --all-seeded --out-dir /kaggle/working/e6_results
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from activelearning.adapters.classifiers.pvbin import PVBinClassifier  # noqa: E402
from activelearning.adapters.classifiers.sgd_text import SgdTextClassifier  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_population_curve import load_base, internal_split, SEED, CLASSIFIERS  # noqa: E402

RES = Path(__file__).resolve().parent / "results"
POOL_SIZE = 50_000
CYCLE_HOLDOUT = 4_000

TAB_E6_BRANCHES = [(c, s, "") for c in ("pvbin", "sgd")
                   for s in ("entropy", "random", "drisl", "drisl-c", "drisl-cs")]
SEEDED_BRANCHES = [(c, "entropy", f"_s{s}") for c in ("pvbin", "sgd") for s in range(43, 51)] + \
                  [(c, "random", f"_s{s}") for c in ("pvbin", "sgd") for s in range(43, 51)]


def run_seed_of(tag: str) -> int:
    return SEED if not tag else int(tag.lstrip("_s"))


def _peak_rss_kb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB no Linux
    except Exception:
        return None


def reavaliar(classifier_name: str, strategy: str, tag: str, pool, pop_177490, pop_181490,
              out_dir: Path, smoke: bool = False):
    factory = CLASSIFIERS[classifier_name]
    run_seed = run_seed_of(tag)
    pool_texts = [t for t, _ in pool]
    pool_labels = [l for _, l in pool]
    pop177_texts = [t for t, _ in pop_177490]
    pop177_labels = [l for _, l in pop_177490]
    pop181_texts = [t for t, _ in pop_181490]
    pop181_labels = [l for _, l in pop_181490]

    curve_path = RES / f"popcurve_{classifier_name}_{strategy}{tag}.jsonl"
    state_path = RES / f"popcurve_{classifier_name}_{strategy}{tag}_state.json"
    if not curve_path.exists() or not state_path.exists():
        print(f"[PULA] {curve_path.name}: curva ou state ausente", flush=True)
        return
    labeled_idx = json.loads(state_path.read_text())["labeled_idx"]
    old_points = [json.loads(l) for l in curve_path.read_text().splitlines() if l.strip()]
    if smoke:
        old_points = old_points[:2]

    out_path = out_dir / f"popcurve_{classifier_name}_{strategy}{tag}_pop177490.jsonl"
    pred_path = out_dir / f"popcurve_{classifier_name}_{strategy}{tag}_pop177490_final_pred.jsonl"
    done = set()
    if out_path.exists():
        for l in out_path.read_text().splitlines():
            if l.strip():
                done.add(json.loads(l)["n_labels"])

    t_start = time.time()
    for point in old_points:
        k = point["n_labels"]
        if k in done:
            continue
        prefix = labeled_idx[:k]
        lx = [pool_texts[i] for i in prefix]
        ly = [pool_labels[i] for i in prefix]
        tr_x, te_x, tr_y, te_y = internal_split(lx, ly, seed=run_seed)
        clf = factory()
        clf.fit(tr_x, tr_y)

        pred177 = []
        for i in range(0, len(pop177_texts), 20_000):
            pred177.extend(clf.predict(pop177_texts[i:i + 20_000]))
        acc177 = accuracy_score(pop177_labels, pred177)
        f1_177 = f1_score(pop177_labels, pred177, average="macro")

        is_final = k == old_points[-1]["n_labels"]
        if is_final:
            pred181 = []
            for i in range(0, len(pop181_texts), 20_000):
                pred181.extend(clf.predict(pop181_texts[i:i + 20_000]))
            acc181 = accuracy_score(pop181_labels, pred181)
            f1_181 = f1_score(pop181_labels, pred181, average="macro")
            with pred_path.open("w") as fh:
                for text, gold, p177 in zip(pop177_texts, pop177_labels, pred177):
                    fh.write(json.dumps({"text": text, "gold": gold, "pred": p177,
                                          "in_177490": True}) + "\n")
                pop177_set = set(pop177_texts)
                for text, gold, p181 in zip(pop181_texts, pop181_labels, pred181):
                    if text not in pop177_set:
                        fh.write(json.dumps({"text": text, "gold": gold, "pred": p181,
                                              "in_177490": False}) + "\n")
        else:
            acc181 = f1_181 = None

        new_point = {
            "n_labels": k,
            "acc_int": point["acc_int"], "f1_int": point["f1_int"],  # transportado
            "acc_ext_177490": round(acc177, 4), "f1_ext_177490": round(f1_177, 4),
            "acc_ext_181490": round(acc181, 4) if acc181 is not None else None,
            "f1_ext_181490": round(f1_181, 4) if f1_181 is not None else None,
            "acc_ext_181490_original": point["acc_ext"], "f1_ext_181490_original": point["f1_ext"],
            "elapsed_s": round(time.time() - t_start, 1),
            # pico de RSS do processo até aqui (KB no Linux, bytes no macOS) —
            # pedido do `local` pra dimensionar paralelismo por RAM na máquina
            # do autor (worker via spawn no Windows duplica o pool em memória).
            "rss_kb": _peak_rss_kb(),
        }
        with out_path.open("a") as fh:
            fh.write(json.dumps(new_point) + "\n")
        print(f"[{classifier_name}/{strategy}{tag}] |L|={k} "
              f"f1_ext_177490={new_point['f1_ext_177490']} "
              f"(original 181490: {point['f1_ext']})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", help="classificador:estrategia, ex. sgd:entropy")
    ap.add_argument("--tag", default="")
    ap.add_argument("--all-tab-e6", action="store_true")
    ap.add_argument("--all-seeded", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=RES,
                     help="onde ESCREVER os resultados novos (default: junto do "
                          "repositório). No Kaggle, aponte para /kaggle/working/... "
                          "— é o único jeito de sobreviver a um corte de sessão, "
                          "porque só esse diretório vira output do kernel.")
    args = ap.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # retomada: se out_dir é diferente de RES (caso Kaggle), traz pra dentro dele
    # qualquer resultado que uma rodada anterior já tenha commitado no repositório
    # — assim esta rodada não refaz checkpoint que já está salvo em algum lugar.
    if out_dir.resolve() != RES.resolve():
        for src in RES.glob("popcurve_*_pop177490*.jsonl"):
            dst = out_dir / src.name
            if not dst.exists():
                dst.write_bytes(src.read_bytes())

    dedup = load_base()
    pool = dedup[:POOL_SIZE]
    pop_177490 = dedup[POOL_SIZE + CYCLE_HOLDOUT:]
    pop_181490 = dedup[POOL_SIZE:]
    print(f"pool={len(pool)} população_177490={len(pop_177490)} "
          f"população_181490={len(pop_181490)}", flush=True)
    assert len(pop_177490) == 177_490 and len(pop_181490) == 181_490

    branches = []
    if args.all_tab_e6:
        branches += TAB_E6_BRANCHES
    if args.all_seeded:
        branches += SEEDED_BRANCHES
    if args.branch:
        c, s = args.branch.split(":")
        branches.append((c, s, args.tag))
    if not branches:
        ap.error("passe --branch, --all-tab-e6 e/ou --all-seeded")

    for classifier_name, strategy, tag in branches:
        reavaliar(classifier_name, strategy, tag, pool, pop_177490, pop_181490,
                  out_dir=out_dir, smoke=args.smoke)


if __name__ == "__main__":
    main()
