#!/usr/bin/env python3
"""Empurra, acompanha e baixa o E3' no Kaggle Notebooks — sem navegador.

Dono: executor01 e executor02 (compartilhado; parametrizado por semente).
Uso típico — uma semente por invocação:

    python experiments/e2e3/kaggle/run_kaggle.py --seed 123
    python experiments/e2e3/kaggle/run_kaggle.py --seed 7

O que faz, em ordem:
  1. confere as credenciais (KAGGLE_USERNAME/KAGGLE_KEY ou ~/.kaggle/kaggle.json);
  2. monta uma cópia do notebook com a semente pedida e o kernel-metadata.json
     correspondente, numa pasta temporária (o notebook versionado não é alterado);
  3. `kaggle kernels push`;
  4. `kaggle kernels status` em laço até terminar;
  5. `kaggle kernels output` e copia os e3prime_*_s<semente>.json para
     experiments/e2e3/results/;
  6. se o kernel morrer com braços faltando, reempurra (o run_e3prime.py pula
     braço já concluído, então reexecutar é seguro e barato).

NUNCA imprime nem grava a chave da API.

Pré-requisitos no Kaggle: conta com telefone verificado (exigência para
`enable_internet`) e cota de GPU disponível (30 h/semana).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[2]
NOTEBOOK = AQUI / "e3prime_kaggle.ipynb"
RESULTS = RAIZ / "experiments/e2e3/results"

BRACOS_COM_CACHE = ["A", "B", "C", "E", "D", "E20", "E25", "E30", "E35"]
BRACOS_SEM_CACHE = ["E", "D", "E20", "E25", "E30", "E35"]

# Estados terminais. O CLI responde no formato
#   <id> has status "KernelWorkerStatus.RUNNING"
# — o prefixo do enum e os underscores precisam entrar na conta, senao o laco
# de espera nunca reconhece o fim.
FIM_OK = {"complete"}
FIM_ERRO = {"error", "cancel_requested", "cancel_acknowledged"}


def usuario_kaggle() -> str:
    """Descobre o usuário sem jamais tocar/imprimir a chave.

    Cobre as três formas de credencial aceitas pelo CLI, em ordem:
      1. KAGGLE_USERNAME/KAGGLE_KEY no ambiente;
      2. ~/.kaggle/kaggle.json (formato antigo, traz o usuário dentro);
      3. token novo (KGAT_...) em KAGGLE_API_TOKEN ou ~/.kaggle/access_token — este
         NÃO carrega o usuário, então perguntamos ao próprio CLI.
    """
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    cfg_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    antigo = cfg_dir / "kaggle.json"
    if antigo.exists():
        return json.loads(antigo.read_text())["username"]
    if os.environ.get("KAGGLE_API_TOKEN") or (cfg_dir / "access_token").exists():
        # `kaggle config view` imprime "- username: <nome>" e nunca a chave
        saida = subprocess.run(["kaggle", "config", "view"],
                               capture_output=True, text=True).stdout
        m = re.search(r"^-\s*username:\s*(\S+)", saida, re.MULTILINE)
        if m and m.group(1).lower() != "none":
            return m.group(1)
        sys.exit("token encontrado, mas `kaggle config view` não revelou o usuário; "
                 "passe KAGGLE_USERNAME=... explicitamente.")
    sys.exit(
        "Credenciais do Kaggle ausentes. Qualquer uma destas serve:\n"
        "  export KAGGLE_API_TOKEN=KGAT_...            (token novo)\n"
        "  ~/.kaggle/access_token                      (mesmo token, chmod 600)\n"
        "  export KAGGLE_USERNAME=... KAGGLE_KEY=...   (formato antigo)\n"
        "O token sai de https://www.kaggle.com/settings -> API.\n"
        "NUNCA commite esse arquivo nem cole a chave em mensagem: os repos são públicos."
    )


def kaggle(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    if check and proc.returncode != 0:
        sys.exit(f"`kaggle {' '.join(args)}` falhou:\n{proc.stdout}\n{proc.stderr}")
    return proc


def monta_pasta(destino: Path, seed: int, kid: str, modo: str,
                datasets: list[str], kernels: list[str],
                maquina: str = "NvidiaTeslaT4") -> None:
    """Copia o notebook com a semente/modo aplicados + escreve o metadata."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    trocas = 0
    for cel in nb["cells"]:
        if cel["cell_type"] != "code":
            continue
        novo = []
        for linha in cel["source"]:
            if re.match(r"^SEED\s*=\s*\d+", linha):
                linha = re.sub(r"^SEED\s*=\s*\d+", f"SEED = {seed}", linha)
                trocas += 1
            elif re.match(r'^MODO\s*=\s*"', linha):
                linha = re.sub(r'^MODO\s*=\s*"[^"]*"', f'MODO = "{modo}"', linha)
                trocas += 1
            novo.append(linha)
        cel["source"] = novo
    if trocas < 2:
        sys.exit(f"não achei as linhas SEED/MODO em {NOTEBOOK} (achei {trocas}) — "
                 "o notebook mudou de forma; ajuste o regex antes de empurrar.")
    (destino / NOTEBOOK.name).write_text(
        json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    # O Kaggle deriva o slug do TITULO. Se o titulo nao "slugificar" exatamente no
    # id, o kernel nasce noutro endereco e `kernels status/output` batem no lugar
    # errado (visto na pratica: id falco-e3prime-s123 x slug falco-e3prime-semente-123).
    # Por isso o titulo e o proprio slug com espacos.
    slug = kid.split("/", 1)[1]
    meta = {
        "id": kid,
        "title": slug.replace("-", " "),
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        # T4 (sm_75) explicitamente. Sem isto o Kaggle pode entregar uma P100
        # (sm_60), que o PyTorch pre-instalado NAO suporta — ele so cobre
        # sm_70..sm_120 — e todo lancamento de kernel CUDA falha. Visto na pratica.
        "machine_shape": maquina,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": datasets,
        "competition_sources": [],
        "kernel_sources": kernels,
        "model_sources": [],
    }
    (destino / "kernel-metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def status(kid: str) -> str:
    saida = kaggle("kernels", "status", kid, check=False).stdout.lower()
    m = re.search(r'status\s*"?(?:kernelworkerstatus\.)?([a-z_]+)', saida)
    return m.group(1) if m else "unknown"


def colhe_saida(kid: str, seed: int) -> list[str]:
    with tempfile.TemporaryDirectory() as tmp:
        kaggle("kernels", "output", kid, "-p", tmp, check=False)
        RESULTS.mkdir(parents=True, exist_ok=True)
        trazidos = []
        for p in Path(tmp).rglob(f"e3prime_*_s{seed}*.json"):
            shutil.copy(p, RESULTS / p.name)
            trazidos.append(p.name)
        return sorted(trazidos)


def bracos_faltando(seed: int, esperados: list[str]) -> list[str]:
    return [b for b in esperados
            if not (RESULTS / f"e3prime_{b}_s{seed}.json").exists()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, help="semente de TREINO (7, 123, ...)")
    ap.add_argument("--modo", choices=["canonico", "pareado_s42"], default="canonico",
                    help="canonico = bs128/eval-limit 0; pareado_s42 = bs16/eval-limit 20000 "
                         "(comparável aos resultados _s42 já publicados)")
    ap.add_argument("--dataset", action="append", default=[],
                    help="dataset do Kaggle a anexar, ex.: usuario/falco-annotation-cache "
                         "(necessário para os braços A, B e C). Pode repetir.")
    ap.add_argument("--retomar-de", action="append", default=[],
                    help="kernel anterior cuja saída serve de retomada, ex.: usuario/slug")
    ap.add_argument("--intervalo", type=int, default=300, help="segundos entre checagens")
    ap.add_argument("--max-tentativas", type=int, default=4,
                    help="reempurradas se o kernel morrer com braços faltando")
    ap.add_argument("--so-monta", action="store_true",
                    help="só gera notebook+metadata numa pasta e sai (sem token)")
    ap.add_argument("--slug", default=None,
                    help="slug do kernel, se diferente de falco-e3prime-s<semente> "
                         "(útil para acompanhar um kernel empurrado antes desta correção)")
    ap.add_argument("--sem-push", action="store_true",
                    help="não empurra: só acompanha e baixa um kernel já em execução")
    ap.add_argument("--maquina", default="NvidiaTeslaT4",
                    choices=["NvidiaTeslaT4", "NvidiaTeslaP100"],
                    help="acelerador. T4 (sm_75) é o padrão: a P100 é sm_60 e o "
                         "PyTorch pré-instalado do Kaggle não a suporta.")
    args = ap.parse_args()

    esperados = BRACOS_COM_CACHE if args.dataset else BRACOS_SEM_CACHE
    if not args.dataset:
        print("AVISO: sem --dataset com o annotation_cache_nemotron.jsonl, os braços "
              "A, B e C NÃO rodam (o cache é excluído pelo .gitignore do repositório).")

    if args.so_monta:
        destino = Path(tempfile.mkdtemp(prefix=f"e3prime_s{args.seed}_"))
        monta_pasta(destino, args.seed, f"SEU_USUARIO/falco-e3prime-s{args.seed}",
                    args.modo, args.dataset, args.retomar_de, args.maquina)
        print(f"pronto em {destino} — suba manualmente ou rode sem --so-monta com o token.")
        return 0

    if not shutil.which("kaggle"):
        sys.exit("CLI do Kaggle ausente: pip install kaggle")
    kid = f"{usuario_kaggle()}/{args.slug or f'falco-e3prime-s{args.seed}'}"
    print(f"kernel: {kid} | semente={args.seed} | modo={args.modo} | "
          f"maquina={args.maquina} | braços={esperados}")

    kernels_retomada = list(args.retomar_de)
    for tentativa in range(1, args.max_tentativas + 1):
        if args.sem_push and tentativa == 1:
            print("[tentativa 1] --sem-push: só acompanhando o kernel já em execução")
        else:
            with tempfile.TemporaryDirectory() as tmp:
                monta_pasta(Path(tmp), args.seed, kid, args.modo, args.dataset,
                            kernels_retomada, args.maquina)
                print(f"[tentativa {tentativa}] push...")
                print(kaggle("kernels", "push", "-p", tmp).stdout.strip())

        desconhecidos = 0
        while True:
            time.sleep(args.intervalo)
            st = status(kid)
            print(f"  [{time.strftime('%H:%M:%S')}] status={st}", flush=True)
            if st in FIM_OK | FIM_ERRO:
                break
            # "unknown" seguido = slug errado ou API fora; nao girar para sempre
            desconhecidos = desconhecidos + 1 if st == "unknown" else 0
            if desconhecidos >= 5:
                sys.exit(f"5 consultas seguidas sem status reconhecível para {kid}. "
                         "Confira o slug real na URL do notebook "
                         "(kaggle.com/code/<usuario>/<slug>) e repita com --slug <slug>.")

        trazidos = colhe_saida(kid, args.seed)
        print(f"  baixados: {trazidos or 'nada'}")
        faltam = bracos_faltando(args.seed, esperados)
        if not faltam:
            print(f"CONCLUÍDO: todos os braços da semente {args.seed} estão em {RESULTS}")
            return 0
        print(f"  ainda faltam {faltam} (status final: {st}) — reempurrando com retomada")
        # a próxima tentativa usa a saída desta como insumo: braço pronto é pulado
        if kid not in kernels_retomada:
            kernels_retomada.append(kid)

    print(f"PAROU com braços faltando: {bracos_faltando(args.seed, esperados)}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
