"""Gera experiments/e6population/kaggle/reavaliar_kaggle.ipynb (fonte legível aqui)."""
import json
from pathlib import Path

cells = [
    ("markdown", """# E6 — reavaliação em 177.490 (uniformização, tarefa 20260822-1915)

Re-avalia as 42 curvas do E6 (10 células de `tab:e6` + 32 curvas com semente,
entropia e aleatório × 2 classificadores × 8 sementes) no denominador único
**177.490**, no lugar dos 181.490 atuais. **CPU só — sklearn, sem GPU.**

**Método (roteiro do `revisor1`, tarefa 1900): re-avaliação, nunca re-seleção.**
A seleção de cada curva está CONGELADA em `*_state.json` (`labeled_idx`, os
50.000 índices do pool na ordem escolhida pelo seletor original). Para cada
checkpoint |L| que já existe na curva publicada, este notebook retreina o
classificador com o MESMO prefixo (`labeled_idx[:k]`) e reavalia só as
métricas externas — nos dois denominadores (177.490 e 181.490 inteiro) no
mesmo passe, persistindo as predições por instância do checkpoint final.
`acc_int`/`f1_int` são transportados sem recálculo (o pool não muda).

Resultados **ao lado** dos antigos (sufixo `_pop177490`), nada sobrescrito.
Retomada automática por checkpoint: se a sessão cair, a próxima rodada pula
tudo que já está no `.jsonl` de saída.
"""),
    ("code", """# 1) Configuração
BRANCH = "claude/e3prime-seed-7-bx08ks"   # o runner reescreve esta linha
"""),
    ("code", """# 2) Clonar o repositório (vai para /tmp, fora da saída do kernel)
import os, subprocess
REPO = "https://github.com/GHDaru/activelearning.git"
REPO_DIR = "/tmp/activelearning"
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", "--branch", BRANCH, REPO, REPO_DIR], check=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "fetch", "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", REPO_DIR, "checkout", BRANCH], check=True)
    subprocess.run(["git", "-C", REPO_DIR, "reset", "--hard", f"origin/{BRANCH}"], check=True)
os.chdir(REPO_DIR)
print("cwd:", os.getcwd())
"""),
    ("code", """# 3) Diagnóstico de hardware/ambiente — pedido do agente `local` (tarefa
#    20260823-0500) pra comparar o Kaggle contra a máquina do autor ANTES de
#    decidir migrar. Specs MEDIDAS aqui, não as prometidas pela documentação.
import importlib, os, platform, subprocess

for mod in ("sklearn", "numpy", "scipy"):
    m = importlib.import_module(mod)
    print(f"{mod} {m.__version__}")
print("python", platform.python_version())

print("\\ncpu_count (os.cpu_count):", os.cpu_count())
try:
    print(subprocess.run(["nproc", "--all"], capture_output=True, text=True).stdout.strip())
except Exception as exc:
    print("nproc indisponível:", exc)
try:
    cpuinfo = open("/proc/cpuinfo").read()
    modelo = next((l for l in cpuinfo.splitlines() if "model name" in l), "?")
    print(modelo)
except Exception as exc:
    print("/proc/cpuinfo indisponível:", exc)
try:
    meminfo = {l.split(":")[0]: l.split(":")[1].strip()
               for l in open("/proc/meminfo") if ":" in l}
    print("MemTotal:", meminfo.get("MemTotal"), "| MemAvailable:", meminfo.get("MemAvailable"))
except Exception as exc:
    print("/proc/meminfo indisponível:", exc)

print("\\npip freeze (filtrado):")
freeze = subprocess.run(["python", "-m", "pip", "freeze"], capture_output=True, text=True).stdout
for linha in freeze.splitlines():
    if linha.split("==")[0].lower() in ("scikit-learn", "numpy", "scipy", "joblib", "threadpoolctl"):
        print(" ", linha)
"""),
    ("code", """# 4) A execução — as 42 curvas, retomada automática por checkpoint.
#
#    --out-dir aponta DIRETO pra /kaggle/working: é o único diretório que
#    sobrevive ao fim da sessão e que `kaggle kernels output` consegue baixar.
#    Escrever primeiro no clone (/tmp) e só copiar no final foi o bug da v1
#    deste notebook — se a sessão do Kaggle corta o processo antes de ele
#    terminar as 42 curvas (bem provável, a campanha inteira passa de 20h),
#    a cópia final nunca roda e TODO o progresso da sessão se perde. Escrever
#    direto em /kaggle/working faz cada checkpoint sobreviver assim que é
#    gravado, igual ao padrão já usado no e3prime_kaggle.ipynb (célula 5).
#
#    subprocess (não !python) de propósito: lista de argumentos explícita e
#    código de saída real, como o padrão do E3'/E1E4.
import subprocess, sys, time

OUT = "/kaggle/working/e6_results"
t0 = time.time()
cmd = [sys.executable, "experiments/e6population/reavaliar_177490.py",
       "--all-tab-e6", "--all-seeded", "--out-dir", OUT]
print("comando:", " ".join(cmd))
proc = subprocess.run(cmd)
print(f"\\nsaiu com código {proc.returncode} em {(time.time() - t0)/3600:.2f} h")
"""),
    ("code", """# 5) Conferência final — o output já está em /kaggle/working/e6_results
#    (célula 4 escreveu direto lá); esta célula só lista o que existe.
import glob, os

OUT = "/kaggle/working/e6_results"
novos = sorted(glob.glob(f"{OUT}/*_pop177490*.jsonl"))
print(f"{len(novos)} arquivo(s) em {OUT}:")
for p in novos:
    print(" ", os.path.basename(p), os.path.getsize(p), "bytes")
"""),
]

nb = {
    "cells": [
        {"cell_type": t, "id": f"c{i:02d}", "metadata": {},
         "source": s.splitlines(keepends=True)}
        | ({"execution_count": None, "outputs": []} if t == "code" else {})
        for i, (t, s) in enumerate(cells)
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path("/home/user/activelearning/experiments/e6population/kaggle/reavaliar_kaggle.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("escrito:", out, out.stat().st_size, "bytes,", len(cells), "celulas")
