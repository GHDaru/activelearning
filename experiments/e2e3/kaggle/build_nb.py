"""Gera experiments/e2e3/kaggle/e3prime_kaggle.ipynb (fonte legível aqui)."""
import json
from pathlib import Path

MD = []
CODE = []


def md(s):
    MD.append(s)
    return ("markdown", s)


def code(s):
    CODE.append(s)
    return ("code", s)


cells = [
    md("""# E3′ no Kaggle — BERTimbau multi-semente

Treina o BERTimbau (o classificador **forte**, fora do laço) em 9 braços e mede
na população reservada se o pipeline barato do FALCO (laço leve + oráculo LLM)
entrega um modelo competitivo. Cada braço é **um ajuste fino completo** — por
isso este notebook **exige GPU**.

| Braço | Treino | Responde |
|---|---|---|
| A | itens anotados pelo pipeline real, rótulos do **oráculo** | o que o pipeline entrega |
| B | mesmos itens de A, rótulos **gold** | custo do ruído do oráculo (A−B) |
| C | mesmo tamanho, **aleatórios** do pool, gold | valor da seleção (B−C) |
| D | pool inteiro (50k), gold | a régua (teto do pool) |
| E, E20…E35 | prefixos de 15k…35k da trajetória de entropia (E6), gold | varredura de orçamento |

**Hipótese:** F1(A) ≥ 0,95 × F1(D) com ~18% dos rótulos.

## Antes de rodar — 3 ajustes na barra lateral (Notebook options)
1. **Accelerator: GPU T4 x2 ou P100.** Sem GPU o notebook **para** na célula 1.
2. **Internet: On** (clonar o repositório e baixar o BERTimbau do Hugging Face).
   Exige conta com telefone verificado.
3. **Persistence: Files only** (ou rode via *Save Version → Save & Run All*),
   para os resultados sobreviverem ao fim da sessão.

## Retomada
O `run_e3prime.py` **pula braço já concluído**. Se a sessão cair, rode de novo:
a célula 4 restaura os resultados de `/kaggle/input/` (saída de uma execução
anterior anexada como *dataset*) e só os braços que faltam são treinados.

## Semente
`SEED` na célula 2 é a semente de **treino** (init do cabeçalho, shuffle
intra-época, dropout e o sorteio do braço C). O particionamento pool/população
e a amostra de avaliação ficam **fixos em 42** — é o que mantém os braços
pareáveis entre sementes."""),

    code("""# 1) GPU obrigatória — para aqui se não houver
import subprocess, sys
try:
    print(subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, check=True).stdout)
except Exception:
    sys.exit('SEM GPU. Notebook options -> Accelerator -> GPU T4 x2 (ou P100) e rode de novo. '
             'Nao rode este experimento em CPU: sao 9 ajustes finos completos do BERTimbau.')"""),

    code('''# 2) Configuração
SEED = 123          # semente de TREINO (executor01 usa 7). O runner reescreve esta linha.

# MODO decide batch size e tamanho da avaliação:
#   "canonico"       -> --batch-size 128 --eval-limit 0     (populacao inteira, 177.490 itens)
#   "pareado_s42"    -> --batch-size 16  --eval-limit 20000 (identico a semente 42 legada)
#   "subtreino_bs16" -> --batch-size 16  --eval-limit 0     (avaliacao canonica, lote reduzido —
#        confirmado em 2026-08-18: D sobe +22,5% de Macro F1 com lote 16 vs 128, mesmas epocas;
#        autorizado pelo autor a virar o regime canonico definitivo apos verificacao)
# ATENCAO: os resultados _s42 do repositorio foram gerados com bs=16 e eval-limit=20000.
# Media +- desvio entre sementes so e valida entre execucoes com o MESMO modo.
MODO = "canonico"

# Saida do modo subtreino_bs16 leva o sufixo abaixo (nunca sobrescreve os arquivos
# canonicos _s<semente>.json ja publicados em bs=128).
SFX = "_bs16" if MODO == "subtreino_bs16" else ""

REPO = 'https://github.com/GHDaru/activelearning.git'   # publico: nao precisa de token
BRANCH = 'claude/e3prime-seed-7-rwatey'
REPO_DIR = '/tmp/activelearning'   # fora de /kaggle/working: nao suja a saida do kernel
ARMS_COMPLETOS = 'A,B,C,E,D,E20,E25,E30,E35'
EPOCHS = 3
OUT = '/kaggle/working/results'

BATCH_SIZE = 128 if MODO == 'canonico' else 16
EVAL_LIMIT = 20_000 if MODO == 'pareado_s42' else 0
print(f'semente de treino={SEED} | modo={MODO} | batch={BATCH_SIZE} | '
      f'eval-limit={EVAL_LIMIT} | sufixo={SFX or "(nenhum)"}')'''),

    code("""# 3) Clonar o repositório e instalar só o que falta (o Kaggle já traz torch com CUDA)
#    O clone vai para /tmp (NAO /kaggle/working): assim ele nao entra na saida do
#    kernel, que deve conter so os resultados.
import os, subprocess
if not os.path.exists(REPO_DIR):
    subprocess.run(['git', 'clone', '--depth', '1', '--branch', BRANCH, REPO, REPO_DIR],
                   check=True)
os.chdir(REPO_DIR)
subprocess.run(['git', 'log', '-1', '--format=repositorio em %h %s'], check=True)

%pip -q install transformers scikit-learn
import torch, transformers
print('torch', torch.__version__, '| transformers', transformers.__version__,
      '| CUDA:', torch.cuda.is_available(),
      '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')

# As barras de progresso do Hugging Face inundam o log do Kaggle e chegam a
# truncar o traceback do erro de verdade. Desligadas.
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

# A GPU precisa ser COMPATIVEL, nao so existir. A Tesla P100 e sm_60 e o torch
# pre-instalado do Kaggle so cobre sm_70+: sem esta checagem o notebook rodava
# ~1 min e morria la dentro, com o traceback perdido no log truncado.
if not torch.cuda.is_available():
    raise SystemExit('SEM GPU. Notebook options -> Accelerator -> GPU T4 x2.')
cap = torch.cuda.get_device_capability(0)
print('GPU:', torch.cuda.get_device_name(0), '| compute capability', '%d.%d' % cap)
if cap[0] < 7:
    raise SystemExit(
        f'GPU {torch.cuda.get_device_name(0)} tem compute capability {cap[0]}.{cap[1]} '
        '(sm_%d%d), abaixo do sm_70 minimo suportado pelo PyTorch instalado. '
        'Troque o acelerador para T4 (Notebook options -> Accelerator -> GPU T4 x2) '
        'ou empurre com --maquina NvidiaTeslaT4.' % cap)
# prova de fogo: um kernel CUDA de verdade antes de gastar 2 h
_t = torch.randn(8, 8, device='cuda'); _ = (_t @ _t).sum().item()
print('teste de kernel CUDA: OK')"""),

    code('''# 4) Insumos: cache do oráculo (braços A/B/C) + resultados de uma execução anterior
#
# O cache de anotações do oráculo NAO vem no clone: o .gitignore do repositorio
# exclui experiments/*/results/*.jsonl. Sem ele os bracos A, B e C nao rodam.
# Para habilita-los: suba annotation_cache_nemotron.jsonl como um Dataset privado
# do Kaggle e anexe-o em "Add Input". A celula abaixo o encontra sozinha.
import glob, os, shutil

CACHE_DEST = 'experiments/e5cycle/results/annotation_cache_nemotron.jsonl'
os.makedirs(os.path.dirname(CACHE_DEST), exist_ok=True)
if not os.path.exists(CACHE_DEST):
    achados = glob.glob('/kaggle/input/**/annotation_cache_nemotron.jsonl', recursive=True)
    if achados:
        shutil.copy(achados[0], CACHE_DEST)
        print('cache do oraculo restaurado de', achados[0])

# Retomada: traz de volta resultados desta mesma semente gerados antes. Busca
# pelo nome COM sufixo (e o que um dataset de retomada anexado traria), mas
# restaura no OUT SEM sufixo — e o nome que run_e3prime.py usa para decidir
# se pula o braco (ele nao sabe nada sobre SFX).
os.makedirs(OUT, exist_ok=True)
for p in glob.glob(f'/kaggle/input/**/e3prime_*_s{SEED}{SFX}*.json', recursive=True):
    base = os.path.basename(p)
    if SFX:
        base = base.replace(f'_s{SEED}{SFX}_pred.json', f'_s{SEED}_pred.json') \
                    .replace(f'_s{SEED}{SFX}.json', f'_s{SEED}.json')
    destino = os.path.join(OUT, base)
    if not os.path.exists(destino):
        shutil.copy(p, destino)

tem_cache = os.path.exists(CACHE_DEST)
ARMS = ARMS_COMPLETOS if tem_cache else 'E,D,E20,E25,E30,E35'
if tem_cache:
    print('cache presente -> 9 bracos:', ARMS)
else:
    print('CACHE AUSENTE -> rodando so os 6 bracos que independem dele:', ARMS)
    print('   A, B e C ficam pendentes ate o cache ser anexado (rode de novo depois:')
    print('   os 6 ja prontos sao pulados e so A, B e C treinam).')

ja = sorted(os.path.basename(p) for p in glob.glob(f'{OUT}/e3prime_*_s{SEED}.json')
            if not p.endswith('_pred.json'))
print('ja concluidos nesta semente:', ja or 'nenhum')'''),

    code('''# 5) A execução. Retomada automática: braço concluído é pulado.
#    Duração esperada numa T4: 1,5-2,5 h no modo canônico.
#    subprocess (e nao !python) de proposito: lista de argumentos explicita,
#    sem depender de continuacao de linha do IPython, e com codigo de saida real.
import subprocess, sys, time

cmd = [sys.executable, 'experiments/e2e3/run_e3prime.py',
       '--arms', ARMS, '--epochs', str(EPOCHS),
       '--batch-size', str(BATCH_SIZE), '--eval-limit', str(EVAL_LIMIT),
       '--seed', str(SEED), '--out-dir', OUT]
print('$', ' '.join(cmd), flush=True)

# Toda a saida vai TAMBEM para um arquivo dentro de OUT: o log do Kaggle e
# truncado, e sem isso o traceback do erro se perde (aconteceu na 1a tentativa).
LOG = os.path.join(OUT, f'run_s{SEED}.log')
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
linhas = []
with open(LOG, 'a', encoding='utf-8') as fh:
    fh.write(f'\\n===== nova tentativa: {" ".join(cmd)} =====\\n')
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for linha in proc.stdout:
        fh.write(linha)
        linhas.append(linha)
        print(linha, end='', flush=True)
    proc.wait()

print(f'\\nrun_e3prime.py terminou em {(time.time() - t0) / 60:.1f} min '
      f'(codigo {proc.returncode}) | log completo em {LOG}', flush=True)
if proc.returncode != 0:
    print('\\n===== ULTIMAS 60 LINHAS ANTES DA FALHA =====', flush=True)
    for linha in linhas[-60:]:
        print(linha, end='', flush=True)
    raise SystemExit(f'run_e3prime.py FALHOU (codigo {proc.returncode}). '
                     f'O log foi salvo em {LOG} e desce junto com a saida do kernel. '
                     'Bracos ja concluidos estao salvos: rodar de novo os pula.')

# Renomeia com o sufixo do MODO (se houver) — feito aqui, uma vez, para nao
# colidir com os arquivos canonicos _s<semente>.json ja publicados quando os
# resultados forem copiados para o repositorio. Roda mesmo em falha parcial
# (bracos ja escritos antes do erro tambem levam o sufixo).
if SFX:
    for p in glob.glob(f'{OUT}/e3prime_*_s{SEED}.json') + glob.glob(f'{OUT}/e3prime_*_s{SEED}_pred.json'):
        base = os.path.basename(p)
        novo = base.replace(f'_s{SEED}_pred.json', f'_s{SEED}{SFX}_pred.json') \
                    .replace(f'_s{SEED}.json', f'_s{SEED}{SFX}.json')
        if novo != base:
            os.rename(p, os.path.join(OUT, novo))
    print(f'renomeado com sufixo {SFX}:',
          sorted(os.path.basename(p) for p in glob.glob(f'{OUT}/e3prime_*_s{SEED}{SFX}*.json')))'''),

    code('''# 6) Consolidar e empacotar para download
import glob, json, os, re, shutil

linhas = []
for p in sorted(glob.glob(f'{OUT}/e3prime_*_s{SEED}{SFX}.json')):
    if p.endswith('_pred.json'):
        continue
    d = json.load(open(p))
    linhas.append(d)

if not linhas:
    print('NENHUM resultado — veja o erro na celula 5.')
else:
    porD = {d['arm']: d for d in linhas}.get('D')
    if porD:
        print(f"criterio: F1 >= {0.95 * porD['macro_f1']:.4f} | acc >= {0.95 * porD['accuracy']:.4f}")
    print(f"{'braco':>5} {'n_train':>8} {'Macro F1':>9} {'acuracia':>9}  {'fit(s)':>8}")
    for d in sorted(linhas, key=lambda x: x['n_train']):
        marca = ''
        if porD:
            marca = ('  [F1 OK]' if d['macro_f1'] >= 0.95 * porD['macro_f1'] else '  [F1 --]')
        print(f"{d['arm']:>5} {d['n_train']:>8} {d['macro_f1']:>9.4f} "
              f"{d['accuracy']:>9.4f}  {d['fit_seconds']:>8.1f}{marca}")
    a, dd = {d['arm']: d for d in linhas}.get('A'), porD
    if a and dd:
        ok = a['macro_f1'] >= 0.95 * dd['macro_f1']
        print(f"\\nHIPOTESE (semente {SEED}): F1(A)={a['macro_f1']} vs "
              f"0,95xF1(D)={0.95 * dd['macro_f1']:.4f} -> "
              f"{'SUSTENTADA' if ok else 'NAO sustentada'}")

# tudo o que o executor precisa buscar fica na raiz de /kaggle/working
for p in glob.glob(f'{OUT}/e3prime_*_s{SEED}*.json'):
    shutil.copy(p, os.path.join('/kaggle/working', os.path.basename(p)))
shutil.make_archive(f'/kaggle/working/e3prime_s{SEED}', 'zip', OUT)
print('\\narquivos em /kaggle/working:',
      sorted(os.path.basename(p) for p in glob.glob(f'/kaggle/working/e3prime_*s{SEED}*')))'''),
]

nb = {
    # id por celula: sem ele o nbformat do Kaggle emite MissingIDFieldWarning
    # (e avisa que virara erro duro).
    "cells": [
        {"cell_type": t, "id": f"c{i:02d}", "metadata": {},
         "source": s.splitlines(keepends=True)}
        | ({"execution_count": None, "outputs": []} if t == "code" else {})
        for i, (t, s) in enumerate(cells)
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.11"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path("/home/user/activelearning/experiments/e2e3/kaggle/e3prime_kaggle.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("escrito:", out, out.stat().st_size, "bytes,", len(cells), "celulas")
