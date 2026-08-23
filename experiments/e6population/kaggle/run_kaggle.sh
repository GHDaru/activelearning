#!/usr/bin/env bash
# Empurra a reavaliação do E6 em 177.490 para o Kaggle (CPU só), acompanha até
# o fim, baixa os resultados e RETOMA sozinho se a sessão cair ou bater no
# limite de tempo do Kaggle.
#
# A retomada usa o próprio mecanismo de `reavaliar_177490.py`: checkpoint já
# gravado no `.jsonl` de saída é pulado. A cada rodada este script baixa o
# parcial, commita na branch e empurra uma nova versão do notebook — que
# clona a branch já com os parciais e continua de onde parou.
#
# Pré-requisitos: ver experiments/e2e3/kaggle/run_kaggle.sh (mesmo padrão).
#
# Uso
#   experiments/e6population/kaggle/run_kaggle.sh
#
# Variáveis de ambiente opcionais
#   BRANCH        branch que o notebook clona (padrão: a branch atual)
#   MAX_RODADAS   quantas vezes reempurrar (padrão: 12 — sessões de CPU do
#                 Kaggle têm teto de tempo menor que uma T4/P100; a campanha
#                 inteira (42 curvas) é mais longa que uma sessão só)
#   INTERVALO     segundos entre consultas de status (padrão: 300)
set -euo pipefail

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$RAIZ"

BRANCH="${BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"
MAX_RODADAS="${MAX_RODADAS:-12}"
INTERVALO="${INTERVALO:-300}"
RESULTADOS="experiments/e6population/results"

TAB_E6_CLASSIF=(pvbin sgd)
TAB_E6_ESTRAT=(entropy random drisl drisl-c drisl-cs)
SEEDS=(43 44 45 46 47 48 49 50)

command -v kaggle >/dev/null || { echo "ERRO: 'kaggle' não está no PATH (pip install kaggle)"; exit 1; }

if [[ -n "${KAGGLE_USERNAME:-}" ]]; then
  USUARIO="$KAGGLE_USERNAME"
elif [[ -f "$HOME/.kaggle/kaggle.json" ]]; then
  USUARIO="$(python3 -c 'import json,os;print(json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))["username"])')"
else
  USUARIO="$(kaggle config view 2>/dev/null | sed -n 's/^- username: //p' | head -1)"
fi
if [[ -z "${USUARIO:-}" || "$USUARIO" == "None" ]]; then
  echo "ERRO: sem credenciais do Kaggle utilizáveis."
  exit 1
fi
SLUG="$(python3 -c 'import json;print(json.load(open("experiments/e6population/kaggle/kernel-metadata.json"))["id"].split("/")[-1])')"
KERNEL="${USUARIO}/${SLUG}"

echo "kernel .......: $KERNEL"
echo "branch .......: $BRANCH"
echo "curvas alvo ..: 10 (tab:e6) + 32 (32 com semente) = 42"

# ---------------------------------------------------- quantos pontos cada curva tem
pontos_esperados() {  # $1 = nome-base da curva (ex.: sgd_entropy ou sgd_entropy_s43)
  local orig="$RESULTADOS/popcurve_${1}.jsonl"
  [[ -f "$orig" ]] && wc -l < "$orig" || echo 0
}

# ------------------------------------------------------- quais curvas faltam?
faltantes() {
  local falta=()
  local nomes=()
  for c in "${TAB_E6_CLASSIF[@]}"; do
    for s in "${TAB_E6_ESTRAT[@]}"; do
      nomes+=("${c}_${s}")
    done
    for s in entropy random; do
      for sem in "${SEEDS[@]}"; do
        nomes+=("${c}_${s}_s${sem}")
      done
    done
  done
  for n in "${nomes[@]}"; do
    local alvo="$RESULTADOS/popcurve_${n}_pop177490.jsonl"
    local esperado="$(pontos_esperados "$n")"
    local atual=0
    [[ -f "$alvo" ]] && atual="$(wc -l < "$alvo")"
    [[ "$atual" -lt "$esperado" ]] && falta+=("$n:${atual}/${esperado}")
  done
  echo "${falta[@]:-}"
}

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
preparar() {
  cp experiments/e6population/kaggle/reavaliar_kaggle.ipynb "$STAGE/"
  python3 - "$STAGE" "$KERNEL" "$BRANCH" <<'PY'
import json, sys
stage, kernel, branch = sys.argv[1:4]

meta = json.load(open("experiments/e6population/kaggle/kernel-metadata.json", encoding="utf-8"))
meta["id"] = kernel
json.dump(meta, open(f"{stage}/kernel-metadata.json", "w", encoding="utf-8"), indent=2)

nb_path = f"{stage}/reavaliar_kaggle.ipynb"
nb = json.load(open(nb_path, encoding="utf-8"))
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    cell["source"] = [
        f'BRANCH = "{branch}"          # injetado por run_kaggle.sh\n'
        if l.startswith("BRANCH = ") else l
        for l in cell["source"]
    ]
json.dump(nb, open(nb_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("pacote pronto:", kernel, "| branch", branch)
PY
}

for ((rodada = 1; rodada <= MAX_RODADAS; rodada++)); do
  FALTA="$(faltantes)"
  if [[ -z "$FALTA" ]]; then
    echo "TODAS as 42 curvas concluídas."
    break
  fi
  echo
  echo "=== rodada $rodada/$MAX_RODADAS — faltam: $FALTA ==="

  preparar
  SAIDA_PUSH="$(kaggle kernels push -p "$STAGE" 2>&1)"
  echo "$SAIDA_PUSH"
  if grep -qi "does not resolve to the specified id" <<<"$SAIDA_PUSH"; then
    echo "ERRO: o título do kernel-metadata.json não resolve para o id '$SLUG'."
    exit 1
  fi

  ERROS_SEGUIDOS=0
  while true; do
    sleep "$INTERVALO"
    STATUS="$(kaggle kernels status "$KERNEL" 2>&1 || true)"
    echo "[$(date -u +%H:%M:%SZ)] $STATUS"
    case "$STATUS" in
      *running*|*RUNNING*|*queued*|*QUEUED*)
        ERROS_SEGUIDOS=0; continue ;;
      *complete*|*COMPLETE*|*error*|*ERROR*|*cancel*|*CANCEL*)
        break ;;
      *)
        ERROS_SEGUIDOS=$((ERROS_SEGUIDOS + 1))
        echo "  (status não reconhecido — tentativa $ERROS_SEGUIDOS de 3)"
        [[ "$ERROS_SEGUIDOS" -ge 3 ]] && {
          echo "ERRO: não consigo ler o status de '$KERNEL'."
          exit 1
        }
        continue ;;
    esac
  done

  BAIXA="$(mktemp -d)"
  kaggle kernels output "$KERNEL" -p "$BAIXA" || echo "AVISO: download do output falhou"
  find "$BAIXA" -name "popcurve_*_pop177490*.jsonl" -exec cp {} "$RESULTADOS/" \;
  rm -rf "$BAIXA"

  NOVO="$(git status --porcelain "$RESULTADOS" | wc -l)"
  if [[ "$NOVO" -gt 0 ]]; then
    git add "$RESULTADOS"
    git commit -q -m "E6 177490: parciais da rodada ${rodada} no Kaggle ($(faltantes | wc -w) curvas com pendência)"
    git pull --rebase origin "$BRANCH" || true
    git push -u origin "$BRANCH"
    echo "parciais commitados e empurrados — a próxima rodada retoma daqui"
  else
    echo "AVISO: rodada $rodada não produziu nenhum checkpoint novo."
    [[ "$rodada" -ge 3 ]] && { echo "três rodadas sem progresso — parando para não queimar cota"; break; }
  fi
done

echo
echo "=== estado final ==="
faltantes
