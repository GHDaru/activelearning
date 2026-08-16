#!/usr/bin/env bash
# Empurra o E3′ (semente de treino 7) para o Kaggle, acompanha até o fim, baixa
# os resultados e RETOMA sozinho se a sessão cair.
#
# A retomada usa o mecanismo que o próprio run_e3prime.py já tem: braço com
# JSON no diretório de saída é pulado. A cada rodada este script baixa o
# parcial, commita os braços prontos na branch e empurra uma nova versão do
# notebook — que clona a branch já com os parciais e continua de onde parou.
#
# Pré-requisitos
#   pip install kaggle
#   credenciais, em qualquer um dos formatos que o cliente aceita:
#     - token novo: ~/.kaggle/access_token (chmod 600) ou KAGGLE_API_TOKEN
#     - par antigo: ~/.kaggle/kaggle.json  ou KAGGLE_USERNAME + KAGGLE_KEY
#   NUNCA commite credenciais: elas moram em $HOME, fora de qualquer repositório.
#
# Uso
#   experiments/e2e3/kaggle/run_kaggle.sh
#
# Variáveis de ambiente opcionais
#   BRANCH          branch que o notebook clona (padrão: a branch atual)
#   SEED            semente de treino (padrão: 7)
#   CACHE_DATASET   dataset do Kaggle com annotation_cache_nemotron.jsonl,
#                   no formato "usuario/slug". Sem ele os braços A, B e C não
#                   rodam (o cache do oráculo não está versionado no git).
#   MAX_RODADAS     quantas vezes reempurrar após queda (padrão: 6)
#   INTERVALO       segundos entre consultas de status (padrão: 300)
set -euo pipefail

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$RAIZ"

SEED="${SEED:-7}"
BRANCH="${BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"
MAX_RODADAS="${MAX_RODADAS:-6}"
INTERVALO="${INTERVALO:-300}"
RESULTADOS="experiments/e2e3/results"
BRACOS_TODOS=(A B C E D E20 E25 E30 E35)
BRACOS_SEM_ORACULO=(E D E20 E25 E30 E35)

# ---------------------------------------------------------------- pré-checagem
command -v kaggle >/dev/null || { echo "ERRO: 'kaggle' não está no PATH (pip install kaggle)"; exit 1; }

# O Kaggle tem duas formas de credencial e elas guardam o usuário em lugares
# diferentes: o par usuário+chave vive em ~/.kaggle/kaggle.json, e o token novo
# (~/.kaggle/access_token, KGAT_...) não guarda usuário nenhum. Por isso a
# última tentativa é perguntar ao próprio cliente.
if [[ -n "${KAGGLE_USERNAME:-}" ]]; then
  USUARIO="$KAGGLE_USERNAME"
elif [[ -f "$HOME/.kaggle/kaggle.json" ]]; then
  USUARIO="$(python3 -c 'import json,os;print(json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))["username"])')"
else
  USUARIO="$(kaggle config view 2>/dev/null | sed -n 's/^- username: //p' | head -1)"
fi
if [[ -z "${USUARIO:-}" || "$USUARIO" == "None" ]]; then
  echo "ERRO: sem credenciais do Kaggle utilizáveis."
  echo "  token novo : ~/.kaggle/access_token (chmod 600) ou KAGGLE_API_TOKEN"
  echo "  formato par: ~/.kaggle/kaggle.json ou KAGGLE_USERNAME + KAGGLE_KEY"
  exit 1
fi
# O slug sai do kernel-metadata.json — uma fonte só. Cuidado herdado de uma
# lição prática: o Kaggle deriva a URL real do TÍTULO, não do id. Se os dois não
# baterem, o kernel nasce num slug e o acompanhamento consulta outro, para
# sempre. Por isso o título ali é escrito de forma a produzir exatamente este
# slug, e a checagem logo abaixo se recusa a seguir se isso deixar de valer.
SLUG="$(python3 -c 'import json;print(json.load(open("experiments/e2e3/kaggle/kernel-metadata.json"))["id"].split("/")[-1])')"
KERNEL="${USUARIO}/${SLUG}"

# Sem o cache do oráculo, A/B/C não rodam: avisa alto, mas não impede o resto.
if [[ -n "${CACHE_DATASET:-}" ]]; then
  ESPERADOS=("${BRACOS_TODOS[@]}")
else
  ESPERADOS=("${BRACOS_SEM_ORACULO[@]}")
  echo "AVISO: CACHE_DATASET não definido — os braços A, B e C serão pulados."
  echo "       Sem eles não há A-B (ruído do oráculo) nem B-C (valor da seleção)."
fi

echo "kernel .......: $KERNEL"
echo "branch .......: $BRANCH"
echo "semente ......: $SEED"
echo "braços .......: ${ESPERADOS[*]}"

# ------------------------------------------------------- quais braços faltam?
faltantes() {
  local falta=()
  for b in "${ESPERADOS[@]}"; do
    [[ -f "$RESULTADOS/e3prime_${b}_s${SEED}.json" ]] || falta+=("$b")
  done
  echo "${falta[@]:-}"
}

# ------------------------------------------- monta o pacote que vai ao Kaggle
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
preparar() {
  cp experiments/e2e3/e3prime_kaggle.ipynb "$STAGE/"
  python3 - "$STAGE" "$KERNEL" "$BRANCH" "$SEED" "${CACHE_DATASET:-}" <<'PY'
import json, sys
stage, kernel, branch, seed, cache_ds = sys.argv[1:6]

meta = json.load(open("experiments/e2e3/kaggle/kernel-metadata.json", encoding="utf-8"))
meta["id"] = kernel
meta["dataset_sources"] = [cache_ds] if cache_ds else []
json.dump(meta, open(f"{stage}/kernel-metadata.json", "w", encoding="utf-8"), indent=2)

# injeta branch e semente no notebook empurrado (o do repositório fica intacto)
nb_path = f"{stage}/e3prime_kaggle.ipynb"
nb = json.load(open(nb_path, encoding="utf-8"))
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    cell["source"] = [
        f'BRANCH = "{branch}"          # injetado por run_kaggle.sh\n'
        if l.startswith("BRANCH = ") else
        f"SEED = {seed}                                   # injetado por run_kaggle.sh\n"
        if l.startswith("SEED = ") else l
        for l in cell["source"]
    ]
json.dump(nb, open(nb_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("pacote pronto:", kernel, "| branch", branch, "| semente", seed,
      "| dataset:", cache_ds or "(nenhum)")
PY
}

# ------------------------------------------------------------ o laço principal
for ((rodada = 1; rodada <= MAX_RODADAS; rodada++)); do
  FALTA="$(faltantes)"
  if [[ -z "$FALTA" ]]; then
    echo "TODOS os braços concluídos: ${ESPERADOS[*]}"
    break
  fi
  echo
  echo "=== rodada $rodada/$MAX_RODADAS — faltam: $FALTA ==="

  preparar
  SAIDA_PUSH="$(kaggle kernels push -p "$STAGE" 2>&1)"
  echo "$SAIDA_PUSH"
  # O aviso abaixo é fatal, não cosmético: com título e id divergentes o kernel
  # nasce num slug e o resto do script conversa com outro.
  if grep -qi "does not resolve to the specified id" <<<"$SAIDA_PUSH"; then
    echo "ERRO: o título do kernel-metadata.json não resolve para o id '$SLUG'."
    echo "      Ajuste o título para gerar exatamente esse slug e rode de novo."
    exit 1
  fi

  # acompanha até sair de running/queued.
  # Status desconhecido NÃO é 'terminou': o cliente devolve texto de erro em
  # caso de slug errado ou kernel privado, e tratar isso como fim faria o script
  # baixar output inexistente e reempurrar por cima de uma execução viva.
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
          echo "ERRO: não consigo ler o status de '$KERNEL'. Confira o slug em"
          echo "      https://www.kaggle.com/code/${KERNEL} e o kernel-metadata.json."
          exit 1
        }
        continue ;;
    esac
  done

  # baixa o output mesmo se terminou em erro: o parcial é o que permite retomar
  BAIXA="$(mktemp -d)"
  kaggle kernels output "$KERNEL" -p "$BAIXA" || echo "AVISO: download do output falhou"
  find "$BAIXA" -name "e3prime_*_s${SEED}*.json" -exec cp -n {} "$RESULTADOS/" \;
  find "$BAIXA" -name "e3prime_s${SEED}.log" -exec cp {} "$RESULTADOS/" \;
  rm -rf "$BAIXA"

  NOVO="$(git status --porcelain "$RESULTADOS" | wc -l)"
  if [[ "$NOVO" -gt 0 ]]; then
    git add "$RESULTADOS"
    git commit -q -m "E3' semente ${SEED}: parciais da rodada ${rodada} no Kaggle ($(faltantes | wc -w) braços restantes)"
    git pull --rebase origin "$BRANCH" || true
    git push -u origin "$BRANCH"
    echo "parciais commitados e empurrados — a próxima rodada retoma daqui"
  else
    echo "AVISO: rodada $rodada não produziu nenhum braço novo."
    [[ "$rodada" -ge 2 ]] && { echo "duas rodadas sem progresso — parando para não queimar cota de GPU"; break; }
  fi
done

echo
echo "=== estado final (semente $SEED) ==="
python3 - "$SEED" "$RESULTADOS" <<'PY'
import glob, json, os, sys
seed, pasta = sys.argv[1], sys.argv[2]
achados = {}
for p in glob.glob(os.path.join(pasta, f"e3prime_*_s{seed}.json")):
    if p.endswith("_pred.json"):
        continue
    d = json.load(open(p, encoding="utf-8"))
    achados[d["arm"]] = d
if not achados:
    print("nenhum resultado ainda.")
    raise SystemExit(1)
print(f"{'braço':>5} {'n_treino':>9} {'Macro F1':>9} {'acurácia':>9}")
for a, d in sorted(achados.items(), key=lambda kv: kv[1]["n_train"]):
    print(f"{a:>5} {d['n_train']:>9} {d['macro_f1']:>9.4f} {d['accuracy']:>9.4f}")
if "A" in achados and "D" in achados:
    fa, fd = achados["A"]["macro_f1"], achados["D"]["macro_f1"]
    print(f"\nHIPÓTESE F1(A) >= 0,95 x F1(D): {fa:.4f} vs {0.95 * fd:.4f} -> "
          f"{'SUSTENTADA' if fa >= 0.95 * fd else 'NÃO sustentada'}")
PY
