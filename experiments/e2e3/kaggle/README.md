# E3′ no Kaggle Notebooks

Receita para rodar o E3′ (9 ajustes finos do BERTimbau) numa GPU grátis do
Kaggle — T4 ou P100, sessão de até 12 h. Serve as duas sementes que faltam
(7 e 123): tudo aqui é **parametrizado pela semente**, então os dois executores
usam os mesmos dois arquivos.

| Arquivo | Para que serve |
|---|---|
| `e3prime_kaggle.ipynb` | o notebook. `SEED` e `MODO` ficam na célula 2 |
| `run_kaggle.py` | push + acompanhamento + download pela API, sem navegador |
| `build_nb.py` | gera o `.ipynb` a partir de fonte legível — **edite aqui**, não no JSON |

## A pegadinha que custa 2 h: a GPU P100 não serve

O Kaggle entrega **T4 ou P100**, conforme disponibilidade. A **P100 é compute
capability 6.0 (`sm_60`)** e o PyTorch pré-instalado na imagem só cobre
**`sm_70` a `sm_120`** — com uma P100 todo lançamento de kernel CUDA falha, e o
notebook morre dentro do primeiro braço:

    Tesla P100-PCIE-16GB with CUDA capability sm_60 is not compatible with the
    current PyTorch installation.

Por isso o `kernel-metadata.json` gerado fixa `"machine_shape": "NvidiaTeslaT4"`
(sobrescreva com `--maquina`), e a célula 3 confere a compute capability e roda
um `matmul` de prova antes de gastar horas. **Quem sobe pela interface tem que
escolher `GPU T4 x2` à mão.**

## Caminho A — com o token da API (automático)

```bash
pip install kaggle
export KAGGLE_USERNAME=... KAGGLE_KEY=...     # https://www.kaggle.com/settings -> API
python experiments/e2e3/kaggle/run_kaggle.py --seed 123 \
    --dataset <usuario>/falco-annotation-cache
```

O script empurra o kernel, consulta o status a cada 5 min, baixa a saída para
`experiments/e2e3/results/` e — se a sessão morrer com braços faltando —
reempurra usando a saída anterior como retomada. O token nunca é impresso nem
gravado.

## Caminho B — sem token (o autor roda pela interface)

```bash
python experiments/e2e3/kaggle/run_kaggle.py --seed 123 --so-monta
```

Gera numa pasta temporária o notebook já com a semente aplicada. No Kaggle:
*New Notebook → File → Import Notebook*, e antes de rodar ajuste a barra lateral:

1. **Accelerator: GPU T4 x2** (ou P100) — sem isso a célula 1 aborta de propósito;
2. **Internet: On** — precisa clonar o repositório e baixar o BERTimbau do
   Hugging Face (o Kaggle exige telefone verificado para liberar internet);
3. **Persistence: Files only**, ou rode por *Save Version → Save & Run All*.

Ao terminar, os `e3prime_*_s<semente>.json` ficam na raiz de `/kaggle/working`
(a célula 6 os copia para lá) e também num `.zip` para download em bloco.

## Duas pendências conhecidas

### 1. O cache do oráculo não vem no clone — braços A, B e C ficam de fora

`experiments/e5cycle/results/annotation_cache_nemotron.jsonl` é excluído pelo
`.gitignore` (regra `experiments/*/results/*.jsonl`), então **não existe no
repositório**. Sem ele, `build_arms()` nem chega a treinar: quebra ao abrir o
arquivo. São 9.357 linhas (~1–2 MB) com as anotações reais do oráculo NIM — não
dá para regerar sem chamar o oráculo de novo.

Para destravar, suba o arquivo como **Dataset privado do Kaggle** e passe
`--dataset <usuario>/<slug>`; a célula 4 o encontra sozinha em `/kaggle/input/`.
Sem ele o notebook **não quebra**: roda os 6 braços que independem do cache
(E, D, E20, E25, E30, E35) e avisa. Quando o cache chegar, basta rodar de novo
— os 6 prontos são pulados e só A, B e C treinam.

### 2. `MODO` decide se a semente é comparável com a semente 42 já publicada

Os `e3prime_*_s42.json` do repositório foram gerados com **`--batch-size 16
--eval-limit 20000`** (avaliação em 20.092 itens). O comando canônico das
tarefas das sementes 7 e 123 é **`--batch-size 128 --eval-limit 0`**, que avalia
na população inteira — **177.490 itens**.

São dois regimes diferentes em duas frentes: o tamanho do lote muda a
trajetória de otimização (mesmo `learning_rate` de 5e-5) e o conjunto de
avaliação muda o denominador do Macro F1 sobre 714 classes. Média ± desvio
entre sementes só é válida entre execuções do **mesmo** modo.

- `--modo canonico` (padrão) → `bs=128`, `eval-limit=0`. É o comando das tarefas.
- `--modo pareado_s42` → `bs=16`, `eval-limit=20000`. Reproduz o regime do s42.

A escolha é do `principal`/autor: ou as sementes 7 e 123 saem em `pareado_s42`,
ou a semente 42 é refeita em `canonico`. Numa T4 as duas opções cabem numa
sessão — os tempos longos do s42 no repositório (D levou 10.593 s de ajuste)
são de um regime sem GPU dedicada.
