# BERTimbau local (RTX 3090) — passo a passo

Guia para treinar o BERTimbau na base completa (~250k) na sua máquina e medir
a capacidade preditiva com supervisão completa. Validado em CPU no sandbox em
17/07/2026 (smoke: 900 docs/30 classes/3 épocas → acc 86,3%, Macro-F1 0,859);
os passos abaixo são idênticos na GPU, só mudam velocidade e lote.

## 0. Pré-requisitos

- Windows com WSL2 **ou** Linux; driver NVIDIA atualizado (o `nvidia-smi`
  deve funcionar e mostrar a RTX 3090).
- Python 3.11+ e git.
- ~5 GB livres em disco (modelo 440 MB + ambiente + dados).

## 1. Clonar e preparar o ambiente

```bash
git clone https://github.com/GHDaru/activelearning.git
cd activelearning
python3 -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1

# PyTorch com CUDA (cu121 cobre a 3090; ajuste se seu driver for mais antigo)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers scikit-learn
```

Verifique a GPU:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# esperado: ... True NVIDIA GeForce RTX 3090
```

## 2. Dados

Coloque o CSV da base em `data/dataset.csv` (colunas `nm_item`, `nm_product`
— o mesmo arquivo usado nos experimentos E0/E1). Se você já tem o repositório
com a base, nada a fazer.

## 3. Smoke test (2–3 min) — valida a cadeia na sua máquina

```bash
python experiments/e2e3/run_smoke_cpu.py --classes 30 --per-class 30 --epochs 3
```

Esperado: acc ≈ 0,86 no subconjunto (como no sandbox). Na primeira execução o
modelo `neuralmind/bert-base-portuguese-cased` (440 MB) é baixado do
HuggingFace.

## 4. Ensaio rápido em GPU (~2–4 min) — calibra a velocidade real

```bash
python experiments/e2e3/train_full.py --limit 20000 --epochs 1 --batch-size 64
```

Anote o `fit_seconds`: a razão `docs/segundo` desse ensaio permite prever o
tempo da base cheia com precisão (250k/época ≈ 12,5× esse tempo).

## 5. Treino completo (base ~250k)

```bash
python experiments/e2e3/train_full.py --epochs 3 --batch-size 64
# variações úteis:
#   --batch-size 128        (a 3090 tem 24 GB; ml=32 permite lotes grandes)
#   --max-length 32         (padrão; descrições têm 4-50 caracteres)
#   --test-size 0.1         (teste estratificado intocado, padrão)
#   --seed 42               (reprodutibilidade)
```

O relatório final (acc, Macro-F1, tempos) é salvo em
`experiments/e2e3/results/full_*.json` — commit esse JSON: é o artefato
rastreável do número que entra na tese.

### Estimativas de tempo

| Ambiente | Vazão medida/estimada | 1 época (225k treino) | 3 épocas |
|---|---|---|---|
| Sandbox CPU (medido) | ~11,5 docs/s | ~5,5 h | ~16–18 h |
| RTX 3090 fp32 b=64 (estimado) | ~350–500 docs/s | ~8–11 min | ~25–35 min |
| RTX 3090 b=128 (estimado) | ~500–800 docs/s | ~5–8 min | ~15–25 min |

Some ~2–4 min de tokenização + ~1–3 min de predição no teste (25k). Total
esperado na 3090: **~30–60 min** para o quadro completo. Use o passo 4 para
substituir a estimativa pela medição da sua máquina.

### O que esperar do número

O teto do PVBin com os 250k é ~89,6% de acurácia / Macro-F1 ~0,70. A
expectativa (a confirmar — é exatamente o propósito do E2) é o BERTimbau
superar em Macro-F1 (melhor nas classes raras); o teto de medição do gabarito
é ~99,3% (auditoria do Cap. 3).

## 6. Depois do teto: E2 (épocas × |L|)

Com a máquina validada, o E2 da tese é a grade `épocas ∈ {1..5} × |L| ∈
{1k, 5k, 15k, 50k, 250k}` com o mesmo script (`--limit` controla |L|;
rode 8 sementes com `--seed 0..7`). Guarde todos os JSONs de
`experiments/e2e3/results/`.

## Problemas comuns

- **`torch.cuda.is_available()` = False**: driver antigo ou torch CPU
  instalado — reinstale com o índice cu121; em WSL2, confirme `nvidia-smi`
  dentro do WSL.
- **OOM na GPU**: reduza `--batch-size` (32) — improvável com ml=32 na 3090.
- **Aviso "classifier.weight ... newly initialized"**: normal — a cabeça de
  classificação nasce nova no fine-tuning.
- **Download do HF falha atrás de proxy**: exporte `HF_HOME` para um caminho
  gravável e confira o certificado corporativo (`REQUESTS_CA_BUNDLE`).
