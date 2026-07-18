# E2/E3 BERTimbau em Docker — e o que fazer sem GPU

Imagem dedicada ao experimento BERTimbau (E2: teto supervisionado; E3:
validação integrada do FALCO). A imagem não contém dados nem resultados:
`data/` entra como volume somente-leitura, `results/` como volume de escrita
e o modelo HuggingFace (440 MB) fica num volume de cache reutilizado.

Validado em 18/07/2026 no sandbox (variante CPU): build 71 s; smoke dentro do
contêiner idêntico ao nativo (400 docs/2 épocas → acc 0,77 em 75 s; o smoke
completo 900 docs/3 épocas → acc 0,89).

## Com GPU (RTX 3090 ou qualquer NVIDIA)

Pré-requisito único além do Docker: o
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
(`nvidia-ctk`). No Windows, Docker Desktop + WSL2 já expõem a GPU sem toolkit
manual. Teste: `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`.

```bash
# na raiz do repositório
docker compose -f experiments/e2e3/docker-compose.yml build e2e3-gpu

# smoke (valida a cadeia, ~1 min em GPU)
docker compose -f experiments/e2e3/docker-compose.yml run --rm e2e3-gpu

# calibração de velocidade (~2-4 min): anote fit_seconds
docker compose -f experiments/e2e3/docker-compose.yml run --rm e2e3-gpu \
  experiments/e2e3/train_full.py --limit 20000 --epochs 1 --batch-size 64

# E2 completo (base ~250k; ~30-60 min na 3090)
docker compose -f experiments/e2e3/docker-compose.yml run --rm e2e3-gpu \
  experiments/e2e3/train_full.py --epochs 3 --batch-size 128
```

Os resultados aparecem em `experiments/e2e3/results/` no host.

## Sem GPU — alternativas em ordem de recomendação

1. **Google Colab (gratuito, T4)** — o notebook
   `experiments/e2e3/bertimbau_colab_tpu.ipynb` já traz tudo embutido
   (instruções, clone, dados, treino). Uma T4 faz o E2 em ~1–2 h. Limite
   prático: sessões podem cair; o notebook salva checkpoints por época.
2. **Kaggle Notebooks (gratuito, 30 h/semana de P100/T4×2)** — mais previsível
   que o Colab para execuções longas. Suba `data/dataset.csv` como *dataset*
   privado e cole as células do mesmo notebook.
3. **GPU spot por hora (vast.ai / RunPod)** — RTX 3090/4090 por
   ~US\$ 0,20–0,40/h; E2 + E3 reduzido ≈ US\$ 3–8 no total. Estas plataformas
   consomem exatamente uma imagem Docker: a imagem deste diretório é a unidade
   de execução (`docker build` + `docker push` para o Docker Hub, aponte a
   plataforma para a imagem e rode os mesmos comandos acima).
4. **Cluster da universidade** — o C3SL/UFPR mantém infraestrutura com GPU;
   vale uma mensagem ao orientador. A imagem Docker também é o formato que
   ambientes com Singularity/Apptainer aceitam (`apptainer build img.sif
   docker://...`).
5. **CPU pura (último recurso, só E2)** — medido no sandbox (4 vCPU):
   ~11,5 docs/s por época → base cheia (250k × 3 épocas) ≈ 18 h; num desktop
   de 8–16 núcleos, 5–10 h. Viável para o E2 se rodar de madrugada:

   ```bash
   docker compose -f experiments/e2e3/docker-compose.yml build e2e3-cpu
   docker compose -f experiments/e2e3/docker-compose.yml run --rm e2e3-cpu \
     experiments/e2e3/train_full.py --epochs 3 --batch-size 32
   ```

   O E3 (laço com re-treinos) em CPU levaria dias — não recomendado.

## Notas

- O volume `hf-cache` evita rebaixar o modelo a cada execução
  (`docker volume rm` para limpar). Para aproveitar um cache HuggingFace já
  existente no host: acrescente `-v ~/.cache/huggingface:/cache/hf` ao
  `docker run`.
- Atrás de proxy corporativo com interceptação TLS, injete o certificado da
  empresa no build (`COPY ca.crt ...` + `ENV PIP_CERT=...`) — o Dockerfile
  publicado assume rede limpa.
- A variante GPU usa a base `nvidia/cuda:12.4.1-runtime` + wheel `cu124`
  (cobre da série 20xx à 40xx; a 3090 inclusive).
