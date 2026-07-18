# BERTimbau E2/E3 — imagem dedicada ao experimento (GPU por padrão; CPU via build args).
#
# Build (na raiz do repositório):
#   GPU:  docker compose -f experiments/e2e3/docker-compose.yml build e2e3-gpu
#   CPU:  docker compose -f experiments/e2e3/docker-compose.yml build e2e3-cpu
# Uso: ver docs/e2e3-docker.md
#
# A imagem NÃO contém dados nem resultados: data/ e results/ entram por volume,
# e o modelo HuggingFace (440 MB) fica num volume de cache reutilizado entre
# execuções (HF_HOME=/cache/hf).

ARG BASE_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

# cu124 na GPU; "cpu" no build CPU (ver docker-compose.yml)
ARG TORCH_CHANNEL=cu124

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/hf

# A base CUDA não traz Python; a base python:*-slim já traz (o if evita o apt lá)
RUN if ! command -v python3 >/dev/null 2>&1; then \
      apt-get update && \
      apt-get install -y --no-install-recommends python3 python3-pip ca-certificates && \
      rm -rf /var/lib/apt/lists/*; \
    fi

RUN python3 -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/${TORCH_CHANNEL} && \
    python3 -m pip install --no-cache-dir transformers scikit-learn

WORKDIR /app
COPY src/ src/
COPY experiments/e2e3/*.py experiments/e2e3/

# Pontos de montagem: dados (somente leitura), resultados e cache do modelo
VOLUME ["/app/data", "/app/experiments/e2e3/results", "/cache"]

ENTRYPOINT ["python3"]
# Padrão: smoke test (2-4 min em CPU) — valida a cadeia inteira
CMD ["experiments/e2e3/run_smoke_cpu.py", "--classes", "30", "--per-class", "30", "--epochs", "3"]
