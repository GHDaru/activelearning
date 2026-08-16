# FastAPI do FlowBuilder (backend completo — spec 004) para host de contêiner.
#
# Não confundir com experiments/e2e3/Dockerfile: aquele é a imagem dedicada ao
# experimento BERTimbau (GPU, torch/transformers). Esta serve a API que executa
# runs de aprendizado ativo e experimentos com oráculo — CPU, sem torch.
#
# Build (na raiz do repositório):
#   docker build -t falco-flowbuilder-api .
# Uso: docs/e2e3-docker.md cobre o experimento; DEPLOY_AND_PUBLISH.md §1.4
# cobre esta imagem.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Camada de dependências primeiro (cache do build não invalida a cada mudança
# de código): só o necessário para a API + oráculos, sem torch/transformers
# (esses ficam na imagem dedicada do E2/E3, não aqui).
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install ".[api,oracles]"

# experiments/ entra por bind mount em produção (dados + artefatos não cabem
# na imagem); aqui só o suficiente para o catálogo default não quebrar o build.
COPY experiments/e0/config.json experiments/e0/config.json

EXPOSE 8000

# Sem --reload: essa flag é de desenvolvimento (recarrega a cada mudança de
# arquivo, o que não existe numa imagem imutável) e além de inútil aqui
# custa memória com o watcher de arquivos.
CMD ["sh", "-c", "uvicorn --factory activelearning.adapters.api.app:create_app --host 0.0.0.0 --port ${PORT}"]
