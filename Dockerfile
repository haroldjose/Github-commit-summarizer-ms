FROM python:3.12-slim

LABEL org.opencontainers.image.title="Github-commit-summarizer-ms" \
    org.opencontainers.image.description="Microservicio de IA: resumen automatico de diffs de commits" \
    org.opencontainers.image.vendor="Githubx"

WORKDIR /app

COPY requirements.txt .
# torch desde el indice CPU (sin CUDA): ~200 MB en vez de ~2 GB
RUN pip install --no-cache-dir torch~=2.6 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app
# Modelos versionados junto con la imagen (MLOps basico)
# v1 (produccion, default) y v2 (experimental, MODEL_VERSION=v2)
COPY training/artefactos ./models
COPY training/artefactos_v2 ./models_v2

RUN useradd --system --uid 1001 appuser
USER appuser

EXPOSE 8096
ENV SERVER_PORT=8096

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${SERVER_PORT}"]
