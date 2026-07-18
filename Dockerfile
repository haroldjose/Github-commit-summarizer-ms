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
# Modelo multilenguaje v3 versionado junto con la imagen (MLOps basico):
# vocabularios + corpus de recuperacion + checkpoint partido en .part-*
# (GitHub no acepta archivos >100 MB) que se reensambla aqui.
COPY training/artefactos_v3 ./models
RUN cat ./models/modelo_resumidor_ptrgen.pt.part-* > ./models/modelo_resumidor_ptrgen.pt \
    && rm ./models/modelo_resumidor_ptrgen.pt.part-*

RUN useradd --system --uid 1001 appuser
USER appuser

EXPOSE 8096
ENV SERVER_PORT=8096

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${SERVER_PORT}"]
