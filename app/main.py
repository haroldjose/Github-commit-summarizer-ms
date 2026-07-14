"""commit-summarizer-ms — microservicio de IA de Mini-GitHub.

Genera un mensaje de commit en lenguaje natural a partir de un diff.
Modelo seq2seq entrenado desde cero (ver docs/ENTRENAMIENTO.md).
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.config import get_settings
from app.security.jwt_auth import CurrentUser
from app.services.summarizer_service import SummarizerService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)-5s %(name)s - %(message)s",
)
log = logging.getLogger(__name__)

servicio: SummarizerService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global servicio
    servicio = SummarizerService(get_settings().models_dir)
    yield


app = FastAPI(
    title="Mini-GitHub Commit Summarizer API",
    version="1.0.0",
    description="Resumen automático de cambios de código (diff → mensaje)",
    lifespan=lifespan,
)


class SummarizeRequest(BaseModel):
    diff: str = Field(min_length=1, max_length=200_000)


class SummarizeResponse(BaseModel):
    resumen: str


@app.get("/actuator/health", tags=["infra"])
def health() -> dict:
    return {"status": "UP"}


@app.post("/v1/summarize", response_model=SummarizeResponse, tags=["resumen"])
def summarize(req: SummarizeRequest, claims: dict = CurrentUser) -> SummarizeResponse:
    resumen = servicio.resumir(req.diff)
    log.info("summarize por %s (%d chars de diff)",
             claims.get("preferred_username", "?"), len(req.diff))
    return SummarizeResponse(resumen=resumen)
