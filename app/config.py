"""Configuración del servicio vía variables de entorno (12-factor).

Nombres alineados con los demás microservicios de Mini-GitHub
(JWT_ISSUER_URI / JWT_JWK_SET_URI, igual que organizations-ms).
"""
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "commit-summarizer-ms"
    server_port: int = 8096

    # Seguridad (mismo patrón de toggle que files-ms/pullrequests-ms)
    app_security_oauth2_enabled: bool = True
    jwt_issuer_uri: str = "http://localhost:8080/realms/Github"
    jwt_jwk_set_uri: str = (
        "http://localhost:8080/realms/Github/protocol/openid-connect/certs"
    )

    # Artefactos de los modelos (Prompt 1)
    models_dir: str = "models"

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
