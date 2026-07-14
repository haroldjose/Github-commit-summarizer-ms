"""Validación de JWT emitidos por Keycloak (OAuth2 Resource Server).

Replica el comportamiento de los microservicios Java de Mini-GitHub:
- verifica firma contra el JWKS del realm (RS256)
- verifica el issuer
- extrae los roles de realm_access.roles
"""
import logging

import jwt
from fastapi import Depends, HTTPException, Request, status
from jwt import PyJWKClient

from app.config import get_settings

log = logging.getLogger(__name__)
_jwks_client: PyJWKClient | None = None


def _get_jwks_client() -> PyJWKClient:
    global _jwks_client
    if _jwks_client is None:
        _jwks_client = PyJWKClient(get_settings().jwt_jwk_set_uri, cache_keys=True)
    return _jwks_client


async def require_jwt(request: Request) -> dict:
    """Dependencia FastAPI: exige un Bearer token válido y devuelve sus claims."""
    settings = get_settings()
    if not settings.app_security_oauth2_enabled:
        return {"preferred_username": "anonimo", "realm_access": {"roles": []}}

    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Falta el token Bearer")
    token = auth.removeprefix("Bearer ").strip()
    try:
        signing_key = _get_jwks_client().get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            issuer=settings.jwt_issuer_uri,
            options={"verify_aud": False},  # igual que los servicios Java
        )
        return claims
    except jwt.PyJWTError as exc:
        log.warning("Token rechazado: %s", exc)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Token inválido") from exc


CurrentUser = Depends(require_jwt)
