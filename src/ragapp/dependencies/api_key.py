import hmac
import logging

from fastapi import Depends, HTTPException, Request

from ragapp.core.config import get_settings

logger = logging.getLogger(__name__)

WWW_AUTHENTICATE = "Bearer"


def _authorized(request: Request, expected: str) -> bool:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return False
    token = auth[len("Bearer ") :]
    return hmac.compare_digest(token, expected)


def require_api_key(request: Request, settings=Depends(get_settings)) -> None:
    if not settings.API_KEY_ENABLED:
        return
    if not _authorized(request, settings.API_KEY):
        logger.warning(
            "api_key_rejected",
            extra={
                "request_id": request.headers.get("X-Request-ID"),
                "http": {
                    "method": request.method,
                    "path": request.url.path,
                    "status": 401,
                },
            },
        )
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": WWW_AUTHENTICATE},
        )
