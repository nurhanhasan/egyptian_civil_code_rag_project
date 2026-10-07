"""Rate-limit dependency - MLOPS_TODO #9.

Mirrors require_api_key as a flag-gated edge dependency, but instead of
401 it emits 429 + Retry-After via RateLimitError (handled by
exceptions/handlers.py, mapped to HTTP 429).

The limiter's .allow() is blocking sync I/O (Redis), so it is bridged off the
event loop with asyncio.to_thread to avoid stalling other requests.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import Depends, Request

from ragapp.core.config import get_settings
from ragapp.core.context import request_id_ctx_var
from ragapp.core.rate_limit import get_rate_limiter
from ragapp.exceptions.domain import RateLimitError

logger = logging.getLogger(__name__)


def _client_key(request: Request) -> str:
    # Identity used to bucket requests. Default: client IP (distributed cap per
    # client). Override via a header so authenticated callers get their own
    # bucket (e.g. an API-key principal) instead of sharing an egress IP.
    principal = request.headers.get("X-Rate-Key")
    if principal:
        return f"principal:{principal}"
    client_ip = request.client.host if request.client else "anonymous"
    return f"ip:{client_ip}"


async def require_rate_limit(request: Request, settings=Depends(get_settings)) -> None:
    limiter = get_rate_limiter(settings)
    if limiter is None:
        return

    key = _client_key(request)

    decision = await asyncio.to_thread(limiter.allow, key)

    if not decision.allowed:
        logger.warning(
            "rate_limited",
            extra={
                "request_id": request_id_ctx_var.get(),
                "http": {
                    "method": request.method,
                    "path": request.url.path,
                    "status": 429,
                    "retry_after": decision.retry_after,
                },
            },
        )
        raise RateLimitError(retry_after=decision.retry_after, key=key)
