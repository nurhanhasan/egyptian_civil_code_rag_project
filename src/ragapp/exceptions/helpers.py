import logging
import uuid
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse

from ragapp.core.context import request_id_ctx_var

logger = logging.getLogger(__name__)

PROBLEM_JSON_MEDIA_TYPE = "application/problem+json"


def _get_request_id(request: Request) -> str:
    """Get or generate request ID for trace correlation."""
    request_id = request_id_ctx_var.get()
    if request_id is None:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    return request_id


def _sanitize_detail(message: str, details: dict) -> str:
    """Return a sanitized human-readable detail per RFC 7807 / SPEC.md §3.4.
    No stack traces, no PII, no internal structure.
    """
    return message


def _log_error(
    request: Request,
    status_code: int,
    title: str,
    message: str,
    details: dict,
) -> None:
    """Structured error log — one call per error response.
    4xx → warning, 5xx → error.
    """
    level = logging.ERROR if status_code >= 500 else logging.WARNING
    logger.log(
        level,
        "error_response",
        extra={
            "error": {
                "request_id": _get_request_id(request),
                "status_code": status_code,
                "title": title,
                "message": message,
                "path": request.url.path,
                "method": request.method,
                **details,
            },
        },
    )


def _build_problem_response(
    request: Request,
    status_code: int,
    title: str,
    detail: str,
    type_uri: str,
    extra: Optional[dict] = None,
) -> JSONResponse:
    """
    Build an RFC 7807 problem+json response.
    """
    request_id = _get_request_id(request)
    body = {
        "type": type_uri,
        "title": title,
        "status": status_code,
        "detail": detail,
        "instance": request_id,
    }
    if extra:
        body.update(extra)

    return JSONResponse(
        status_code=status_code,
        content=body,
        media_type=PROBLEM_JSON_MEDIA_TYPE,
    )


def _type_uri(error_name: str) -> str:
    """Generate a problem type URI. In production, use a stable project namespace."""
    return f"https://egyptian-civil-code-rag.example.com/problems/{error_name.lower()}"
