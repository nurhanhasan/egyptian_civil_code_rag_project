import logging
import time
import uuid
from typing import Awaitable, Callable

from fastapi import Request, Response
from fastapi.concurrency import iterate_in_threadpool
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ragapp.core.context import request_id_ctx_var

logger = logging.getLogger(__name__)

PROBLEM_JSON_MEDIA_TYPE = "application/problem+json"


def _build_500_problem_response(request_id: str) -> JSONResponse:
    """Build a RFC 7807 problem+json response for 500 errors."""
    return JSONResponse(
        status_code=500,
        content={
            "type": "https://egyptian-civil-code-rag.example.com/problems/internal-server-error",
            "title": "Internal Server Error",
            "status": 500,
            "detail": "An unexpected error occurred",
            "instance": request_id,
        },
        media_type=PROBLEM_JSON_MEDIA_TYPE,
    )


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """
        Middleware to log incoming requests and outgoing responses, including errors.
        """
        request_id = str(uuid.uuid4())
        request_id_ctx_var.set(request_id)

        start_time = time.time()

        # FastAPI/Starlette responses (especially StreamingResponse)
        # Do NOT expose the body natively.
        # Extract response body for logging (if needed, specially for ERRORs)
        response = await call_next(request)
        try:
            res_body = [section async for section in response.body_iterator]
            response.body_iterator = iterate_in_threadpool(iter(res_body))
            res_body = res_body[0].decode()
        except Exception as e:
            logger.error(f"Error reading response body: {e}")
            res_body = "<Could not read response body>"

        process_time = time.time() - start_time

        if response.status_code == 500:
            logger.error(
                "Internal server error",
                exc_info=True,
                extra={
                    "request_id": request_id,
                    "http": {
                        "method": request.method,
                        "path": request.url.path,
                        "status": response.status_code,
                        "error": res_body,
                        "duration": process_time,
                    },
                },
            )

            return _build_500_problem_response(request_id)

        logger.info(
            "request_completed",
            extra={
                "request_id": request_id,
                "http": {
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration": process_time,
                },
            },
        )

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)
        return response
