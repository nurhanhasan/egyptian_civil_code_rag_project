import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ragapp.exceptions.domain import (
    AuthenticationError,
    AuthorizationError,
    BadRequestError,
    DatabaseConnectionError,
    DomainError,
    DuplicateError,
    GeneratorError,
    NotFoundError,
    RateLimitError,
    ValidationError,
    get_http_status_code,
)
from ragapp.exceptions.helpers import (
    _build_problem_response,
    _log_error,
    _sanitize_detail,
    _type_uri,
)

logger = logging.getLogger(__name__)


async def domain_error_handler(request: Request, exc: DomainError) -> JSONResponse:
    status_code = get_http_status_code(exc)
    detail = _sanitize_detail(exc.message, exc.details)
    _log_error(request, status_code, exc.__class__.__name__, detail, exc.details)
    return _build_problem_response(
        request=request,
        status_code=status_code,
        title=exc.__class__.__name__,
        detail=detail,
        type_uri=_type_uri(exc.__class__.__name__),
    )


async def not_found_error_handler(request: Request, exc: NotFoundError) -> JSONResponse:
    detail = _sanitize_detail(exc.message, exc.details)
    _log_error(request, 404, "NotFoundError", detail, exc.details)
    return _build_problem_response(
        request=request,
        status_code=404,
        title="NotFoundError",
        detail=detail,
        type_uri=_type_uri("NotFoundError"),
    )


async def duplicate_error_handler(
    request: Request, exc: DuplicateError
) -> JSONResponse:
    detail = _sanitize_detail(exc.message, exc.details)
    _log_error(request, 409, "DuplicateError", detail, exc.details)
    return _build_problem_response(
        request=request,
        status_code=409,
        title="DuplicateError",
        detail=detail,
        type_uri=_type_uri("DuplicateError"),
    )


async def validation_error_handler(
    request: Request, exc: ValidationError
) -> JSONResponse:
    detail = _sanitize_detail(exc.message, exc.details)
    _log_error(request, 422, "ValidationError", detail, exc.details)
    return _build_problem_response(
        request=request,
        status_code=422,
        title="ValidationError",
        detail=detail,
        type_uri=_type_uri("ValidationError"),
    )


async def request_validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    error = exc.errors()[0]
    field = ".".join(str(part) for part in error["loc"])
    detail = f"Invalid value for field '{field}': {error['msg']}"
    _log_error(request, 422, "ValidationError", detail, {"field": field})
    return _build_problem_response(
        request=request,
        status_code=422,
        title="ValidationError",
        detail=detail,
        type_uri=_type_uri("ValidationError"),
    )


async def bad_request_error_handler(
    request: Request, exc: BadRequestError
) -> JSONResponse:
    detail = _sanitize_detail(exc.message, exc.details)
    _log_error(request, 400, "BadRequestError", detail, exc.details)
    return _build_problem_response(
        request=request,
        status_code=400,
        title="BadRequestError",
        detail=detail,
        type_uri=_type_uri("BadRequestError"),
    )


async def authentication_error_handler(
    request: Request, exc: AuthenticationError
) -> JSONResponse:
    detail = _sanitize_detail(exc.message, {})
    _log_error(request, 401, "AuthenticationError", detail, {})
    return _build_problem_response(
        request=request,
        status_code=401,
        title="AuthenticationError",
        detail=detail,
        type_uri=_type_uri("AuthenticationError"),
        extra={"headers": {"WWW-Authenticate": "Bearer"}},
    )


async def authorization_error_handler(
    request: Request, exc: AuthorizationError
) -> JSONResponse:
    detail = _sanitize_detail(exc.message, exc.details)
    _log_error(request, 403, "AuthorizationError", detail, exc.details)
    return _build_problem_response(
        request=request,
        status_code=403,
        title="AuthorizationError",
        detail=detail,
        type_uri=_type_uri("AuthorizationError"),
    )


async def database_connection_error_handler(
    request: Request, exc: DatabaseConnectionError
) -> JSONResponse:
    detail = "A database connection error occurred"
    _log_error(request, 500, "Internal Server Error", detail, {})
    return _build_problem_response(
        request=request,
        status_code=500,
        title="Internal Server Error",
        detail=detail,
        type_uri=_type_uri("DatabaseConnectionError"),
    )


async def generator_error_handler(
    request: Request, exc: GeneratorError
) -> JSONResponse:
    # Upstream LLM failure -> 502. The detail is fixed so the raw upstream
    # error (possibly vendor-specific, possibly with PII in a prompt) never
    # reaches the client; the full context is in the server log.
    detail = "The generator service is unavailable or failed to produce a response"
    _log_error(request, 502, "GeneratorError", detail, exc.details)
    return _build_problem_response(
        request=request,
        status_code=502,
        title="GeneratorError",
        detail=detail,
        type_uri=_type_uri("GeneratorError"),
    )


async def rate_limit_error_handler(
    request: Request, exc: RateLimitError
) -> JSONResponse:
    detail = _sanitize_detail(exc.message, exc.details)
    _log_error(request, 429, "RateLimitError", detail, exc.details)
    response = _build_problem_response(
        request=request,
        status_code=429,
        title="RateLimitError",
        detail=detail,
        type_uri=_type_uri("RateLimitError"),
    )
    response.headers["Retry-After"] = str(exc.retry_after)
    return response


def register_exception_handlers(app: FastAPI):
    app.add_exception_handler(DomainError, domain_error_handler)
    app.add_exception_handler(NotFoundError, not_found_error_handler)
    app.add_exception_handler(DuplicateError, duplicate_error_handler)
    app.add_exception_handler(ValidationError, validation_error_handler)
    app.add_exception_handler(RequestValidationError, request_validation_error_handler)
    app.add_exception_handler(BadRequestError, bad_request_error_handler)
    app.add_exception_handler(AuthenticationError, authentication_error_handler)
    app.add_exception_handler(AuthorizationError, authorization_error_handler)
    app.add_exception_handler(
        DatabaseConnectionError, database_connection_error_handler
    )
    app.add_exception_handler(RateLimitError, rate_limit_error_handler)
    app.add_exception_handler(GeneratorError, generator_error_handler)
    logger.info("Registered domain exception handlers")
