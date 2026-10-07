from fastapi import status


class DomainError(Exception):
    """Base exception for all domain errors."""

    def __init__(self, message: str = "Domain error occurred", details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class NotFoundError(DomainError):
    """Raised when a requested resource is not found."""

    def __init__(self, resource: str, identifier: str = None, message: str = None):
        self.resource = resource
        self.identifier = identifier
        msg = message or f"{resource} not found"
        details = {"resource": resource}
        if identifier:
            details["identifier"] = str(identifier)
        super().__init__(msg, details)


class DuplicateError(DomainError):
    """Raised when attempting to create a duplicate resource."""

    def __init__(self, resource: str, field: str, value: str):
        self.resource = resource
        self.field = field
        self.value = value
        msg = f"{resource} with {field}='{value}' already exists"
        details = {"resource": resource, "field": field, "value": str(value)}
        super().__init__(msg, details)


class RateLimitError(DomainError):
    """MLOPS_TODO #9: 429 Too Many Requests.

    Carries the Retry-After value the client should wait before retrying.
    """

    def __init__(self, retry_after: int = 1, key: str = ""):
        self.retry_after = retry_after
        msg = f"Rate limit exceeded. Retry after {retry_after}s."
        details: dict = {"retry_after": retry_after}
        if key:
            details["key_hint"] = key[:16]  # don't leak full key
        super().__init__(msg, details)


class ValidationError(DomainError):
    """
    Raised when input values validation fails. 422 Unprocessable Entity for
      - Values validation errors
      - Invalid field values
      - Business rule violations
      - Schema validation failures
    """

    def __init__(self, field: str, message: str = None, details: dict = None):
        self.field = field
        self.message = message or f"Invalid value for {field}"
        details = details or {}
        details["field"] = field
        super().__init__(self.message, details)


class BadRequestError(DomainError):
    """
    Raised when input parsing validation fails. 400 Bad Request for
      - Malformed JSON
      - Missing required HTTP headers (if required for parsing)
      - Invalid query/path parameter format (e.g. /users/abc where id must be int)
      - Invalid Content-Type
    """

    def __init__(self, field: str, message: str = None, details: dict = None):
        self.field = field
        self.message = message or f"Invalid value for {field}"
        details = details or {}
        details["field"] = field
        super().__init__(self.message, details)


class AuthenticationError(DomainError):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, {})


class AuthorizationError(DomainError):
    """Raised when user lacks permission for an action."""

    def __init__(self, action: str = "perform this action", resource: str = None):
        self.action = action
        self.resource = resource
        msg = f"Not authorized to {action}"
        details = {"action": action}
        if resource:
            details["resource"] = resource
        super().__init__(msg, details)


class DatabaseConnectionError(DomainError):
    """Raise when there is a database ConnectionRefusedError."""

    def __init__(self, message="Database connection issue", exception_msg: str = None):
        self.message = message
        if exception_msg:
            details = {"exception msg": exception_msg}
        super().__init__(message, details)


class GeneratorError(DomainError):
    """M0-2 Generator seam: the LLM/upstream failed to produce a completion.

    Every provider normalizes its own failure modes (auth, network, HTTP,
    empty response) to this single type so the HTTP layer maps it to a stable
    502 Bad Gateway problem+json instead of leaking vendor-specific exceptions
    (openai.APIStatusError, httpx.ReadTimeout, ...). `details` carries
    non-sensitive context (e.g. the model name), never the raw upstream error.
    """

    def __init__(self, message: str = "Generator error", details: dict = None):
        super().__init__(message, details or {})


# HTTP status code mapping for domain errors
DOMAIN_ERROR_STATUS_MAP = {
    BadRequestError: status.HTTP_400_BAD_REQUEST,
    AuthenticationError: status.HTTP_401_UNAUTHORIZED,
    AuthorizationError: status.HTTP_403_FORBIDDEN,
    NotFoundError: status.HTTP_404_NOT_FOUND,
    DuplicateError: status.HTTP_409_CONFLICT,
    ValidationError: status.HTTP_422_UNPROCESSABLE_CONTENT,
    DatabaseConnectionError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    RateLimitError: status.HTTP_429_TOO_MANY_REQUESTS,
    GeneratorError: status.HTTP_502_BAD_GATEWAY,
}


def get_http_status_code(error: DomainError) -> int:
    """Get the HTTP status code for a domain error."""
    for error_class, status_code in DOMAIN_ERROR_STATUS_MAP.items():
        if isinstance(error, error_class):
            return status_code
    return status.HTTP_500_INTERNAL_SERVER_ERROR
