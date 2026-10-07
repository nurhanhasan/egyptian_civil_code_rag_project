import json
import logging
import sys

from ragapp.core.config import get_settings
from ragapp.core.context import request_id_ctx_var

settings = get_settings()


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx_var.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", None),
            "http": getattr(record, "http", None),
            "timestamp": self.formatTime(record),
        }
        return json.dumps(log_record)


LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def setup_logging():
    handler = logging.StreamHandler(sys.stdout)

    handler.setFormatter(JsonFormatter())
    handler.addFilter(RequestIdFilter())

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    root.setLevel(LOG_LEVEL_MAP.get(settings.LOGGING_LEVEL.upper(), logging.INFO))

    # remove default handlers
    root.handlers.clear()

    root.addHandler(handler)
