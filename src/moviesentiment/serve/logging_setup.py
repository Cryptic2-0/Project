"""Structured logging with request-id correlation.

Every log line from a request handler carries the request_id bound at the middleware,
so tokenize/inference/prom/log rows share a single trace identifier without manual plumbing.
"""

from __future__ import annotations

import logging
import re
import sys
import uuid

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars

log = structlog.get_logger()

# Sanitize incoming x-request-id to prevent log injection (CRLF, control chars, JSON-
# breaking quotes) and bound length so a hostile client cannot grow log volume.
_RID_ALLOWED = re.compile(r"[^A-Za-z0-9._-]")
_RID_MAX_LEN = 64


def _sanitize_rid(s: str) -> str:
    cleaned = _RID_ALLOWED.sub("", s)[:_RID_MAX_LEN]
    return cleaned or uuid.uuid4().hex[:12]


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def bind_request_id(incoming: str | None) -> str:
    rid = _sanitize_rid(incoming) if incoming else uuid.uuid4().hex[:12]
    clear_contextvars()
    bind_contextvars(request_id=rid)
    return rid
