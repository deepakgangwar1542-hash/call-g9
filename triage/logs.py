"""Structured JSON logging with a correlation ID.

Rules: log event names, reason codes, counts, durations and identifiers. Never log call inputs, transcripts,
audio, API keys or exception messages that may echo user content (use `reason_code` instead).
"""
from __future__ import annotations

import contextvars
import datetime as dt
import json
import logging
import re
import uuid

_correlation: contextvars.ContextVar[str | None] = contextvars.ContextVar("correlation_id", default=None)
_VALID_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")
_RESERVED = set(logging.LogRecord("x", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}
logger = logging.getLogger("triage")


def new_correlation_id() -> str:
    return uuid.uuid4().hex


def sanitize_correlation_id(candidate: str | None) -> str:
    """Accept a caller supplied ID only when it is short and boring; otherwise mint one."""
    return candidate if candidate and _VALID_ID.match(candidate) else new_correlation_id()


def set_correlation_id(value: str | None) -> contextvars.Token:
    return _correlation.set(value)


def reset_correlation_id(token: contextvars.Token) -> None:
    _correlation.reset(token)


def current_correlation_id() -> str:
    """The active ID, minting one (and keeping it for this context) when none is set."""
    value = _correlation.get()
    if value is None:
        value = new_correlation_id()
        _correlation.set(value)
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": dt.datetime.fromtimestamp(record.created, dt.timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "event": record.getMessage(),
            "correlation_id": _correlation.get(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                entry[key] = value
        return json.dumps(entry, default=str, separators=(",", ":"))


def configure_logging(level: str = "INFO") -> None:
    """Idempotent: installs one JSON handler on the `triage` logger."""
    logger.setLevel(level)
    if not any(getattr(h, "_triage_json", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        handler._triage_json = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
        logger.propagate = False


def log_event(event: str, level: int = logging.INFO, **fields) -> None:
    logger.log(level, event, extra=fields)
