"""Logging setup.

One configuration function, called once per process entry point. Library modules
only ever call ``get_logger(__name__)`` so that importing this package never has
a side effect on someone else's logging configuration.
"""

from __future__ import annotations

import logging
import sys
from typing import Final

_LOG_FORMAT: Final = "%(asctime)s | %(levelname)-8s | %(name)-38s | %(message)s"
_DATE_FORMAT: Final = "%Y-%m-%d %H:%M:%S"

# Third-party loggers that are noisy at INFO and add nothing at this scale.
_QUIET_LOGGERS: Final = (
    "httpx",
    "httpcore",
    "chromadb",
    "chromadb.telemetry",
    "urllib3",
    "watchdog",
)

_is_configured = False


def configure_logging(level: str = "INFO", *, force: bool = False) -> None:
    """Configure root logging for an entry point (API, watcher, CLI, tests).

    Idempotent: calling it from both ``run_backend.py`` and a service module
    must not double every log line.
    """
    global _is_configured
    if _is_configured and not force:
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    for existing_handler in list(root_logger.handlers):
        root_logger.removeHandler(existing_handler)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root_logger.addHandler(stream_handler)

    for noisy_logger_name in _QUIET_LOGGERS:
        logging.getLogger(noisy_logger_name).setLevel(logging.WARNING)

    _is_configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
