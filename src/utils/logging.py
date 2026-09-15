"""src/utils/logging.py — centralised logging configuration.

All modules in this project import `get_logger` from here so that
log format, level, and handlers are controlled from one place.
"""
from __future__ import annotations

import logging
import sys
from typing import Optional


_LOG_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured: bool = False


def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> None:
    """Configure root logger once for the whole process.

    Safe to call multiple times — only the first call takes effect.
    """
    global _configured
    if _configured:
        return
    _configured = True

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stderr),
    ]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        handlers=handlers,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, ensuring root config exists."""
    configure_logging()  # no-op if already configured
    return logging.getLogger(name)
