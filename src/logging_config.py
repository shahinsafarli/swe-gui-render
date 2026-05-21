"""
Structured logging configuration for the Async Research Assistant.

Provides a single ``setup_logging()`` function that configures the root
logger exactly once regardless of how many entry points call it.

Usage
-----
Call early in any entry point (CLI, scripts, ``python -m researcher``):

    from src.logging_config import setup_logging
    setup_logging()                   # INFO by default
    setup_logging(level="DEBUG")      # or override the level
    setup_logging(level="WARNING")    # quiet for benchmarks

The CLI (``src/cli.py``) continues to pass ``--log-level`` through the
Click option; it calls this function instead of ``logging.basicConfig``
directly so the format stays consistent everywhere.

Design
------
- Idempotent: a module-level ``_configured`` flag prevents double-init.
  Calling ``setup_logging()`` from multiple entry points or test helpers
  is safe — only the first call takes effect.
- The log format is identical to the format previously hard-coded in the
  CLI group callback, so existing log consumers are unaffected.
- No third-party dependencies (structlog, loguru, etc.) — plain stdlib.
"""
from __future__ import annotations

import logging

_configured = False

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger once.

    Subsequent calls are no-ops so importing this from multiple entry
    points never produces duplicate handlers or format changes.

    Parameters
    ----------
    level:
        One of ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR`` (case-insensitive).
        Defaults to ``INFO``.
    """
    global _configured
    if _configured:
        return

    logging.basicConfig(
        level=level.upper(),
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
    )
    # Explicitly set root level in case basicConfig was already called
    # by a third-party library before us (basicConfig is a no-op then,
    # but setLevel is not).
    logging.getLogger().setLevel(level.upper())
    _configured = True
