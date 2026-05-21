"""Tests for src.logging_config.setup_logging().

Verifies idempotency, level propagation, and consistent format so that
all entry points (CLI, scripts, python -m researcher) produce uniform logs.
"""
from __future__ import annotations

import logging
import sys


def _reload_logging_config():
    """Return a freshly-imported logging_config with _configured reset to False."""
    mod_name = "src.logging_config"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    import src.logging_config as lc
    return lc


def test_setup_logging_configures_root_level():
    lc = _reload_logging_config()
    lc.setup_logging(level="WARNING")
    assert logging.getLogger().level == logging.WARNING
    # cleanup
    lc._configured = False
    logging.getLogger().setLevel(logging.WARNING)


def test_setup_logging_idempotent(caplog):
    """Calling setup_logging() twice must not add a second handler."""
    lc = _reload_logging_config()
    root = logging.getLogger()

    lc.setup_logging(level="DEBUG")
    after_first = len(root.handlers)

    lc.setup_logging(level="ERROR")  # second call — must be a no-op
    after_second = len(root.handlers)

    # No additional handlers added by the second call
    assert after_second == after_first
    # Level was NOT changed by the second call
    assert root.level == logging.DEBUG

    # cleanup
    lc._configured = False


def test_setup_logging_default_is_info():
    lc = _reload_logging_config()
    lc.setup_logging()  # no level arg
    assert logging.getLogger().level == logging.INFO
    lc._configured = False


def test_setup_logging_case_insensitive():
    lc = _reload_logging_config()
    lc.setup_logging(level="debug")  # lowercase
    assert logging.getLogger().level == logging.DEBUG
    lc._configured = False
