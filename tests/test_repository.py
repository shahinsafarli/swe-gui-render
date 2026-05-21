"""Tests for repository graceful degradation when the DB pool is unavailable.

These tests verify that ``save_session``, ``get_sessions``, and ``get_session``
never raise when ``_get_pool()`` returns ``None`` — they must degrade to
``None`` / ``[]`` instead of propagating ``AttributeError``.
"""
from __future__ import annotations

import pytest

from src.storage.repository import ResearchRepository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_repo() -> ResearchRepository:
    return ResearchRepository()


# ---------------------------------------------------------------------------
# Pool-is-None guards (the core regression)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_session_returns_none_when_no_pool(monkeypatch):
    """save_session must return None, not raise, when the DB pool is unavailable."""
    import src.storage.repository as repo_mod

    monkeypatch.setattr(repo_mod, "_get_pool", _async_none)

    result = await _make_repo().save_session(
        question="q",
        answer="a",
        citations=[],
        sources_failed=[],
        cached=False,
        elapsed_seconds=0.1,
    )
    assert result is None


@pytest.mark.asyncio
async def test_get_sessions_returns_empty_list_when_no_pool(monkeypatch):
    """get_sessions must return [], not raise, when the DB pool is unavailable."""
    import src.storage.repository as repo_mod

    monkeypatch.setattr(repo_mod, "_get_pool", _async_none)

    result = await _make_repo().get_sessions()
    assert result == []


@pytest.mark.asyncio
async def test_get_session_returns_none_when_no_pool(monkeypatch):
    """get_session must return None, not raise, when the DB pool is unavailable."""
    import src.storage.repository as repo_mod

    monkeypatch.setattr(repo_mod, "_get_pool", _async_none)

    result = await _make_repo().get_session(session_id=1)
    assert result is None


# ---------------------------------------------------------------------------
# Exception-in-pool guard (double safety: pool exists but query raises)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_session_returns_none_on_db_error(monkeypatch):
    """save_session must return None, not raise, when a DB error occurs."""
    import src.storage.repository as repo_mod

    monkeypatch.setattr(repo_mod, "_get_pool", _async_raises)

    result = await _make_repo().save_session(
        question="q",
        answer="a",
        citations=[],
        sources_failed=[],
        cached=False,
        elapsed_seconds=0.0,
    )
    assert result is None


@pytest.mark.asyncio
async def test_get_sessions_returns_empty_list_on_db_error(monkeypatch):
    import src.storage.repository as repo_mod

    monkeypatch.setattr(repo_mod, "_get_pool", _async_raises)

    result = await _make_repo().get_sessions()
    assert result == []


@pytest.mark.asyncio
async def test_get_session_returns_none_on_db_error(monkeypatch):
    import src.storage.repository as repo_mod

    monkeypatch.setattr(repo_mod, "_get_pool", _async_raises)

    result = await _make_repo().get_session(session_id=42)
    assert result is None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


async def _async_none():
    """Simulates _get_pool() returning None (DB not configured)."""
    return None


async def _async_raises():
    """Simulates _get_pool() raising an unexpected error."""
    raise RuntimeError("simulated DB connection failure")
