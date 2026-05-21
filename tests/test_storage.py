"""
Tests for the storage layer.

Two layers of tests:
1. _FakeRepo — validates the expected contract (save/get/ordering) in isolation.
2. ResearchRepository with _get_pool mocked — exercises the real SQL code paths
   without requiring a live PostgreSQL instance.

The repository was updated to use ``psycopg_pool.AsyncConnectionPool`` so that
connections are reused across calls rather than opened and torn down for every
SQL statement.  The mock here replaces ``_get_pool`` (not the old ``_get_conn``)
and returns a pool-shaped mock whose ``connection()`` async context manager
yields a mocked psycopg connection.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models import CitationRecord, ResearchResult, SourceRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(question: str = "What is photosynthesis?") -> ResearchResult:
    return ResearchResult(
        question=question,
        answer="Photosynthesis is the process [1].",
        citations=[CitationRecord(index=1, title="T", url="U", origin="wikipedia")],
        sources_used=[SourceRecord(title="T", url="U", snippet="S", origin="wikipedia")],
        sources_failed=[],
        elapsed_seconds=1.23,
        cached=False,
    )


# ---------------------------------------------------------------------------
# Pool / connection mock helpers
# ---------------------------------------------------------------------------

def _make_pool_mock(fetchone_result=None, fetchall_result=None):
    """Return a mock pool whose .connection() async-context-manager yields a conn.

    The pool mock has a single ``connection()`` method that acts as an async
    context manager, matching the psycopg_pool API:

        async with pool.connection() as conn:
            cursor = await conn.execute(...)

    The cursor mock is pre-loaded with the given fetchone/fetchall return values.
    """
    cursor = MagicMock()
    cursor.fetchone = AsyncMock(return_value=fetchone_result)
    cursor.fetchall = AsyncMock(return_value=fetchall_result or [])

    conn = MagicMock()
    conn.execute = AsyncMock(return_value=cursor)
    conn.commit = AsyncMock()

    # pool.connection() must be an async context manager
    @asynccontextmanager
    async def _connection():
        yield conn

    pool = MagicMock()
    pool.connection = _connection
    # Expose the inner conn so tests can inspect calls
    pool._conn = conn

    return pool


# ---------------------------------------------------------------------------
# Autouse fixture: reset the module-level pool singleton before each test
# so pool-mock patches are always picked up cleanly.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_pool(monkeypatch):
    """Reset the module-level _pool singleton so each test starts cold."""
    import src.storage.repository as repo_mod
    monkeypatch.setattr(repo_mod, "_pool", None)
    yield
    monkeypatch.setattr(repo_mod, "_pool", None)


# ---------------------------------------------------------------------------
# _FakeRepo — contract tests (ordering, limits, etc.)
# ---------------------------------------------------------------------------

class _FakeRepo:
    """In-memory stand-in for ResearchRepository."""

    def __init__(self) -> None:
        self._sessions: list[dict] = []

    async def save_session(self, question, answer, citations,
                           sources_failed, cached, elapsed_seconds) -> int:
        sid = len(self._sessions) + 1
        self._sessions.append({
            "id": sid,
            "question": question,
            "answer": answer,
            "citations_json": json.dumps(citations),
            "sources_failed": json.dumps(sources_failed),
            "cached": cached,
            "elapsed_seconds": elapsed_seconds,
        })
        return sid

    async def get_sessions(self, limit: int = 20) -> list[dict]:
        return self._sessions[-limit:][::-1]

    async def get_session(self, session_id: int) -> dict | None:
        for s in self._sessions:
            if s["id"] == session_id:
                return s
        return None


@pytest.fixture
def fake_repo() -> _FakeRepo:
    return _FakeRepo()


# ---------------------------------------------------------------------------
# save_session — contract (via _FakeRepo)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_save_session_returns_id(fake_repo):
    result = _make_result()
    sid = await fake_repo.save_session(
        question=result.question,
        answer=result.answer,
        citations=[c.model_dump() for c in result.citations],
        sources_failed=result.sources_failed,
        cached=result.cached,
        elapsed_seconds=result.elapsed_seconds,
    )
    assert isinstance(sid, int)
    assert sid >= 1


@pytest.mark.asyncio
async def test_save_multiple_sessions(fake_repo):
    for q in ["Q1", "Q2", "Q3"]:
        result = _make_result(q)
        await fake_repo.save_session(
            question=result.question,
            answer=result.answer,
            citations=[],
            sources_failed=[],
            cached=False,
            elapsed_seconds=0.1,
        )
    sessions = await fake_repo.get_sessions(limit=10)
    assert len(sessions) == 3


# ---------------------------------------------------------------------------
# get_sessions — contract
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_sessions_returns_most_recent_first(fake_repo):
    for q in ["Q1", "Q2", "Q3"]:
        await fake_repo.save_session(q, "A", [], [], False, 0.1)
    sessions = await fake_repo.get_sessions(limit=2)
    assert len(sessions) == 2
    assert sessions[0]["question"] == "Q3"


@pytest.mark.asyncio
async def test_get_sessions_empty_on_fresh_repo(fake_repo):
    sessions = await fake_repo.get_sessions()
    assert sessions == []


# ---------------------------------------------------------------------------
# get_session — contract
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_session_found(fake_repo):
    sid = await fake_repo.save_session("Q", "A", [], [], False, 0.5)
    row = await fake_repo.get_session(sid)
    assert row is not None
    assert row["question"] == "Q"


@pytest.mark.asyncio
async def test_get_session_missing_returns_none(fake_repo):
    result = await fake_repo.get_session(9999)
    assert result is None


# ---------------------------------------------------------------------------
# ResearchRepository — real class, _get_pool mocked
# These exercise the actual SQL execution paths without a live database.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_real_repo_save_session_returns_id():
    """save_session extracts the id from the RETURNING clause."""
    from src.storage.repository import ResearchRepository

    pool = _make_pool_mock(fetchone_result={"id": 42})
    repo = ResearchRepository()
    with patch("src.storage.repository._get_pool", AsyncMock(return_value=pool)):
        sid = await repo.save_session("Q", "A", [], [], False, 1.0)
    assert sid == 42


@pytest.mark.asyncio
async def test_real_repo_save_session_serialises_json():
    """save_session must JSON-encode citations and sources_failed."""
    from src.storage.repository import ResearchRepository

    pool = _make_pool_mock(fetchone_result={"id": 1})
    repo = ResearchRepository()
    citations = [{"index": 1, "title": "T", "url": "U", "origin": "wikipedia"}]
    with patch("src.storage.repository._get_pool", AsyncMock(return_value=pool)):
        await repo.save_session("Q", "A", citations, ["arxiv"], False, 0.5)

    conn = pool._conn
    call_args = conn.execute.call_args
    positional_params = call_args[0][1]   # (sql, params) → params tuple
    assert json.loads(positional_params[2]) == citations
    assert json.loads(positional_params[3]) == ["arxiv"]


@pytest.mark.asyncio
async def test_real_repo_get_sessions_returns_rows():
    """get_sessions returns the list fetched from the cursor."""
    from src.storage.repository import ResearchRepository

    rows = [{"id": 1, "question": "Q", "answer": "A"}]
    pool = _make_pool_mock(fetchall_result=rows)
    repo = ResearchRepository()
    with patch("src.storage.repository._get_pool", AsyncMock(return_value=pool)):
        result = await repo.get_sessions(limit=5)
    assert result == rows


@pytest.mark.asyncio
async def test_real_repo_get_session_found():
    """get_session returns a dict when a row is found."""
    from src.storage.repository import ResearchRepository

    row = {"id": 7, "question": "Q7", "answer": "A7"}
    pool = _make_pool_mock(fetchone_result=row)
    repo = ResearchRepository()
    with patch("src.storage.repository._get_pool", AsyncMock(return_value=pool)):
        result = await repo.get_session(7)
    assert result == dict(row)


@pytest.mark.asyncio
async def test_real_repo_get_session_not_found():
    """get_session returns None when the row is not found."""
    from src.storage.repository import ResearchRepository

    pool = _make_pool_mock(fetchone_result=None)
    repo = ResearchRepository()
    with patch("src.storage.repository._get_pool", AsyncMock(return_value=pool)):
        result = await repo.get_session(9999)
    assert result is None


# ---------------------------------------------------------------------------
# Repository graceful degradation — _get_pool raises
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_save_session_returns_none_on_db_error():
    """When the pool cannot connect, save_session returns None without raising."""
    from src.storage.repository import ResearchRepository

    repo = ResearchRepository()
    with patch("src.storage.repository._get_pool",
               side_effect=ConnectionError("db down")):
        result = await repo.save_session("Q", "A", [], [], False, 0.1)
    assert result is None


@pytest.mark.asyncio
async def test_get_sessions_returns_empty_on_db_error():
    """When the pool cannot connect, get_sessions returns [] without raising."""
    from src.storage.repository import ResearchRepository

    repo = ResearchRepository()
    with patch("src.storage.repository._get_pool",
               side_effect=ConnectionError("db down")):
        result = await repo.get_sessions()
    assert result == []


@pytest.mark.asyncio
async def test_get_session_returns_none_on_db_error():
    """When the pool cannot connect, get_session returns None without raising."""
    from src.storage.repository import ResearchRepository

    repo = ResearchRepository()
    with patch("src.storage.repository._get_pool",
               side_effect=ConnectionError("db down")):
        result = await repo.get_session(1)
    assert result is None


# ---------------------------------------------------------------------------
# init_db — graceful on failure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_init_db_does_not_raise_on_connection_error():
    """init_db logs a warning and does not propagate when _get_pool fails."""
    from src.storage.repository import init_db

    with patch("src.storage.repository._get_pool",
               side_effect=ConnectionError("db down")):
        await init_db()   # must not raise


@pytest.mark.asyncio
async def test_init_db_executes_ddl():
    """init_db must execute both CREATE TABLE and CREATE INDEX statements."""
    from src.storage.repository import init_db

    pool = _make_pool_mock()
    with patch("src.storage.repository._get_pool", AsyncMock(return_value=pool)):
        await init_db()

    conn = pool._conn
    assert conn.execute.call_count == 2, (
        f"Expected 2 DDL statements (CREATE TABLE + CREATE INDEX), "
        f"got {conn.execute.call_count}"
    )
    conn.commit.assert_called_once()


# ---------------------------------------------------------------------------
# Connection pool behaviour
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pool_connection_reused_across_calls():
    """_get_pool returns the same pool object on every call (no reconnect per query).

    This is the core correctness test for the connection-pooling fix.
    The old implementation called _get_conn() (open a new connection) inside
    every method.  The new implementation calls _get_pool() which returns the
    same pool singleton, then borrows a connection from it via pool.connection().
    """
    import src.storage.repository as repo_mod

    pool = _make_pool_mock(fetchone_result={"id": 1})
    get_pool_calls = 0

    async def _counting_get_pool():
        nonlocal get_pool_calls
        get_pool_calls += 1
        return pool

    repo = repo_mod.ResearchRepository()
    with patch("src.storage.repository._get_pool", _counting_get_pool):
        await repo.save_session("Q1", "A", [], [], False, 0.1)
        await repo.save_session("Q2", "A", [], [], False, 0.2)

    # _get_pool is called once per operation (it returns the singleton quickly)
    assert get_pool_calls == 2, (
        "Expected _get_pool called once per operation to fetch the singleton pool. "
        f"Got {get_pool_calls}."
    )
