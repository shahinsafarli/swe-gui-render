"""
PostgreSQL repository for research session persistence.

Uses psycopg (psycopg3) async connections via a connection pool.

Graceful degradation
--------------------
If DATABASE_URL is unset, still contains placeholder values (YOUR_DB_*),
or Postgres is unreachable, every public function degrades gracefully:
errors are logged at WARNING level and the caller receives None / []
instead of an exception.  The rest of the pipeline continues without
history persistence.

Schema
------
  research_sessions
    id              SERIAL PRIMARY KEY
    question        TEXT NOT NULL
    answer          TEXT NOT NULL
    citations_json  JSONB NOT NULL DEFAULT '[]'
    sources_failed  JSONB NOT NULL DEFAULT '[]'
    cached          BOOLEAN NOT NULL DEFAULT FALSE
    elapsed_seconds FLOAT  NOT NULL DEFAULT 0.0
    created_at      TIMESTAMPTZ DEFAULT NOW()
"""
from __future__ import annotations

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS research_sessions (
    id              SERIAL PRIMARY KEY,
    question        TEXT NOT NULL,
    answer          TEXT NOT NULL,
    citations_json  JSONB NOT NULL DEFAULT '[]',
    sources_failed  JSONB NOT NULL DEFAULT '[]',
    cached          BOOLEAN NOT NULL DEFAULT FALSE,
    elapsed_seconds FLOAT NOT NULL DEFAULT 0.0,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_research_sessions_created_at
    ON research_sessions (created_at DESC);
"""

# ---------------------------------------------------------------------------
# Backward-compatibility shim
# ---------------------------------------------------------------------------
# The instance-based design owns pool state on each ResearchRepository via
# ``self._pool``.  The existing test suite patches two module-level symbols:
#
#   monkeypatch.setattr(repo_mod, "_pool", None)
#   patch("src.storage.repository._get_pool", AsyncMock(...))
#
# We expose both as real module-level names.  Instance methods delegate to
# the module-level ``_get_pool`` (looked up in globals at call-time), so
# unittest.mock patches are intercepted automatically by every instance.
#
# Writing ``_pool = None`` via monkeypatch resets the module-level variable;
# because ``_get_pool`` reads the same global it will re-create the pool on
# the next call, which is exactly the cold-start behaviour the tests expect.

_pool = None   # authoritative pool state; read/written by _get_pool below


async def _get_pool():
    """Return (or lazily create) the process-level connection pool.

    Declared at module scope so that ``patch("src.storage.repository._get_pool")``
    in tests intercepts every call made by ResearchRepository instance methods,
    which look this name up in the module globals at runtime.
    """
    global _pool
    if _pool is not None:
        return _pool

    from src.config import settings

    if settings.db_url_is_placeholder:
        logger.debug("db_skipped_placeholder_url")
        return None

    try:
        from psycopg_pool import AsyncConnectionPool  # type: ignore[import]
        from psycopg.rows import dict_row

        _pool = AsyncConnectionPool(
            conninfo=settings.database_url,
            kwargs={"row_factory": dict_row},
            min_size=1,
            max_size=5,
            open=False,
            reconnect_timeout=5,
            timeout=2.0,
        )
        await _pool.open(wait=False)
        logger.info("db_pool_opened", extra={"min": 1, "max": 5})
        return _pool
    except Exception as exc:
        logger.warning("db_pool_open_failed", extra={"error": str(exc)})
        _pool = None
        return None


async def init_db() -> None:
    """Open the connection pool and create the schema if it does not exist.

    Safe to call multiple times.  No-ops when DB is not configured.
    """
    try:
        pool = await _get_pool()
        if pool is None:
            return
        async with pool.connection() as conn:
            await conn.execute(_CREATE_TABLE)
            await conn.execute(_CREATE_INDEX)
            await conn.commit()
        logger.info("db_initialized")
    except Exception as exc:
        logger.warning("db_init_failed", extra={"error": str(exc)})


class ResearchRepository:
    """Encapsulates all SQL for research session persistence.

    OOP encapsulation note
    ----------------------
    All SQL logic lives in this class.  Pool lifecycle (creation, caching,
    placeholder detection) is handled by the module-level ``_get_pool``
    function.  Instance methods call ``_get_pool()`` via the module global
    rather than ``self._get_pool()``; this one-line indirection means that
    ``unittest.mock.patch("src.storage.repository._get_pool", ...)`` in
    tests intercepts every database call without requiring any special test
    hooks on the class itself.  The class therefore remains free of pool
    management code while still being fully mockable.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def save_session(
        self,
        question: str,
        answer: str,
        citations: list[dict],
        sources_failed: list[str],
        cached: bool,
        elapsed_seconds: float,
    ) -> Optional[int]:
        """Persist a research session. Returns the new row id, or None on failure."""
        try:
            pool = await _get_pool()
            if pool is None:
                return None
            async with pool.connection() as conn:
                cursor = await conn.execute(
                    """
                    INSERT INTO research_sessions
                        (question, answer, citations_json, sources_failed,
                         cached, elapsed_seconds)
                    VALUES (%s, %s, %s::jsonb, %s::jsonb, %s, %s)
                    RETURNING id
                    """,
                    (
                        question,
                        answer,
                        json.dumps(citations),
                        json.dumps(sources_failed),
                        cached,
                        elapsed_seconds,
                    ),
                )
                row = await cursor.fetchone()
                await conn.commit()
                session_id = row["id"]
            logger.info("session_saved", extra={"id": session_id})
            return session_id
        except Exception as exc:
            logger.warning("session_save_failed", extra={"error": str(exc)})
            return None

    async def get_sessions(self, limit: int = 20) -> list[dict]:
        """Return the most recent research sessions."""
        try:
            pool = await _get_pool()
            if pool is None:
                return []
            async with pool.connection() as conn:
                cursor = await conn.execute(
                    """
                    SELECT id, question, answer, citations_json, sources_failed,
                           cached, elapsed_seconds, created_at
                    FROM research_sessions
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = await cursor.fetchall()
            return list(rows)
        except Exception as exc:
            logger.warning("session_list_failed", extra={"error": str(exc)})
            return []

    async def get_session(self, session_id: int) -> dict | None:
        """Fetch a single session by id; returns None if not found."""
        try:
            pool = await _get_pool()
            if pool is None:
                return None
            async with pool.connection() as conn:
                cursor = await conn.execute(
                    "SELECT * FROM research_sessions WHERE id = %s",
                    (session_id,),
                )
                row = await cursor.fetchone()
            return dict(row) if row else None
        except Exception as exc:
            logger.warning("session_get_failed", extra={"error": str(exc)})
            return None


# Module-level singleton — import this everywhere
repository = ResearchRepository()
