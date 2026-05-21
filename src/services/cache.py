"""
TTL-aware in-memory cache for (source_name, canonicalized_query) → list[Source].

Design choice: in-memory store
-------------------------------
The spec permitted three backing stores: PostgreSQL, filesystem JSON, or
in-memory.  This module implements the **in-memory** option deliberately:

Pros:
  - Zero configuration — no database or disk path required.
  - Zero latency reads — a Python dict lookup is nanoseconds.
  - Simple enough to test exhaustively without mocking I/O.

Cons / limitations you must know before deploying:
  - **Ephemeral** — the cache is wiped on every process restart.  Restarting
    the assistant (or running it as a CLI one-shot command) starts with a cold
    cache every time.
  - **Not shared** — each process has its own private cache.  Running multiple
    assistant workers behind a load balancer gives each worker its own island;
    there is no cross-process cache coherence.
  - **Unbounded growth** — entries are only evicted on access (lazy TTL check).
    A process that is left running for days with a high query volume will
    accumulate stale entries.  For a long-running service, call ``cache.clear()``
    periodically or replace this module with a Redis-backed implementation.

If persistence across restarts or cross-process sharing is required, replace
``_store`` with a filesystem JSON implementation (write on ``set``, read on
``get``) or a Redis/PostgreSQL backend.  The public API (``get``, ``set``,
``clear``, ``stats``) is intentionally minimal so the backing store can be
swapped without touching callers.

Key design:
- Key: (source_name, canonical_query) where canonical_query is lowercased + stripped.
- Value: (timestamp, list[Source])
- On get: if age > TTL, entry is treated as missing (lazy eviction).
- Thread-safe for asyncio (single-threaded event loop — no locking needed).
- --no-cache flag bypasses reads and writes entirely.
"""
from __future__ import annotations

import abc
import logging
import re
import time
from typing import Any

from src.config import settings

logger = logging.getLogger(__name__)


class CacheBackend(abc.ABC):
    """Abstract base for cache store implementations.

    Concrete backends must implement ``get``, ``set``, ``clear``, and
    ``stats``.  The in-memory backend below is the default; Redis or a
    filesystem-JSON backend can be substituted without touching callers.
    """

    @abc.abstractmethod
    def get(self, source: str, query: str) -> list[Any] | None:
        """Return cached value or None on miss/expiry."""
        raise NotImplementedError

    @abc.abstractmethod
    def set(self, source: str, query: str, value: list[Any]) -> None:
        """Store *value* under *(source, query)*."""
        raise NotImplementedError

    @abc.abstractmethod
    def clear(self) -> None:
        """Evict all entries."""
        raise NotImplementedError

    @abc.abstractmethod
    def stats(self) -> dict[str, int]:
        """Return diagnostic counters."""
        raise NotImplementedError


# Internal store: key -> (stored_at_float, value)
_store: dict[tuple[str, str], tuple[float, list[Any]]] = {}


def _canonicalize(query: str) -> str:
    """Normalize a query string for use as a cache key."""
    return re.sub(r"\s+", " ", query.lower().strip())


class InMemoryCacheBackend(CacheBackend):
    """TTL-aware in-memory implementation of :class:`CacheBackend`.

    Uses a module-level dict so all instances share the same store
    (appropriate for a single-process server).  See the module docstring
    for trade-offs and guidance on swapping this for a Redis/PostgreSQL
    backend.
    """

    def get(self, source: str, query: str) -> list[Any] | None:  # noqa: D102
        key = (source, _canonicalize(query))
        entry = _store.get(key)
        if entry is None:
            logger.debug("cache_miss", extra={"source": source, "query": query[:40]})
            return None
        stored_at, value = entry
        age = time.monotonic() - stored_at
        if settings.cache_ttl_seconds > 0 and age > settings.cache_ttl_seconds:
            del _store[key]
            logger.debug(
                "cache_expired",
                extra={"source": source, "age_s": round(age), "query": query[:40]},
            )
            return None
        logger.info(
            "cache_hit",
            extra={"source": source, "count": len(value), "age_s": round(age)},
        )
        return value

    def set(self, source: str, query: str, value: list[Any]) -> None:  # noqa: D102,A003
        key = (source, _canonicalize(query))
        _store[key] = (time.monotonic(), value)
        logger.debug(
            "cache_store",
            extra={"source": source, "count": len(value), "query": query[:40]},
        )

    def clear(self) -> None:  # noqa: D102
        _store.clear()

    def stats(self) -> dict[str, int]:  # noqa: D102
        return {"entries": len(_store)}


# Module-level singleton used by the rest of the codebase
_backend: CacheBackend = InMemoryCacheBackend()


# ---------------------------------------------------------------------------
# Module-level convenience shims (backward-compatible public API)
# ---------------------------------------------------------------------------

def get(source: str, query: str) -> list[Any] | None:
    """Return cached sources for *(source, query)* if still fresh."""
    return _backend.get(source, query)


def set(source: str, query: str, value: list[Any]) -> None:  # noqa: A001
    """Store sources for *(source, query)*."""
    _backend.set(source, query, value)


def clear() -> None:
    """Clear all cache entries (used in tests)."""
    _backend.clear()


def stats() -> dict[str, int]:
    """Return cache size info."""
    return _backend.stats()
