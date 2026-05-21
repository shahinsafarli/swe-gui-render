"""Tests for the TTL-aware in-memory cache."""
from __future__ import annotations

import time
import pytest

from src.services import cache


def _make_sources(n: int = 2):
    from ai.schemas import Source
    return [
        Source(
            title=f"Source {i}",
            url=f"https://example.com/{i}",
            snippet=f"Snippet {i}",
            origin="wikipedia",
        )
        for i in range(n)
    ]


def test_cache_miss_on_empty():
    assert cache.get("wikipedia", "photosynthesis") is None


def test_cache_roundtrip():
    sources = _make_sources(2)
    cache.set("wikipedia", "photosynthesis", sources)
    result = cache.get("wikipedia", "photosynthesis")
    assert result is not None
    assert len(result) == 2


def test_cache_canonicalises_query():
    """Case and whitespace differences should hit the same key."""
    sources = _make_sources(1)
    cache.set("arxiv", "Quantum Computing", sources)
    assert cache.get("arxiv", "quantum computing") is not None
    assert cache.get("arxiv", "  QUANTUM  COMPUTING  ") is not None


def test_cache_different_sources_independent():
    """Different source names are independent keys."""
    s1 = _make_sources(1)
    s2 = _make_sources(2)
    cache.set("wikipedia", "test", s1)
    cache.set("arxiv", "test", s2)
    assert len(cache.get("wikipedia", "test")) == 1
    assert len(cache.get("arxiv", "test")) == 2


def test_cache_clear():
    cache.set("wikipedia", "test", _make_sources())
    cache.clear()
    assert cache.get("wikipedia", "test") is None


def test_cache_expiry(monkeypatch):
    """Entry is treated as a miss after TTL expires."""
    from src.services import cache as c
    from src.config import settings

    monkeypatch.setattr(settings, "cache_ttl_seconds", 1)
    sources = _make_sources()
    cache.set("wikipedia", "expiry-test", sources)

    # Manually age the entry
    key = ("wikipedia", "expiry-test")
    stored_at, val = c._store[key]
    c._store[key] = (stored_at - 2, val)  # make it 2 seconds old

    assert cache.get("wikipedia", "expiry-test") is None


def test_cache_zero_ttl_never_expires(monkeypatch):
    """TTL of 0 means entries never expire."""
    from src.config import settings

    monkeypatch.setattr(settings, "cache_ttl_seconds", 0)
    sources = _make_sources()
    cache.set("web", "no-expire", sources)

    from src.services import cache as c
    key = ("web", "no-expire")
    stored_at, val = c._store[key]
    c._store[key] = (stored_at - 99999, val)  # very old

    result = cache.get("web", "no-expire")
    assert result is not None
