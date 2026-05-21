"""Tests for the concurrent orchestration pipeline."""
from __future__ import annotations

import asyncio
import pytest

from src.concurrency.orchestrator import run_pipeline, fetch_sources_concurrent


# ---- run_pipeline -----------------------------------------------------------


@pytest.mark.asyncio
async def test_all_succeed():
    """All coroutines succeed — results returned in input order."""
    async def task(n: int) -> int:
        await asyncio.sleep(0.001)
        return n * 2

    results = await run_pipeline([task(i) for i in range(5)])
    assert results == [0, 2, 4, 6, 8]


@pytest.mark.asyncio
async def test_one_fails_others_complete():
    """One failure does not abort the other tasks."""
    async def good(n: int) -> int:
        return n

    async def bad() -> int:
        raise ValueError("simulated failure")

    results = await run_pipeline([good(1), bad(), good(3)])
    assert results[0] == 1
    assert isinstance(results[1], ValueError)
    assert results[2] == 3


@pytest.mark.asyncio
async def test_semaphore_limits_concurrency():
    """Peak concurrency never exceeds max_concurrent."""
    running = 0
    peak = 0

    async def task() -> None:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.02)
        running -= 1

    await run_pipeline([task() for _ in range(10)], max_concurrent=3)
    assert peak <= 3


@pytest.mark.asyncio
async def test_empty_pipeline():
    """Empty input returns empty list."""
    results = await run_pipeline([])
    assert results == []


# ---- fetch_sources_concurrent ----------------------------------------------


@pytest.mark.asyncio
async def test_fetch_sources_all_succeed(fake_web):
    """All three sources return results when no failures."""
    fetched, failed = await fetch_sources_concurrent(
        "photosynthesis",
        sources={"wikipedia", "arxiv", "web"},
        web_provider=fake_web,
    )
    assert "wikipedia" in fetched
    assert "arxiv" in fetched
    assert "web" in fetched
    assert failed == []


@pytest.mark.asyncio
async def test_fetch_sources_subset(fake_web):
    """Only the requested subset of sources are queried."""
    fetched, failed = await fetch_sources_concurrent(
        "quantum computing",
        sources={"wikipedia"},
        web_provider=fake_web,
    )
    assert "wikipedia" in fetched
    assert "arxiv" not in fetched
    assert "web" not in fetched


@pytest.mark.asyncio
async def test_fetch_sources_graceful_degradation(monkeypatch, fake_web):
    """If one source errors, others still return results."""
    from src.concurrency import orchestrator

    original_fns = orchestrator._FETCH_FNS.copy()

    async def _failing_arxiv(query, **kwargs):
        raise ConnectionError("arXiv is down")

    new_fns = dict(original_fns)
    new_fns["arxiv"] = _failing_arxiv
    monkeypatch.setattr(orchestrator, "_FETCH_FNS", new_fns)

    fetched, failed = await fetch_sources_concurrent(
        "neural networks",
        sources={"wikipedia", "arxiv"},
        web_provider=fake_web,
    )
    assert "arxiv" in failed
    assert "wikipedia" in fetched


@pytest.mark.asyncio
async def test_fetch_sources_all_fail(monkeypatch, fake_web):
    """All sources failing returns empty fetched and all in failed."""
    from src.concurrency import orchestrator

    async def _fail(query, **kwargs):
        raise ConnectionError("down")

    new_fns = {
        "wikipedia": _fail,
        "arxiv": _fail,
        "web": _fail,
    }
    monkeypatch.setattr(orchestrator, "_FETCH_FNS", new_fns)

    fetched, failed = await fetch_sources_concurrent(
        "test",
        sources={"wikipedia", "arxiv"},
        web_provider=fake_web,
    )
    assert fetched == {}
    assert set(failed) == {"wikipedia", "arxiv"}
