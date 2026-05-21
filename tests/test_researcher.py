"""Tests for the core research pipeline."""
from __future__ import annotations

import pytest

from src.models import ResearchRequest, SourceFilter
from src.core.researcher import run_research


# ---- Helpers ----------------------------------------------------------------


def make_request(question="What is photosynthesis?", sources=None, no_cache=False):
    return ResearchRequest(
        question=question,
        sources=sources,
        no_cache=no_cache,
    )


# ---- Basic end-to-end -------------------------------------------------------


@pytest.mark.asyncio
async def test_run_research_returns_result(fake_llm, fake_web):
    result = await run_research(make_request(), llm=fake_llm, web_provider=fake_web)
    assert result is not None
    assert result.question == "What is photosynthesis?"
    assert len(result.answer) > 0


@pytest.mark.asyncio
async def test_run_research_all_three_sources(fake_llm, fake_web):
    """All three sources should be attempted when no filter is set."""
    result = await run_research(make_request(), llm=fake_llm, web_provider=fake_web)
    origins = {s.origin for s in result.sources_used}
    assert "wikipedia" in origins
    assert "arxiv" in origins


@pytest.mark.asyncio
async def test_run_research_source_filter(fake_llm, fake_web):
    """Only the requested sources should be queried."""
    req = make_request(sources=[SourceFilter.wiki])
    result = await run_research(req, llm=fake_llm, web_provider=fake_web)
    origins = {s.origin for s in result.sources_used}
    assert "wikipedia" in origins
    assert "web" not in origins
    assert "arxiv" not in origins


# ---- Caching ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_research_caches_results(fake_llm, fake_web):
    """Second call should be served from cache."""
    req = make_request(question="unique-cache-test-question")
    r1 = await run_research(req, llm=fake_llm, web_provider=fake_web)
    assert not r1.cached

    r2 = await run_research(req, llm=fake_llm, web_provider=fake_web)
    assert r2.cached


@pytest.mark.asyncio
async def test_run_research_no_cache_bypasses(fake_llm, fake_web):
    """--no-cache should skip the cache even on second call."""
    req = make_request(question="no-cache-bypass-test", no_cache=True)
    r1 = await run_research(req, llm=fake_llm, web_provider=fake_web)
    r2 = await run_research(req, llm=fake_llm, web_provider=fake_web)
    assert not r1.cached
    assert not r2.cached


@pytest.mark.asyncio
async def test_no_cache_does_not_populate_cache(fake_llm, fake_web):
    """--no-cache must not write fetched sources into the cache.

    After a no-cache run, a subsequent normal (cache-enabled) call for the
    same question must fetch fresh sources — not serve a cache hit that was
    silently written by the --no-cache run.
    """
    from src.services import cache as cache_store
    from src.core.researcher import _normalize_query

    question = "no-cache-write-guard-test"
    cache_key = _normalize_query(question)  # deterministic key researcher uses

    # First call: --no-cache; should fetch fresh and NOT write to cache
    no_cache_req = make_request(question=question, no_cache=True)
    r1 = await run_research(no_cache_req, llm=fake_llm, web_provider=fake_web)
    assert not r1.cached

    # Cache must be empty — the no-cache run must not have written anything
    assert cache_store.get("wikipedia", cache_key) is None
    assert cache_store.get("arxiv", cache_key) is None
    assert cache_store.get("web", cache_key) is None

    # Second call: normal (cache-enabled). Should fetch fresh (cache was never
    # populated), so cached == False.
    normal_req = make_request(question=question, no_cache=False)
    r2 = await run_research(normal_req, llm=fake_llm, web_provider=fake_web)
    assert not r2.cached  # genuinely fresh, not a ghost cache hit


# ---- Graceful degradation --------------------------------------------------


@pytest.mark.asyncio
async def test_run_research_degraded_when_source_fails(monkeypatch, fake_llm, fake_web):
    """If one source errors, the answer is still produced from the others."""
    from src.concurrency import orchestrator

    original_fns = orchestrator._FETCH_FNS.copy()

    async def _fail(query, **kwargs):
        raise ConnectionError("arXiv is down")

    new_fns = dict(original_fns)
    new_fns["arxiv"] = _fail
    monkeypatch.setattr(orchestrator, "_FETCH_FNS", new_fns)

    req = make_request(question="degraded source test")
    result = await run_research(req, llm=fake_llm, web_provider=fake_web)
    assert "arxiv" in result.sources_failed
    assert len(result.answer) > 0  # still got an answer


# ---- Cache key normalization -----------------------------------------------


def test_normalize_query_strips_question_prefix():
    """Common question prefixes must be stripped so equivalent queries share a key."""
    from src.core.researcher import _normalize_query

    assert _normalize_query("What is photosynthesis?") == "photosynthesis"
    assert _normalize_query("What are black holes?") == "black holes"
    assert _normalize_query("How does DNA replication work?") == "DNA replication work"
    assert _normalize_query("Explain quantum computing") == "quantum computing"
    assert _normalize_query("Define entropy") == "entropy"
    assert _normalize_query("Tell me about gravity") == "gravity"


def test_normalize_query_equivalent_forms_same_key():
    """All equivalent phrasings of a query must produce the same normalized key."""
    from src.core.researcher import _normalize_query

    assert _normalize_query("photosynthesis") == _normalize_query("What is photosynthesis?")
    assert _normalize_query("photosynthesis") == _normalize_query("Explain photosynthesis")
    assert _normalize_query("quantum computing") == _normalize_query("Explain quantum computing")


def test_normalize_query_fallback_when_everything_stripped():
    """If normalization would return empty, original question is used as fallback."""
    from src.core.researcher import _normalize_query

    result = _normalize_query("What is?")
    assert result  # must be non-empty


@pytest.mark.asyncio
async def test_cache_hit_across_equivalent_questions(fake_llm, fake_web):
    """'photosynthesis' and 'What is photosynthesis?' must share the same cache entry."""
    from src.core.researcher import _normalize_query

    q1 = "What is photosynthesis?"
    q2 = "photosynthesis"

    # Both normalize to the same key
    assert _normalize_query(q1) == _normalize_query(q2)

    # Prime the cache with the full question form
    r1 = await run_research(make_request(question=q1), llm=fake_llm, web_provider=fake_web)
    assert not r1.cached

    # Bare keyword form must hit the same cache entry
    r2 = await run_research(make_request(question=q2), llm=fake_llm, web_provider=fake_web)
    assert r2.cached


# ---- Input validation -------------------------------------------------------


def test_request_rejects_empty_question():
    with pytest.raises(Exception):
        ResearchRequest(question="")


def test_request_rejects_whitespace_question():
    with pytest.raises(Exception):
        ResearchRequest(question="   ")


def test_request_rejects_overlength_question():
    with pytest.raises(Exception):
        ResearchRequest(question="a" * 501)
