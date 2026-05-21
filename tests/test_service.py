"""Tests for AIService — all offline (no real API calls)."""
from __future__ import annotations

import pytest

import ai
from src.services.ai_service import AIService


# ---- synthesize -------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_returns_answer(fake_llm, sample_sources):
    """synthesize() returns an AnswerWithCitations object."""
    service = AIService()
    result = await service.synthesize("What is photosynthesis?", sample_sources, llm=fake_llm)
    assert result is not None
    assert result.question == "What is photosynthesis?"
    assert len(result.answer) > 0


@pytest.mark.asyncio
async def test_synthesize_records_citations(fake_llm, sample_sources):
    """synthesize() correctly maps citations from the LLM response."""
    service = AIService()
    result = await service.synthesize("What is photosynthesis?", sample_sources, llm=fake_llm)
    citation_indices = {c.index for c in result.citations}
    assert citation_indices.issubset({1, 2})


@pytest.mark.asyncio
async def test_synthesize_rejects_empty_question(fake_llm, sample_sources):
    """synthesize() raises ValueError for blank question."""
    service = AIService()
    with pytest.raises(ValueError):
        await service.synthesize("   ", sample_sources, llm=fake_llm)


@pytest.mark.asyncio
async def test_synthesize_rejects_empty_sources(fake_llm):
    """synthesize() raises ValueError when no sources provided."""
    service = AIService()
    with pytest.raises(ValueError):
        await service.synthesize("What is X?", [], llm=fake_llm)


@pytest.mark.asyncio
async def test_synthesize_max_tokens_cap_applied(sample_sources, monkeypatch):
    """_BoundedLLM wrapper ensures max_tokens is passed to llm.complete()."""
    from src.services.ai_service import _SYNTHESIZE_MAX_TOKENS

    received_max_tokens: list[int] = []

    class _RecordingLLM:
        def complete(self, prompt: str, *, json_schema=None, max_tokens: int = 9999) -> str:
            received_max_tokens.append(max_tokens)
            return "Plants use sunlight for energy [1]."

    service = AIService()
    await service.synthesize("What is photosynthesis?", sample_sources, llm=_RecordingLLM())

    assert len(received_max_tokens) == 1
    assert received_max_tokens[0] == _SYNTHESIZE_MAX_TOKENS


# ---- fetch wrappers (offline via fixture patching) --------------------------


@pytest.mark.asyncio
async def test_fetch_wikipedia_returns_sources():
    """fetch_wikipedia() returns a non-empty list for a real query (mocked)."""
    service = AIService()
    results = await service.fetch_wikipedia("photosynthesis")
    assert isinstance(results, list)
    assert len(results) > 0
    assert results[0].origin == "wikipedia"


@pytest.mark.asyncio
async def test_fetch_arxiv_returns_sources(monkeypatch):
    """fetch_arxiv() returns a non-empty list (mocked via ai.fetch_arxiv).

    fetch_arxiv in AIService now creates its own dedicated httpx.AsyncClient
    with follow_redirects=True to handle the arXiv http→https redirect.
    The test patches ai.fetch_arxiv directly so it doesn't matter which
    client the service passes — the mock intercepts at the ai layer.
    """
    from ai.schemas import Source
    from ai.providers.base import ProviderError

    arxiv_source = Source(
        title="arXiv: neural networks survey",
        url="https://arxiv.org/abs/2301.00001",
        snippet="A survey of neural network architectures.",
        origin="arxiv",
    )

    async def _fake_arxiv(query: str, **kwargs) -> list[Source]:
        return [arxiv_source]

    monkeypatch.setattr(ai, "fetch_arxiv", _fake_arxiv)

    service = AIService()
    results = await service.fetch_arxiv("neural networks")
    assert isinstance(results, list)
    assert len(results) > 0
    assert results[0].origin == "arxiv"


@pytest.mark.asyncio
async def test_fetch_web_returns_sources(fake_web):
    """fetch_web() uses the supplied provider and returns sources."""
    service = AIService()
    results = await service.fetch_web("climate change", provider=fake_web)
    assert isinstance(results, list)
    assert all(s.origin == "web" for s in results)


# ---- retry behaviour --------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_wikipedia_retries_on_transient_error(monkeypatch):
    """fetch_wikipedia retries on ConnectionError and returns on eventual success.

    Scenario: the underlying ai.fetch_wikipedia fails twice (ConnectionError)
    then succeeds on the third attempt.  The service must:
      - not raise to the caller
      - return the successful result
      - have called the underlying function exactly 3 times
    """
    from ai.schemas import Source

    call_count = 0
    success_source = Source(
        title="Wikipedia: photosynthesis",
        url="https://en.wikipedia.org/wiki/Photosynthesis",
        snippet="A biological process.",
        origin="wikipedia",
    )

    async def _flaky(query: str, **kwargs) -> list[Source]:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError(f"transient error (attempt {call_count})")
        return [success_source]

    monkeypatch.setattr(ai, "fetch_wikipedia", _flaky)
    # Patch tenacity's async sleep so retries fire instantly in tests.
    monkeypatch.setattr("tenacity.nap.sleep", lambda _: None)

    service = AIService()
    result = await service.fetch_wikipedia("photosynthesis")

    assert call_count == 3, f"Expected 3 attempts, got {call_count}"
    assert len(result) == 1
    assert result[0].url == success_source.url


@pytest.mark.asyncio
async def test_fetch_arxiv_retries_on_transient_error(monkeypatch):
    """fetch_arxiv retries on ProviderError and succeeds on the second attempt."""
    from ai.schemas import Source
    from ai.providers.base import ProviderError

    call_count = 0
    success_source = Source(
        title="arXiv: attention",
        url="https://arxiv.org/abs/1706.03762",
        snippet="Transformer architecture.",
        origin="arxiv",
    )

    async def _flaky(query: str, **kwargs) -> list[Source]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ProviderError("rate limited")
        return [success_source]

    monkeypatch.setattr(ai, "fetch_arxiv", _flaky)
    monkeypatch.setattr("tenacity.nap.sleep", lambda _: None)

    service = AIService()
    result = await service.fetch_arxiv("attention mechanism")

    assert call_count == 2, f"Expected 2 attempts, got {call_count}"
    assert result[0].origin == "arxiv"


@pytest.mark.asyncio
async def test_fetch_wikipedia_raises_after_max_attempts(monkeypatch):
    """fetch_wikipedia raises ConnectionError after exhausting all 4 retry attempts."""
    call_count = 0

    async def _always_fail(query: str, **kwargs):
        nonlocal call_count
        call_count += 1
        raise ConnectionError("persistent failure")

    monkeypatch.setattr(ai, "fetch_wikipedia", _always_fail)
    monkeypatch.setattr("tenacity.nap.sleep", lambda _: None)

    service = AIService()
    with pytest.raises(ConnectionError):
        await service.fetch_wikipedia("photosynthesis")

    assert call_count == 4, f"Expected 4 attempts (max_attempts), got {call_count}"
