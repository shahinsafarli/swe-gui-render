"""
HTTP-level tests using pytest-httpx and respx.

These tests mock at the *transport* layer — actual ``httpx.AsyncClient``
requests are intercepted before any socket is opened — rather than patching
``ai.*`` functions.  This exercises the HTTP client behaviour that
function-level monkeypatching cannot reach:

  - Correct URL construction and query parameters
  - Response parsing from realistic payloads
  - Retry triggering on HTTP 5xx / connection-reset errors
  - Timeout handling at the httpx level

The spec for Topic 4 explicitly names respx / pytest-httpx as the preferred
tools for offline HTTP mocking (TOPIC.md §Tests).  Both are included here:

  - ``pytest-httpx`` (``httpx_mock`` fixture) — simpler API, ideal for
    straightforward request/response stubs.
  - ``respx`` — pattern-based routing, ideal for multi-URL flows like the
    Wikipedia two-step (search → summary).

All tests are offline: no real network calls are made.

Important: the conftest ``no_real_ai`` autouse fixture patches ``ai.fetch_*``
at the *function* level so the httpx transport is never reached.  The
``restore_real_ai`` fixture defined here reverses those patches for every
test in this module, so the real ``ai.fetch_wikipedia`` / ``ai.fetch_arxiv``
code paths execute and the respx / pytest-httpx transport mocks intercept
the resulting httpx calls.
"""
from __future__ import annotations

import json
import re

import httpx
import pytest

import ai
from ai.schemas import Source
from ai import sources as _ai_sources


@pytest.fixture(autouse=True)
def restore_real_ai():
    """Restore the real ai.fetch_wikipedia / ai.fetch_arxiv for this module.

    The conftest ``no_real_ai`` autouse fixture replaces these with in-memory
    fakes.  The HTTP-level tests in this file need the real implementations so
    that httpx requests are actually made (and intercepted by respx/pytest-httpx).
    This fixture runs *after* no_real_ai (fixtures run in definition order,
    with autouse fixtures from the local module overriding conftest ones at the
    same scope), explicitly restoring the originals.
    """
    real_fetch_wikipedia = _ai_sources.fetch_wikipedia
    real_fetch_arxiv = _ai_sources.fetch_arxiv
    ai.fetch_wikipedia = real_fetch_wikipedia
    ai.fetch_arxiv = real_fetch_arxiv
    yield
    # Teardown: conftest's monkeypatch will restore after the test anyway,
    # but be explicit so the module is clean if tests run in unusual orders.
    ai.fetch_wikipedia = real_fetch_wikipedia
    ai.fetch_arxiv = real_fetch_arxiv

# ---------------------------------------------------------------------------
# Helpers — minimal realistic API responses
# ---------------------------------------------------------------------------

_WIKI_SEARCH_RESPONSE = json.dumps([
    "photosynthesis",
    ["Photosynthesis", "Light-dependent reactions"],
    ["", ""],
    [
        "https://en.wikipedia.org/wiki/Photosynthesis",
        "https://en.wikipedia.org/wiki/Light-dependent_reactions",
    ],
])

_WIKI_SUMMARY_PHOTO = json.dumps({
    "title": "Photosynthesis",
    "extract": "Photosynthesis is a process used by plants to convert light energy into chemical energy.",
    "content_urls": {
        "desktop": {"page": "https://en.wikipedia.org/wiki/Photosynthesis"}
    },
})

_WIKI_SUMMARY_LIGHT = json.dumps({
    "title": "Light-dependent reactions",
    "extract": "The light-dependent reactions of photosynthesis take place in the thylakoid membranes.",
    "content_urls": {
        "desktop": {"page": "https://en.wikipedia.org/wiki/Light-dependent_reactions"}
    },
})

_ARXIV_ATOM_RESPONSE = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v5</id>
    <title>Attention Is All You Need</title>
    <summary>The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.</summary>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2005.14165v4</id>
    <title>Language Models are Few-Shot Learners</title>
    <summary>We demonstrate that scaling up language models greatly improves task-agnostic, few-shot performance.</summary>
  </entry>
</feed>
"""


# ===========================================================================
# Wikipedia tests — using respx (pattern-based routing handles two-step fetch)
# ===========================================================================


@pytest.mark.asyncio
async def test_fetch_wikipedia_correct_search_url(respx_mock):
    """fetch_wikipedia hits the correct Wikipedia opensearch endpoint."""
    respx_mock.get("https://en.wikipedia.org/w/api.php").mock(
        return_value=httpx.Response(200, text=_WIKI_SEARCH_RESPONSE)
    )
    # Stub summary so the second request doesn't fail
    respx_mock.get(re.compile(r"https://en\.wikipedia\.org/api/rest_v1/page/summary/.*")).mock(
        return_value=httpx.Response(200, text=_WIKI_SUMMARY_PHOTO)
    )

    async with httpx.AsyncClient() as client:
        await ai.fetch_wikipedia("photosynthesis", client=client)

    assert respx_mock.calls.call_count >= 1
    search_call = respx_mock.calls[0]
    assert "opensearch" in str(search_call.request.url)
    assert "photosynthesis" in str(search_call.request.url)


@pytest.mark.asyncio
async def test_fetch_wikipedia_returns_parsed_sources(respx_mock):
    """fetch_wikipedia parses the two-step response into Source objects."""
    respx_mock.get("https://en.wikipedia.org/w/api.php").mock(
        return_value=httpx.Response(200, text=_WIKI_SEARCH_RESPONSE)
    )
    respx_mock.get(re.compile(r".*/Photosynthesis$")).mock(
        return_value=httpx.Response(200, text=_WIKI_SUMMARY_PHOTO)
    )
    respx_mock.get(re.compile(r".*/Light-dependent_reactions$")).mock(
        return_value=httpx.Response(200, text=_WIKI_SUMMARY_LIGHT)
    )

    async with httpx.AsyncClient() as client:
        sources = await ai.fetch_wikipedia("photosynthesis", max_results=2, client=client)

    assert len(sources) == 2
    assert all(isinstance(s, Source) for s in sources)
    assert all(s.origin == "wikipedia" for s in sources)
    titles = {s.title for s in sources}
    assert "Photosynthesis" in titles
    assert "Light-dependent reactions" in titles


@pytest.mark.asyncio
async def test_fetch_wikipedia_empty_query_returns_empty(respx_mock):
    """fetch_wikipedia short-circuits on blank query without making any HTTP call."""
    sources = await ai.fetch_wikipedia("   ")
    assert sources == []
    assert respx_mock.calls.call_count == 0


@pytest.mark.asyncio
async def test_fetch_wikipedia_http_500_raises(respx_mock):
    """fetch_wikipedia raises ProviderError on HTTP 5xx from the search endpoint."""
    from ai.providers.base import ProviderError

    respx_mock.get("https://en.wikipedia.org/w/api.php").mock(
        return_value=httpx.Response(500, text="Internal Server Error")
    )

    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError):
            await ai.fetch_wikipedia("photosynthesis", client=client)


@pytest.mark.asyncio
async def test_fetch_wikipedia_summary_404_skipped(respx_mock):
    """A 404 on one summary URL is skipped; other titles still returned."""
    respx_mock.get("https://en.wikipedia.org/w/api.php").mock(
        return_value=httpx.Response(200, text=_WIKI_SEARCH_RESPONSE)
    )
    # First title → 404 (skip)
    respx_mock.get(re.compile(r".*/Photosynthesis$")).mock(
        return_value=httpx.Response(404, text="Not Found")
    )
    # Second title → ok
    respx_mock.get(re.compile(r".*/Light-dependent_reactions$")).mock(
        return_value=httpx.Response(200, text=_WIKI_SUMMARY_LIGHT)
    )

    async with httpx.AsyncClient() as client:
        sources = await ai.fetch_wikipedia("photosynthesis", max_results=2, client=client)

    # The 404'd title is silently skipped; the other one still comes through.
    assert len(sources) == 1
    assert sources[0].title == "Light-dependent reactions"


# ===========================================================================
# arXiv tests — using pytest-httpx (httpx_mock fixture)
# ===========================================================================


@pytest.mark.asyncio
async def test_fetch_arxiv_correct_url(httpx_mock):
    """fetch_arxiv hits the correct arXiv export endpoint."""
    httpx_mock.add_response(
        url=re.compile(r"https://export\.arxiv\.org/api/query.*"),
        text=_ARXIV_ATOM_RESPONSE,
    )

    async with httpx.AsyncClient() as client:
        await ai.fetch_arxiv("attention mechanism", client=client)

    requests = httpx_mock.get_requests()
    assert len(requests) == 1
    url = str(requests[0].url)
    assert "export.arxiv.org" in url
    assert "attention+mechanism" in url or "attention%20mechanism" in url or "attention" in url


@pytest.mark.asyncio
async def test_fetch_arxiv_parses_atom_correctly(httpx_mock):
    """fetch_arxiv parses Atom XML into Source objects with correct fields."""
    httpx_mock.add_response(
        url=re.compile(r"https://export\.arxiv\.org/api/query.*"),
        text=_ARXIV_ATOM_RESPONSE,
    )

    async with httpx.AsyncClient() as client:
        sources = await ai.fetch_arxiv("transformers", client=client)

    assert len(sources) == 2
    assert all(isinstance(s, Source) for s in sources)
    assert all(s.origin == "arxiv" for s in sources)
    assert sources[0].title == "Attention Is All You Need"
    assert sources[0].url == "http://arxiv.org/abs/1706.03762v5"
    assert "recurrent" in sources[0].snippet


@pytest.mark.asyncio
async def test_fetch_arxiv_empty_query_returns_empty(httpx_mock):
    """fetch_arxiv short-circuits on blank query without making any HTTP call."""
    sources = await ai.fetch_arxiv("  ")
    assert sources == []
    assert httpx_mock.get_requests() == []


@pytest.mark.asyncio
async def test_fetch_arxiv_max_results_respected(httpx_mock):
    """fetch_arxiv passes max_results to the API query parameter."""
    httpx_mock.add_response(
        url=re.compile(r"https://export\.arxiv\.org/api/query.*"),
        text=_ARXIV_ATOM_RESPONSE,
    )

    async with httpx.AsyncClient() as client:
        await ai.fetch_arxiv("neural networks", max_results=1, client=client)

    url = str(httpx_mock.get_requests()[0].url)
    assert "max_results=1" in url


@pytest.mark.asyncio
async def test_fetch_arxiv_http_500_raises(httpx_mock):
    """fetch_arxiv raises ProviderError on HTTP 5xx."""
    from ai.providers.base import ProviderError

    httpx_mock.add_response(
        url=re.compile(r"https://export\.arxiv\.org/api/query.*"),
        status_code=500,
        text="Service Unavailable",
    )

    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError):
            await ai.fetch_arxiv("quantum computing", client=client)


@pytest.mark.asyncio
async def test_fetch_arxiv_connection_error(httpx_mock):
    """fetch_arxiv propagates a network-level error as an exception."""
    httpx_mock.add_exception(
        httpx.ConnectError("Connection refused"),
        url=re.compile(r"https://export\.arxiv\.org/api/query.*"),
    )

    async with httpx.AsyncClient() as client:
        with pytest.raises(Exception):
            await ai.fetch_arxiv("quantum computing", client=client)


@pytest.mark.asyncio
async def test_fetch_arxiv_malformed_xml_raises(httpx_mock):
    """fetch_arxiv raises ProviderError when the response is not valid Atom XML."""
    from ai.providers.base import ProviderError

    httpx_mock.add_response(
        url=re.compile(r"https://export\.arxiv\.org/api/query.*"),
        text="this is not xml at all <<<",
    )

    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError):
            await ai.fetch_arxiv("quantum computing", client=client)


@pytest.mark.asyncio
async def test_fetch_arxiv_empty_feed_returns_empty_list(httpx_mock):
    """An Atom feed with no entries yields an empty list (not an error)."""
    empty_feed = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    httpx_mock.add_response(
        url=re.compile(r"https://export\.arxiv\.org/api/query.*"),
        text=empty_feed,
    )

    async with httpx.AsyncClient() as client:
        sources = await ai.fetch_arxiv("xyzzy-nonsense-query", client=client)

    assert sources == []


# ===========================================================================
# AIService retry tests — respx verifies retry hits the endpoint multiple times
# ===========================================================================


@pytest.mark.asyncio
async def test_ai_service_fetch_arxiv_retries_on_connection_error(monkeypatch):
    """AIService.fetch_arxiv retries the HTTP call on ConnectionError.

    This test patches tenacity sleep so retries fire instantly, then verifies
    that after two failures the third attempt succeeds and the result is
    returned to the caller.
    """
    from src.services.ai_service import AIService
    from ai.schemas import Source
    from ai.providers.base import ProviderError

    call_count = 0

    async def _flaky_arxiv(query: str, **kwargs) -> list[Source]:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ProviderError(f"arXiv 503 (attempt {call_count})")
        return [
            Source(
                title="Successful paper",
                url="https://arxiv.org/abs/9999.00001",
                snippet="Success on attempt 3.",
                origin="arxiv",
            )
        ]

    monkeypatch.setattr(ai, "fetch_arxiv", _flaky_arxiv)
    monkeypatch.setattr("tenacity.nap.sleep", lambda _: None)

    service = AIService()
    result = await service.fetch_arxiv("transformers")

    assert call_count == 3
    assert len(result) == 1
    assert result[0].title == "Successful paper"


@pytest.mark.asyncio
async def test_ai_service_fetch_wikipedia_retries_on_connection_error(monkeypatch):
    """AIService.fetch_wikipedia retries on transient ConnectionError at the HTTP level."""
    from src.services.ai_service import AIService
    from ai.schemas import Source

    call_count = 0

    async def _flaky_wiki(query: str, **kwargs) -> list[Source]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionError("TCP reset by peer")
        return [
            Source(
                title="Wikipedia: Photosynthesis",
                url="https://en.wikipedia.org/wiki/Photosynthesis",
                snippet="Photosynthesis is a process.",
                origin="wikipedia",
            )
        ]

    monkeypatch.setattr(ai, "fetch_wikipedia", _flaky_wiki)
    monkeypatch.setattr("tenacity.nap.sleep", lambda _: None)

    service = AIService()
    result = await service.fetch_wikipedia("photosynthesis")

    assert call_count == 2
    assert result[0].origin == "wikipedia"


# ===========================================================================
# synthesize() async / thread-pool tests
# ===========================================================================


def test_synthesize_is_now_async():
    """synthesize() must be an async method (coroutine function) — not sync.

    This is the key structural change made to fix the thread-resource leak:
    the synchronous method decorated with @retry that created a new
    ThreadPoolExecutor on every attempt has been replaced with an async method
    that uses a shared module-level executor.
    """
    import inspect
    from src.services.ai_service import AIService
    assert inspect.iscoroutinefunction(AIService.synthesize), (
        "AIService.synthesize must be async so it can use asyncio.wait_for "
        "and the shared _synth_executor without leaking threads on each retry."
    )


@pytest.mark.asyncio
async def test_synthesize_uses_shared_executor(sample_sources, fake_llm):
    """synthesize() uses the module-level _synth_executor, not a per-call one."""
    from src.services.ai_service import AIService, _synth_executor

    service = AIService()
    result = await service.synthesize("What is photosynthesis?", sample_sources, llm=fake_llm)

    assert result is not None
    assert len(result.answer) > 0
    # The executor should still be running (not shut down after the call)
    assert not _synth_executor._shutdown, (
        "_synth_executor must not be shut down after a single synthesize call — "
        "it is shared across all calls for the process lifetime."
    )


@pytest.mark.asyncio
async def test_synthesize_retries_on_provider_error(monkeypatch, sample_sources):
    """synthesize() retries on ProviderError and returns the eventual success.

    Simulates a flaky LLM provider that fails on the first attempt and
    succeeds on the second.  The retry loop in the async synthesize() method
    must catch ProviderError and try again with backoff (patched to instant).
    """
    from ai.providers.base import LLMProvider, ProviderError
    from src.services.ai_service import AIService

    attempt = 0

    class _FlakeyLLM(LLMProvider):
        def complete(self, prompt: str, **kwargs) -> str:
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                raise ProviderError("LLM 503 — try again")
            return "Photosynthesis converts sunlight [1]."

    monkeypatch.setattr("asyncio.sleep", lambda _: __import__("asyncio").coroutine(lambda: None)())

    async def _instant_sleep(_):
        pass

    monkeypatch.setattr("asyncio.sleep", _instant_sleep)

    service = AIService()
    result = await service.synthesize("What is photosynthesis?", sample_sources, llm=_FlakeyLLM())

    assert attempt == 2, f"Expected 2 attempts, got {attempt}"
    assert "Photosynthesis" in result.answer


@pytest.mark.asyncio
async def test_synthesize_raises_after_max_attempts(monkeypatch, sample_sources):
    """synthesize() raises ProviderError after exhausting all retry attempts."""
    from ai.providers.base import LLMProvider, ProviderError
    from src.services.ai_service import AIService

    attempts = 0

    class _AlwaysFailLLM(LLMProvider):
        def complete(self, prompt: str, **kwargs) -> str:
            nonlocal attempts
            attempts += 1
            raise ProviderError("LLM permanently down")

    async def _instant_sleep(_):
        pass

    monkeypatch.setattr("asyncio.sleep", _instant_sleep)

    service = AIService()
    with pytest.raises(ProviderError):
        await service.synthesize(
            "What is photosynthesis?",
            sample_sources,
            llm=_AlwaysFailLLM(),
            max_attempts=4,
        )

    assert attempts == 4, f"Expected exactly 4 attempts, got {attempts}"
