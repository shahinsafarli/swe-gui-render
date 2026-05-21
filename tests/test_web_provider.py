"""Tests for SE-layer web search wrapper (ai/ package must stay unchanged)."""
from __future__ import annotations

import pytest

from src.services.web_provider import ResilientDuckDuckGoProvider, _items_to_sources


def test_items_to_sources_normalizes_keys():
    raw = [
        {"title": "A", "link": "https://a.com", "snippet": "text"},
        {"title": "B", "href": "https://b.com", "body": "body"},
        {"title": "skip", "href": ""},
    ]
    out = _items_to_sources(raw)
    assert len(out) == 2
    assert out[0].url == "https://a.com"
    assert out[1].snippet == "body"


@pytest.mark.asyncio
async def test_resilient_ddg_falls_back_to_next_backend(monkeypatch):
    """When lite is rate-limited, html should be tried."""
    calls: list[str] = []

    class _FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def text(self, query, max_results=3, backend="api"):
            calls.append(backend)
            if backend == "lite":
                from duckduckgo_search.exceptions import RatelimitException

                raise RatelimitException("rate limited")
            return [{"title": "R", "href": "https://x.com", "body": "snippet"}]

    monkeypatch.setattr("duckduckgo_search.DDGS", _FakeDDGS)

    out = await ResilientDuckDuckGoProvider().search("photosynthesis", max_results=1)
    assert len(out) == 1
    assert out[0].url == "https://x.com"
    assert calls[0] == "lite"
    assert "html" in calls
