"""
Shared pytest fixtures for Topic 4 SE-layer tests.

Rules:
- NO real network calls — all ai.* and HTTP are mocked.
- The smoke-test fixtures (FakeLLM, FakeWebSearch, sample_sources) are
  re-exported here so test files only need to import from conftest.
- The no_real_ai autouse fixture ensures production AI is never called.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ai.providers.base import LLMProvider
from ai.sources import WebSearchProvider
from ai.schemas import Source

# ---- Re-export smoke-test fakes so every test file can use them ------------


class FakeLLM(LLMProvider):
    """Deterministic LLM that records every prompt it receives."""

    def __init__(self, response: str | None = None) -> None:
        self.response = response or (
            "Photosynthesis converts light into chemical energy [1]. "
            "The two main stages are the light reactions and the Calvin cycle [2]."
        )
        self.calls: list[str] = []

    def complete(
        self,
        prompt: str,
        *,
        json_schema: dict | None = None,
        max_tokens: int = 1024,
    ) -> str:
        self.calls.append(prompt)
        return self.response


class FakeWebSearch(WebSearchProvider):
    """Returns canned web sources without touching the network."""

    def __init__(self, results: list[Source] | None = None) -> None:
        self.results: list[Source] = results or [
            Source(
                title="Photosynthesis — Overview",
                url="https://example.com/photosynthesis",
                snippet="A biological process used by plants.",
                origin="web",
            )
        ]
        self.calls: list[str] = []

    async def search(
        self,
        query: str,
        *,
        max_results: int = 3,
        client: Any = None,
    ) -> list[Source]:
        self.calls.append(query)
        return self.results[:max_results]


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def fake_web() -> FakeWebSearch:
    return FakeWebSearch()


@pytest.fixture
def sample_sources() -> list[Source]:
    return [
        Source(
            title="Photosynthesis (Wikipedia)",
            url="https://en.wikipedia.org/wiki/Photosynthesis",
            snippet="Photosynthesis is a process used by plants to convert light energy.",
            origin="wikipedia",
        ),
        Source(
            title="Calvin cycle (Wikipedia)",
            url="https://en.wikipedia.org/wiki/Calvin_cycle",
            snippet="The Calvin cycle is a series of biochemical redox reactions.",
            origin="wikipedia",
        ),
    ]


# ---- Async wikipedia / arxiv fakes -----------------------------------------


async def _fake_fetch_wikipedia(query: str, **kwargs) -> list[Source]:
    return [
        Source(
            title=f"Wikipedia: {query}",
            url=f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
            snippet=f"Wikipedia snippet about {query}.",
            origin="wikipedia",
        )
    ]


async def _fake_fetch_arxiv(query: str, **kwargs) -> list[Source]:
    return [
        Source(
            title=f"arXiv paper on {query}",
            url=f"https://arxiv.org/abs/0000.{abs(hash(query)) % 99999:05d}",
            snippet=f"Abstract about {query}.",
            origin="arxiv",
        )
    ]


@pytest.fixture
def fake_fetch_wikipedia():
    return _fake_fetch_wikipedia


@pytest.fixture
def fake_fetch_arxiv():
    return _fake_fetch_arxiv


# ---- autouse: block real AI calls -----------------------------------------


@pytest.fixture(autouse=True)
def no_real_ai(monkeypatch):
    """
    Patch all ai.* network functions so tests never hit real APIs.
    Tests that need custom return values should override via monkeypatch
    or pass explicit llm= / provider= kwargs.
    """
    monkeypatch.setattr(
        "ai.fetch_wikipedia",
        _fake_fetch_wikipedia,
    )
    monkeypatch.setattr(
        "ai.fetch_arxiv",
        _fake_fetch_arxiv,
    )
    # ai.fetch_web is left to individual tests via provider= kwarg or a
    # separate monkeypatch, since web provider is pluggable.
    yield


# ---- Cache isolation -------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the in-memory cache before and after each test."""
    from src.services import cache
    cache.clear()
    yield
    cache.clear()
