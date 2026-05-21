"""
SE-layer web search hardening — does NOT modify the provided ai/ package.

Wraps ai.sources.get_web_search_provider() and, for DuckDuckGo only, adds
backend fallbacks (lite → html → api) plus brief retries on rate limits.
Tavily/Serper are returned unchanged from the ai factory.

Per TOPIC.md: all production fetches still go through ai.fetch_web(..., provider=...)
via AIService; this module only supplies a hardened provider instance.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from ai.providers.base import ProviderError
from ai.schemas import Source
from ai.sources import DuckDuckGoProvider, WebSearchProvider, get_web_search_provider

logger = logging.getLogger(__name__)

_DDG_BACKENDS: tuple[str, ...] = ("lite", "html", "api")


def _items_to_sources(items: list[dict[str, Any]]) -> list[Source]:
    out: list[Source] = []
    for item in items:
        url = (item.get("href") or item.get("link") or "").strip()
        if not url:
            continue
        out.append(
            Source(
                title=item.get("title") or "(untitled)",
                url=url,
                snippet=(item.get("body") or item.get("snippet") or "").strip(),
                origin="web",
            )
        )
    return out


class ResilientDuckDuckGoProvider(DuckDuckGoProvider):
    """SE subclass of the provided DuckDuckGoProvider — ai/ source stays untouched."""

    async def search(
        self,
        query: str,
        *,
        max_results: int = 3,
        client: Any = None,
    ) -> list[Source]:
        del client
        from duckduckgo_search import DDGS  # type: ignore
        from duckduckgo_search.exceptions import (  # type: ignore
            DuckDuckGoSearchException,
            RatelimitException,
        )

        def _fetch(backend: str) -> list[dict[str, Any]]:
            with DDGS() as ddgs:
                return list(
                    ddgs.text(query, max_results=max_results, backend=backend)
                )

        last_exc: Exception | None = None
        for backend in _DDG_BACKENDS:
            for attempt in range(2):
                try:
                    raw = await asyncio.to_thread(_fetch, backend)
                    sources = _items_to_sources(raw)
                    if sources:
                        logger.info(
                            "web_ddg_ok",
                            extra={
                                "backend": backend,
                                "count": len(sources),
                                "attempt": attempt + 1,
                            },
                        )
                        return sources
                    logger.warning(
                        "web_ddg_empty",
                        extra={"backend": backend, "attempt": attempt + 1},
                    )
                except RatelimitException as exc:
                    last_exc = exc
                    logger.warning(
                        "web_ddg_rate_limited",
                        extra={"backend": backend, "attempt": attempt + 1},
                    )
                    if attempt == 0:
                        await asyncio.sleep(2.0)
                    continue
                except DuckDuckGoSearchException as exc:
                    last_exc = exc
                    logger.warning(
                        "web_ddg_backend_failed",
                        extra={"backend": backend, "error": str(exc)[:200]},
                    )
                    break

        if last_exc is not None:
            raise ProviderError(f"DuckDuckGo search failed: {last_exc}") from last_exc
        return []


def resolve_web_provider() -> WebSearchProvider:
    """Configured web provider with SE-layer DDG hardening when applicable."""
    base = get_web_search_provider()
    if type(base) is DuckDuckGoProvider:
        return ResilientDuckDuckGoProvider()
    return base
