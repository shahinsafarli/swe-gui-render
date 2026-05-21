"""
Concurrent source orchestration.

All three sources are queried simultaneously via asyncio.gather.
Each source has its own timeout enforced via asyncio.wait_for so a
slow source cannot hold a semaphore slot past its absolute wall-clock
budget even if retries keep firing.

Compatible with Python 3.10+ (asyncio.timeout was added in 3.11;
this module uses asyncio.wait_for throughout for broad compatibility).

Returns:
    fetched: dict[source_name, list[Source]]  — successful results
    failed:  list[source_name]                — sources that errored / timed out
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.config import settings
from src.services.ai_service import ai_service

logger = logging.getLogger(__name__)

_FETCH_FNS = {
    "wikipedia": ai_service.fetch_wikipedia,
    "arxiv": ai_service.fetch_arxiv,
    "web": ai_service.fetch_web,
}

_SOURCE_TIMEOUT_ATTR: dict[str, str] = {
    "wikipedia": "wikipedia_timeout",
    "arxiv": "arxiv_timeout",
    "web": "web_timeout",
}


async def _fetch_one(
    source_name: str,
    question: str,
    *,
    client: Any = None,
    web_provider: Any = None,
) -> tuple[str, list[Any]]:
    """Fetch from one source with an absolute wall-clock timeout.

    Uses asyncio.wait_for (Python 3.10 compatible) to enforce the budget.
    Budget = per-attempt timeout * max_retry_attempts + 1 s slack.
    """
    fn = _FETCH_FNS[source_name]
    kwargs: dict[str, Any] = {"client": client}
    if source_name == "web" and web_provider is not None:
        kwargs["provider"] = web_provider

    per_attempt: float = getattr(settings, _SOURCE_TIMEOUT_ATTR[source_name])
    # Web may try several DDG backends with short backoff between rate limits.
    if source_name == "web":
        outer_timeout = per_attempt * 6 + 5.0
    else:
        outer_timeout = per_attempt * 4 + 1.0

    async def _do_fetch() -> tuple[str, list[Any]]:
        sources = await fn(question, **kwargs)
        return source_name, sources

    return await asyncio.wait_for(_do_fetch(), timeout=outer_timeout)


async def fetch_sources_concurrent(
    question: str,
    *,
    sources: set[str] | None = None,
    client: Any = None,
    web_provider: Any = None,
    max_concurrent: int | None = None,
) -> tuple[dict[str, list[Any]], list[str]]:
    """
    Query the requested sources simultaneously with bounded concurrency.

    Parameters
    ----------
    question:
        The research question string.
    sources:
        Which sources to query. Defaults to all three.
    client:
        Shared httpx.AsyncClient passed through to each fetch function so
        the whole pipeline reuses a single connection pool.
    web_provider:
        Optional WebSearchProvider override (tests inject fakes here).
    max_concurrent:
        Max simultaneous fetches. Defaults to settings.max_concurrent.

    Returns
    -------
    (fetched, failed)
        fetched: source_name -> list[Source]
        failed:  list of source names that errored
    """
    active_set = sources if sources is not None else {"wikipedia", "arxiv", "web"}
    active_list: list[str] = sorted(active_set)

    limit = max_concurrent or settings.max_concurrent
    sem = asyncio.Semaphore(limit)

    async def _guarded(src_name: str) -> tuple[str, list[Any]]:
        async with sem:
            return await _fetch_one(
                src_name,
                question,
                client=client,
                web_provider=web_provider,
            )

    tasks = [_guarded(s) for s in active_list]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    fetched: dict[str, list[Any]] = {}
    failed: list[str] = []

    for src_name, result in zip(active_list, results):
        if isinstance(result, BaseException):
            logger.warning(
                "source_fetch_failed",
                extra={"source": src_name, "error": str(result)},
            )
            failed.append(src_name)
        else:
            _, src_list = result
            fetched[src_name] = src_list
            logger.info(
                "source_fetch_ok",
                extra={"source": src_name, "count": len(src_list)},
            )

    if failed:
        logger.warning(
            "research_degraded",
            extra={"failed": failed, "succeeded": list(fetched.keys())},
        )

    return fetched, failed


async def run_pipeline(
    coroutines: list[Any],
    max_concurrent: int | None = None,
) -> list[Any]:
    """
    Generic bounded-concurrency runner.
    Returns results in input order; exceptions returned as values (not raised).
    """
    limit = max_concurrent or settings.max_concurrent
    sem = asyncio.Semaphore(limit)

    async def _guarded(coro: Any) -> Any:
        async with sem:
            return await coro

    results = await asyncio.gather(
        *[_guarded(c) for c in coroutines],
        return_exceptions=True,
    )
    errors = [r for r in results if isinstance(r, BaseException)]
    if errors:
        logger.warning(
            "pipeline_partial_failure",
            extra={"total": len(coroutines), "errors": len(errors)},
        )
    return list(results)
