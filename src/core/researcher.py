"""
Core research pipeline business logic.

Orchestrates:
1. Concurrent source fetching (Wikipedia + arXiv + web) via the concurrency module
2. TTL cache lookup/store
3. Source deduplication
4. LLM synthesis (offloaded to a thread pool — synthesis is synchronous)
5. Session persistence
6. Result assembly

All AI calls go through src.services.ai_service — never directly to ai.*.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any

import httpx

from src.concurrency.orchestrator import fetch_sources_concurrent
from src.config import settings
from src.models import (
    CitationRecord,
    ResearchRequest,
    ResearchResult,
    SourceFilter,
    SourceRecord,
)
from src.services.ai_service import ai_service
from src.services import cache as cache_store
from src.storage.repository import repository

logger = logging.getLogger(__name__)

_MAX_ANSWER_CHARS = 4000
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mGKHF]")

# Patterns stripped from queries before sending to Wikipedia/arXiv.
# "What is X?" → "X" gets better Wikipedia results.
_QUERY_CLEAN_RE = re.compile(
    r"^\s*(?:what is|what are|how does|explain|define|tell me about)\s+",
    re.IGNORECASE,
)


def _normalize_query(question: str) -> str:
    """Strip common question prefixes and trailing punctuation for API queries.

    Wikipedia and arXiv work better with keyword queries than with
    full English questions. "What is photosynthesis?" → "photosynthesis".
    """
    q = _QUERY_CLEAN_RE.sub("", question)
    q = q.rstrip("?.!").strip()
    return q or question  # fallback to original if we stripped everything


def _sanitize_text(text: str) -> str:
    """Strip ANSI escapes and cap length."""
    text = _ANSI_RE.sub("", text)
    if len(text) > _MAX_ANSWER_CHARS:
        text = text[:_MAX_ANSWER_CHARS] + "…"
    return text


def _dedup_sources(sources: list[Any]) -> list[Any]:
    """Remove duplicate sources by URL, preserving first-seen order."""
    seen: set[str] = set()
    out: list[Any] = []
    for src in sources:
        if src.url in seen:
            continue
        seen.add(src.url)
        out.append(src)
    return out


def _to_source_record(src: Any) -> SourceRecord:
    return SourceRecord(
        title=src.title,
        url=src.url,
        snippet=src.snippet,
        origin=src.origin,
    )


def _make_http_client() -> httpx.AsyncClient:
    """Create a shared httpx client with proper headers and redirect following.

    - follow_redirects=True: arXiv uses http:// which redirects to https://
    - User-Agent: Wikipedia API requires a descriptive User-Agent to avoid 403
    - timeout: per-source timeouts are enforced in ai_service; this is a fallback
    """
    return httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(30.0, connect=5.0),
        headers={
            "User-Agent": (
                "AsyncResearchAssistant/1.0 "
                "(https://github.com/YOUR/REPO; research-assistant@example.com)"
            )
        },
    )


async def run_research(
    request: ResearchRequest,
    *,
    llm: Any = None,
    web_provider: Any = None,
) -> ResearchResult:
    """Execute a full research pipeline for the given request."""
    t0 = time.perf_counter()
    question = request.question
    # Normalize for API queries (keep original for display)
    search_query = _normalize_query(question)

    active_sources: set[str] = set()
    if request.sources:
        for sf in request.sources:
            if sf == SourceFilter.wiki:
                active_sources.add("wikipedia")
            elif sf == SourceFilter.arxiv:
                active_sources.add("arxiv")
            elif sf == SourceFilter.web:
                active_sources.add("web")
    else:
        active_sources = {"wikipedia", "arxiv", "web"}

    logger.info(
        "research_start",
        extra={"question": question[:60], "sources": sorted(active_sources)},
    )

    # ---- Cache lookup -------------------------------------------------------
    if not request.no_cache:
        all_cached: list[Any] = []
        cache_hit_sources: list[str] = []
        active_miss: set[str] = set()

        for src_name in active_sources:
            cached = cache_store.get(src_name, search_query)
            if cached is not None:
                all_cached.extend(cached)
                cache_hit_sources.append(src_name)
            else:
                active_miss.add(src_name)

        if not active_miss:
            sources_deduped = _dedup_sources(all_cached)[: settings.max_sources]
            if not sources_deduped:
                return ResearchResult(
                    question=question,
                    answer="No sources found for this question.",
                    sources_failed=[],
                    cached=True,
                    elapsed_seconds=round(time.perf_counter() - t0, 3),
                )
            answer_obj = await ai_service.synthesize(question, sources_deduped, llm=llm)
            result = _build_result(
                question=question,
                answer_obj=answer_obj,
                sources=sources_deduped,
                sources_failed=[],
                cached=True,
                elapsed=time.perf_counter() - t0,
            )
            await _persist(result)
            return result
    else:
        all_cached = []
        cache_hit_sources = []
        active_miss = active_sources

    # ---- Concurrent source fetch --------------------------------------------
    async with _make_http_client() as client:
        fetched, failed = await fetch_sources_concurrent(
            search_query,
            sources=active_miss,
            client=client,
            web_provider=web_provider,
        )

    # Store fresh results in cache
    if not request.no_cache:
        for src_name, src_list in fetched.items():
            cache_store.set(src_name, search_query, src_list)

    combined: list[Any] = list(all_cached)
    for src_list in fetched.values():
        combined.extend(src_list)

    sources_deduped = _dedup_sources(combined)[: settings.max_sources]
    logger.info(
        "research_sources_ready",
        extra={
            "total": len(sources_deduped),
            "failed_sources": failed,
        },
    )

    if not sources_deduped:
        note = f"No sources could be retrieved (failed: {', '.join(failed) or 'none'})."
        result = ResearchResult(
            question=question,
            answer=note,
            sources_failed=failed,
            cached=False,
            elapsed_seconds=round(time.perf_counter() - t0, 3),
        )
        await _persist(result)
        return result

    # ---- Synthesis ----------------------------------------------------------
    answer_obj = await ai_service.synthesize(question, sources_deduped, llm=llm)

    result = _build_result(
        question=question,
        answer_obj=answer_obj,
        sources=sources_deduped,
        sources_failed=failed,
        cached=False,
        elapsed=time.perf_counter() - t0,
    )

    logger.info(
        "research_done",
        extra={
            "n_citations": len(result.citations),
            "n_sources": len(sources_deduped),
            "elapsed_s": result.elapsed_seconds,
            "failed": failed,
        },
    )

    await _persist(result)
    return result


async def _persist(result: ResearchResult) -> None:
    """Save result to the repository; log but do not raise on failure."""
    try:
        await repository.save_session(
            question=result.question,
            answer=result.answer,
            citations=[c.model_dump() for c in result.citations],
            sources_failed=result.sources_failed,
            cached=result.cached,
            elapsed_seconds=result.elapsed_seconds,
        )
    except Exception as exc:
        logger.warning("persist_failed", extra={"error": str(exc)})


def _build_result(
    *,
    question: str,
    answer_obj: Any,
    sources: list[Any],
    sources_failed: list[str],
    cached: bool,
    elapsed: float,
) -> ResearchResult:
    """Convert ai module objects into an application-layer ResearchResult."""
    citations = [
        CitationRecord(
            index=c.index,
            title=c.source.title,
            url=c.source.url,
            origin=c.source.origin,
        )
        for c in answer_obj.citations
    ]
    source_records = [_to_source_record(s) for s in sources]
    clean_answer = _sanitize_text(answer_obj.answer)
    return ResearchResult(
        question=question,
        answer=clean_answer,
        citations=citations,
        sources_used=source_records,
        sources_failed=sources_failed,
        cached=cached,
        elapsed_seconds=round(elapsed, 3),
    )
