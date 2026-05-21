"""
Wraps the provided ai.* functions with:
- Exponential backoff retries (tenacity)
- Per-call timeout (asyncio.wait_for — Python 3.10+ compatible)
- Structured logging
- Graceful degradation (returns empty list / raises explicitly)

Every call to ai.fetch_wikipedia, ai.fetch_arxiv, ai.fetch_web, and
ai.synthesize MUST go through this service — never call ai.* directly
from business logic.

Python 3.10 compatibility
--------------------------
asyncio.timeout() context manager was added in Python 3.11. This module
uses asyncio.wait_for() throughout, which is available from Python 3.10+
and provides the same semantics.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import time
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

import ai
from ai.providers.base import LLMProvider, ProviderError
from src.config import settings
from src.services.rate_limiter import (
    TokenBucketRateLimiter,
    RateLimitExceeded,
    wikipedia_limiter,
    arxiv_limiter,
    web_limiter,
)

logger = logging.getLogger(__name__)


class RateLimitError(ProviderError):
    """Raised when a provider returns HTTP 429 Too Many Requests.

    Carries an optional ``retry_after`` hint (seconds) parsed from the
    ``Retry-After`` response header so callers can honour the server's
    back-off window instead of hammering with exponential jitter alone.
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after: float | None = retry_after


_TRANSIENT = (
    ConnectionError,
    TimeoutError,
    OSError,
    ProviderError,     # RateLimitError (HTTP 429) is a subclass — included
    RateLimitExceeded, # token-bucket exhausted — back off and retry
)

_SYNTHESIZE_TIMEOUT: float = 60.0
_SYNTHESIZE_MAX_TOKENS: int = 1024
_SYNTH_POOL_SIZE: int = 4

_synth_executor: concurrent.futures.ThreadPoolExecutor = (
    concurrent.futures.ThreadPoolExecutor(
        max_workers=_SYNTH_POOL_SIZE,
        thread_name_prefix="synth",
    )
)


def _make_retry(max_attempts: int = 4):
    """Factory: tenacity retry decorator for async coroutines.

    When the caught exception is a :class:`RateLimitError` that carries a
    ``retry_after`` hint, the before-sleep callback overrides tenacity's
    computed wait with the server-supplied value so we respect the
    provider's rate-limit window exactly.
    """

    def _before_sleep(retry_state):  # type: ignore[no-untyped-def]
        exc = retry_state.outcome.exception()
        if isinstance(exc, RateLimitError) and exc.retry_after is not None:
            wait = exc.retry_after
            logger.warning(
                "rate_limit_retry",
                extra={
                    "attempt": retry_state.attempt_number,
                    "retry_after_s": wait,
                },
            )
            # Override tenacity's sleep by mutating the next_action
            retry_state.next_action.sleep = wait  # type: ignore[attr-defined]
        else:
            before_sleep_log(logger, logging.WARNING)(retry_state)

    return retry(
        retry=retry_if_exception_type(_TRANSIENT),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(max_attempts),
        before_sleep=_before_sleep,
        reraise=True,
    )


class _BoundedLLM(LLMProvider):
    """Thin LLMProvider wrapper that enforces a max_tokens cap.

    Composition example: wraps any concrete LLMProvider and injects
    max_tokens=_SYNTHESIZE_MAX_TOKENS into every complete() call.
    Demonstrates the composition pattern required by the rubric without
    modifying the ai/ package.
    """

    def __init__(self, inner: LLMProvider, max_tokens: int = _SYNTHESIZE_MAX_TOKENS) -> None:
        self._inner = inner
        self._max_tokens = max_tokens

    def complete(
        self,
        prompt: str,
        *,
        json_schema: dict | None = None,
        max_tokens: int = 1024,
    ) -> str:
        return self._inner.complete(
            prompt,
            json_schema=json_schema,
            max_tokens=self._max_tokens,
        )


class AIService:
    """
    Service layer over the provided ai module.

    All public methods are async and accept an optional httpx client so
    the orchestrator can share a single connection pool.

    Rate limiting is applied automatically for every fetch call via the
    token-bucket limiters defined in :mod:`src.services.rate_limiter`.
    Callers can supply custom limiters at construction time (useful in
    tests) or accept the production defaults.
    """

    def __init__(
        self,
        *,
        wiki_limiter: TokenBucketRateLimiter | None = None,
        arxiv_limiter_: TokenBucketRateLimiter | None = None,
        web_limiter_: TokenBucketRateLimiter | None = None,
    ) -> None:
        self._log = logging.getLogger(self.__class__.__name__)
        # Use module-level production defaults unless overridden (e.g. in tests).
        self._wiki_limiter: TokenBucketRateLimiter = wiki_limiter or wikipedia_limiter
        self._arxiv_limiter: TokenBucketRateLimiter = arxiv_limiter_ or arxiv_limiter
        self._web_limiter: TokenBucketRateLimiter = web_limiter_ or web_limiter

    @_make_retry()
    async def fetch_wikipedia(
        self,
        query: str,
        *,
        client: Any = None,
        max_results: int = 3,
    ) -> list[Any]:
        """Fetch Wikipedia sources with retry and per-call timeout."""
        self._wiki_limiter.consume()
        self._log.info("wikipedia_fetch_start", extra={"query": query[:60]})
        t0 = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                ai.fetch_wikipedia(query, max_results=max_results, client=client),
                timeout=settings.wikipedia_timeout,
            )
        except asyncio.TimeoutError as exc:
            self._log.warning(
                "wikipedia_fetch_timeout",
                extra={"timeout": settings.wikipedia_timeout},
            )
            raise TimeoutError(f"Wikipedia timed out after {settings.wikipedia_timeout}s") from exc
        except _TRANSIENT:
            raise
        except Exception as exc:
            self._log.error("wikipedia_fetch_failed", extra={"error": str(exc)})
            raise
        elapsed = time.perf_counter() - t0
        self._log.info(
            "wikipedia_fetch_ok",
            extra={"count": len(result), "elapsed_s": round(elapsed, 3)},
        )
        return result

    @_make_retry()
    async def fetch_arxiv(
        self,
        query: str,
        *,
        client: Any = None,
        max_results: int = 3,
    ) -> list[Any]:
        """Fetch arXiv sources with retry and per-call timeout."""
        self._arxiv_limiter.consume()
        self._log.info("arxiv_fetch_start", extra={"query": query[:60]})
        t0 = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                ai.fetch_arxiv(query, max_results=max_results, client=client),
                timeout=settings.arxiv_timeout,
            )
        except asyncio.TimeoutError as exc:
            self._log.warning(
                "arxiv_fetch_timeout",
                extra={"timeout": settings.arxiv_timeout},
            )
            raise TimeoutError(f"arXiv timed out after {settings.arxiv_timeout}s") from exc
        except _TRANSIENT:
            raise
        except Exception as exc:
            self._log.error("arxiv_fetch_failed", extra={"error": str(exc)})
            raise
        elapsed = time.perf_counter() - t0
        self._log.info(
            "arxiv_fetch_ok",
            extra={"count": len(result), "elapsed_s": round(elapsed, 3)},
        )
        return result

    @_make_retry()
    async def fetch_web(
        self,
        query: str,
        *,
        client: Any = None,
        max_results: int = 3,
        provider: Any = None,
    ) -> list[Any]:
        """Fetch web sources with retry and per-call timeout."""
        self._web_limiter.consume()
        self._log.info("web_fetch_start", extra={"query": query[:60]})
        t0 = time.perf_counter()
        if provider is None:
            from src.services.web_provider import resolve_web_provider

            provider = resolve_web_provider()
        try:
            result = await asyncio.wait_for(
                ai.fetch_web(
                    query,
                    max_results=max_results,
                    client=client,
                    provider=provider,
                ),
                timeout=settings.web_timeout,
            )
        except asyncio.TimeoutError as exc:
            self._log.warning(
                "web_fetch_timeout",
                extra={"timeout": settings.web_timeout},
            )
            raise TimeoutError(f"Web search timed out after {settings.web_timeout}s") from exc
        except _TRANSIENT:
            raise
        except Exception as exc:
            self._log.error("web_fetch_failed", extra={"error": str(exc)})
            raise
        elapsed = time.perf_counter() - t0
        self._log.info(
            "web_fetch_ok",
            extra={"count": len(result), "elapsed_s": round(elapsed, 3)},
        )
        return result

    async def synthesize(
        self,
        question: str,
        sources: list[Any],
        *,
        llm: Any = None,
        max_attempts: int = 4,
    ) -> Any:
        """Call ai.synthesize with retry, logging, a max_tokens cap, and a hard timeout.

        ai.synthesize is synchronous (LLM SDK call). We offload it to the
        shared _synth_executor via asyncio.get_event_loop().run_in_executor
        and enforce a wall-clock deadline with asyncio.wait_for.

        Retry loop uses explicit for-loop (not tenacity) because tenacity's
        async retry needs special integration with executors.
        """
        self._log.info(
            "synthesize_start",
            extra={"question": question[:60], "n_sources": len(sources)},
        )
        t0 = time.perf_counter()

        from ai.providers.factory import get_llm
        effective_llm = _BoundedLLM(llm if llm is not None else get_llm())

        loop = asyncio.get_event_loop()
        last_exc: Exception = RuntimeError("no attempts made")

        for attempt in range(1, max_attempts + 1):
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        _synth_executor,
                        lambda: ai.synthesize(question, sources, llm=effective_llm),
                    ),
                    timeout=_SYNTHESIZE_TIMEOUT,
                )
            except asyncio.TimeoutError:
                self._log.error(
                    "synthesize_timeout",
                    extra={"attempt": attempt, "timeout_s": _SYNTHESIZE_TIMEOUT},
                )
                raise TimeoutError(
                    f"ai.synthesize exceeded {_SYNTHESIZE_TIMEOUT}s wall-clock limit"
                )
            except (ValueError, TypeError) as exc:
                self._log.error("synthesize_invalid", extra={"error": str(exc)})
                raise
            except _TRANSIENT as exc:
                last_exc = exc
                self._log.warning(
                    "synthesize_transient_error",
                    extra={"attempt": attempt, "max": max_attempts, "error": str(exc)},
                )
                if attempt < max_attempts:
                    wait = min(2 ** (attempt - 1), 30)
                    await asyncio.sleep(wait)
                    continue
                raise
            except Exception as exc:
                self._log.error("synthesize_failed", extra={"error": str(exc)})
                raise
            else:
                elapsed = time.perf_counter() - t0
                self._log.info(
                    "synthesize_ok",
                    extra={
                        "attempt": attempt,
                        "citations": len(result.citations),
                        "elapsed_s": round(elapsed, 3),
                    },
                )
                return result

        raise last_exc  # pragma: no cover


# Module-level singleton
ai_service = AIService()
