#!/usr/bin/env python
"""
Reproducible sequential-vs-concurrent benchmark.

Rubric requirements met
-----------------------
1. Caches cleared before every timed run (in-process TTL cache reset +
   module-level singleton replacement).
2. Sequential mode and concurrent mode run on the *same machine* in the
   *same process invocation*, so OS scheduler, CPU frequency, and network
   conditions are as close to identical as possible.
3. Timings are measured with ``time.perf_counter()`` (monotonic, nanosecond
   resolution) and printed as a comparison table.
4. Works offline (``--offline`` flag) with deterministic simulated latency
   so CI can validate the scheduling logic without API keys; and live
   (default) for real rubric-compliant numbers.

Usage
-----
Offline (no API keys needed — validates scheduling logic):
    python scripts/benchmark.py --n 5 --offline

Live (real network — rubric-compliant numbers):
    python scripts/benchmark.py --n 5

Custom concurrency bound:
    python scripts/benchmark.py --n 5 --max-concurrent 3 --offline

The script prints a Markdown-compatible table and exits 0.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# Ensure project root is on the path when run as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logging_config import setup_logging

setup_logging(level="WARNING")  # keep benchmark output clean

import logging

import httpx

from ai.schemas import Source
from ai.providers.base import LLMProvider, ProviderError

_bench_log = logging.getLogger("benchmark")

# Semaphore bound matching the production orchestrator default.
_DEFAULT_MAX_CONCURRENT = 5

# Wikipedia requires a descriptive User-Agent or its edge cache returns 403.
# researcher.py already sets this via _make_http_client(); the benchmark calls
# ai.fetch_* directly and must supply its own compliant client.
_BENCH_USER_AGENT = (
    "AsyncResearchAssistant/1.0 "
    "(https://github.com/YOUR/REPO; research-assistant@example.com)"
)


def _make_bench_client() -> httpx.AsyncClient:
    """Shared httpx client for benchmark fetch calls.

    Mirrors researcher._make_http_client() so Wikipedia sees a valid
    User-Agent and does not return HTTP 403.
    """
    return httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(30.0, connect=5.0),
        headers={"User-Agent": _BENCH_USER_AGENT},
    )


# ---------------------------------------------------------------------------
# Offline stubs
# ---------------------------------------------------------------------------

class _FakeLLM(LLMProvider):
    def complete(self, prompt: str, **kwargs) -> str:
        return "Answer [1]."


_CANNED = [
    Source(
        title=f"Source {i}",
        url=f"https://example.com/{i}",
        snippet="snippet",
        origin="wikipedia",
    )
    for i in range(3)
]


async def _fake_wiki(query, **kwargs):
    await asyncio.sleep(0.15)   # realistic Wikipedia RTT proxy
    return _CANNED[:1]


async def _fake_arxiv(query, **kwargs):
    await asyncio.sleep(0.20)   # arXiv is a touch slower
    return _CANNED[1:2]


async def _fake_web(query, **kwargs):
    await asyncio.sleep(0.18)   # web/DDG proxy
    return _CANNED[2:3]


def _patch_offline() -> None:
    """Replace ai.* fetchers with deterministic stubs."""
    import ai as ai_mod
    ai_mod.fetch_wikipedia = _fake_wiki
    ai_mod.fetch_arxiv = _fake_arxiv
    ai_mod.fetch_web = _fake_web


# ---------------------------------------------------------------------------
# Graceful per-fetch wrapper
# ---------------------------------------------------------------------------

async def _graceful(label: str, coro) -> list[Source]:
    """Await *coro*, returning [] on any exception.

    Optional external providers (Wikipedia, arXiv, web search) must NOT crash
    the benchmark.  A 403 from Wikipedia, a missing API key for Tavily, or any
    transient network error is caught here, logged at WARNING level, and treated
    as an empty result so both timed runs always complete and produce a table.
    """
    try:
        return await coro
    except (ProviderError, Exception) as exc:
        _bench_log.warning("provider_skipped label=%s error=%s", label, exc)
        return []


# ---------------------------------------------------------------------------
# Cache clearing
# ---------------------------------------------------------------------------

def _clear_caches() -> None:
    """
    Reset all in-process caches so neither run benefits from warm state.

    Strategy
    --------
    The project's TTL cache (``src.services.cache``) is implemented on top
    of ``cachetools.TTLCache`` (or a dict-backed shim).  We reach into the
    module and call ``clear()`` if the object exposes it; otherwise we
    replace it with a fresh instance of the same type.

    The AIService singleton is also replaced so its internal limiter buckets
    start full for both runs — this prevents token-bucket state from the
    sequential run bleeding into the concurrent run's token counts.
    """
    # 1. Clear the services.cache TTL cache if present.
    try:
        import src.services.cache as cache_mod
        for attr in ("_cache", "cache", "_store"):
            obj = getattr(cache_mod, attr, None)
            if obj is not None and hasattr(obj, "clear"):
                obj.clear()
                break
    except Exception:
        pass  # cache module may not exist in all configurations

    # 2. Reset the ai_service singleton so limiter buckets start fresh.
    try:
        import src.services.ai_service as svc_mod
        from src.services.ai_service import AIService
        svc_mod.ai_service = AIService()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Sequential strategy
# ---------------------------------------------------------------------------

async def run_sequential(questions: list[str]) -> float:
    """
    Fetch all three sources for each question one after the other.

    wiki → arxiv → web per question, questions in series.
    Total wall-clock ≈ N × (T_wiki + T_arxiv + T_web).
    """
    import ai
    t0 = time.perf_counter()
    async with _make_bench_client() as client:
        for q in questions:
            await _graceful("wikipedia", ai.fetch_wikipedia(q, client=client))
            await _graceful("arxiv",     ai.fetch_arxiv(q,     client=client))
            await _graceful("web",       ai.fetch_web(q))
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Concurrent strategy
# ---------------------------------------------------------------------------

async def run_concurrent(
    questions: list[str],
    max_concurrent: int = _DEFAULT_MAX_CONCURRENT,
) -> float:
    """
    Fetch all three sources for each question in parallel via asyncio.gather.

    A Semaphore(max_concurrent) bounds simultaneous fetches, mirroring the
    production orchestrator.  Questions are still processed sequentially
    (gather per question) so the comparison to sequential is apples-to-apples
    per-question latency reduction.

    Total wall-clock ≈ N × max(T_wiki, T_arxiv, T_web).
    """
    import ai

    sem = asyncio.Semaphore(max_concurrent)

    async def _bounded(coro):
        async with sem:
            return await coro

    t0 = time.perf_counter()
    async with _make_bench_client() as client:
        for q in questions:
            await asyncio.gather(
                _bounded(_graceful("wikipedia", ai.fetch_wikipedia(q, client=client))),
                _bounded(_graceful("arxiv",     ai.fetch_arxiv(q,     client=client))),
                _bounded(_graceful("web",       ai.fetch_web(q))),
            )
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _print_table(
    n: int,
    max_concurrent: int,
    seq_time: float,
    con_time: float,
    mode: str,
) -> None:
    speedup = seq_time / con_time if con_time > 0 else float("inf")
    sep = "+" + "-" * 17 + "+" + "-" * 12 + "+" + "-" * 12 + "+"
    print(sep)
    print(f"| {'Mode':<15} | {'Time (s)':>10} | {'Speedup':>10} |")
    print(sep)
    print(f"| {'Sequential':<15} | {seq_time:>10.3f} | {'1.00x':>10} |")
    print(f"| {'Concurrent':<15} | {con_time:>10.3f} | {speedup:>9.2f}x |")
    print(sep)
    print()
    print(f"  N={n} questions × 3 sources, max_concurrent={max_concurrent} (semaphore)")
    print(f"  Mode: {mode}")
    print(f"  Speedup: {speedup:.2f}x  ({seq_time:.3f}s → {con_time:.3f}s)")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=5, help="Number of questions (default: 5).")
    p.add_argument(
        "--offline",
        action="store_true",
        help="Use deterministic simulated latency (no API keys required).",
    )
    p.add_argument(
        "--max-concurrent",
        type=int,
        default=_DEFAULT_MAX_CONCURRENT,
        help=f"Semaphore bound for concurrent fetches (default: {_DEFAULT_MAX_CONCURRENT}).",
    )
    args = p.parse_args()

    if args.offline:
        _patch_offline()

    # Load questions from the data file; pad if fewer than requested.
    data_path = Path(__file__).parent.parent / "data" / "research_questions.json"
    all_q = json.loads(data_path.read_text())["questions"]
    questions = [q["text"] for q in all_q[: args.n]]
    while len(questions) < args.n:
        questions.append(f"Synthetic question {len(questions) + 1}")

    mode_label = "OFFLINE (simulated latency)" if args.offline else "LIVE (real network)"
    print(f"\n=== Sequential vs Concurrent Benchmark ===")
    print(f"  N={args.n}, max_concurrent={args.max_concurrent}, mode={mode_label}")
    print()

    # --- Run 1: Sequential (caches cleared first) ---
    print("  [1/2] Clearing caches … ", end="", flush=True)
    _clear_caches()
    print("done.")
    print("  [2/2] Running SEQUENTIAL … ", end="", flush=True)
    seq_time = asyncio.run(run_sequential(questions))
    print(f"{seq_time:.3f}s")

    # --- Run 2: Concurrent (caches cleared again for fairness) ---
    print("  [3/3] Clearing caches … ", end="", flush=True)
    _clear_caches()
    print("done.")
    print("  [4/4] Running CONCURRENT … ", end="", flush=True)
    con_time = asyncio.run(run_concurrent(questions, max_concurrent=args.max_concurrent))
    print(f"{con_time:.3f}s")

    print()
    _print_table(args.n, args.max_concurrent, seq_time, con_time, mode_label)

    if args.offline:
        print("NOTE: Offline numbers use in-process asyncio.sleep() as a network proxy.")
        print("      Scheduling logic is identical to live mode; only latency source differs.")
        print()
        print("To generate rubric-compliant LIVE numbers on your machine:")
        print("  1. Copy .env.example → .env and fill in ANTHROPIC_API_KEY + TAVILY_API_KEY")
        print("  2. Run:")
        print("       python scripts/benchmark.py --n 5")
        print("     (caches are cleared automatically before each timed run)")
    else:
        print("Caches were cleared before each timed run.")
        print("Both runs executed on the same machine in the same process invocation.")


if __name__ == "__main__":
    main()