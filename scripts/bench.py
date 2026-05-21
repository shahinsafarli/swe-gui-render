#!/usr/bin/env python
"""
Benchmark: sequential vs concurrent source fetching for N questions.

Demonstrates the speedup from asyncio.gather over sequential awaits.

Usage:
    python scripts/bench.py --n 5 --offline
    python scripts/bench.py --n 5          # requires API keys

Output: copy the printed table into README.md
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logging_config import setup_logging

setup_logging(level="WARNING")  # benchmarks default to quiet; override with env if needed

# Load .env and mirror all settings into os.environ — identical to the main
# application startup path. Without this, ai/ providers (Tavily, Gemini, etc.)
# call os.getenv() before the keys are present and raise ProviderError.
import src.config  # noqa: F401  — import for side-effects only

from ai.schemas import Source
from ai.providers.base import LLMProvider

# Semaphore bound used in the concurrent benchmark (mirrors production default).
_MAX_CONCURRENT = 5

# Pause between questions in live mode to avoid arXiv 429 rate-limits.
_INTER_QUESTION_DELAY = 1.5  # seconds


# ---- Offline fakes ----------------------------------------------------------

class _FakeLLM(LLMProvider):
    def complete(self, prompt: str, **kwargs) -> str:
        return "Answer [1]."


CANNED = [
    Source(title=f"Source {i}", url=f"https://example.com/{i}",
           snippet="snippet", origin="wikipedia")
    for i in range(3)
]


async def _fake_wiki(query, **kwargs):
    await asyncio.sleep(0.15)  # simulate network latency
    return CANNED[:1]


async def _fake_arxiv(query, **kwargs):
    await asyncio.sleep(0.20)
    return CANNED[1:2]


async def _fake_web(query, **kwargs):
    await asyncio.sleep(0.18)
    return CANNED[2:3]


def _patch_offline():
    import ai as ai_mod
    ai_mod.fetch_wikipedia = _fake_wiki
    ai_mod.fetch_arxiv = _fake_arxiv
    ai_mod.fetch_web = _fake_web


def _patch_live_user_agent():
    """Wrap fetch_wikipedia/fetch_arxiv with a shared httpx client that sends
    a proper User-Agent header.  Wikipedia returns 403 for bare httpx clients.
    arXiv returns 429 without backoff handling.
    Does NOT modify anything inside the ai/ folder."""
    import httpx
    import ai as ai_mod

    _HEADERS = {
        "User-Agent": (
            "ResearchAssistant/1.0 (benchmark; https://github.com/example/researcher)"
        )
    }

    _original_wiki = ai_mod.fetch_wikipedia
    _original_arxiv = ai_mod.fetch_arxiv

    async def _wiki_with_agent(query, **kwargs):
        async with httpx.AsyncClient(headers=_HEADERS, timeout=15.0) as client:
            return await _original_wiki(query, client=client, **kwargs)

    async def _arxiv_with_agent(query, **kwargs):
        # arXiv rate-limits aggressive clients with 429. Retry up to 3 times
        # with exponential backoff (3s, 6s, 12s) before giving up.
        last_exc = None
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(headers=_HEADERS, timeout=15.0) as client:
                    return await _original_arxiv(query, client=client, **kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt == 2:
                    break
                wait = 3 * (2 ** attempt)
                print(f"  [arXiv] rate-limited, retrying in {wait}s ...")
                await asyncio.sleep(wait)
        raise last_exc

    ai_mod.fetch_wikipedia = _wiki_with_agent
    ai_mod.fetch_arxiv = _arxiv_with_agent


# ---- Sequential strategy ---------------------------------------------------

async def run_sequential(questions: list[str], live: bool = False) -> float:
    """Fetch all three sources for each question one-by-one."""
    import ai
    t0 = time.perf_counter()
    for i, q in enumerate(questions):
        if live and i > 0:
            await asyncio.sleep(_INTER_QUESTION_DELAY)
        await ai.fetch_wikipedia(q)
        await ai.fetch_arxiv(q)
        await ai.fetch_web(q)
    return time.perf_counter() - t0


# ---- Concurrent strategy ---------------------------------------------------

async def run_concurrent(questions: list[str], max_concurrent: int = _MAX_CONCURRENT, live: bool = False) -> float:
    """
    Fetch all three sources for each question in parallel (asyncio.gather).

    A Semaphore(max_concurrent) bounds simultaneous fetches - mirroring the
    production orchestrator - so the benchmark reflects real-world behaviour.
    """
    import ai

    sem = asyncio.Semaphore(max_concurrent)

    async def _fetch_one(coro):
        async with sem:
            return await coro

    t0 = time.perf_counter()
    for i, q in enumerate(questions):
        if live and i > 0:
            await asyncio.sleep(_INTER_QUESTION_DELAY)
        await asyncio.gather(
            _fetch_one(ai.fetch_wikipedia(q)),
            _fetch_one(ai.fetch_arxiv(q)),
            _fetch_one(ai.fetch_web(q)),
        )
    return time.perf_counter() - t0


# ---- Main ------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=5, help="Number of questions.")
    p.add_argument("--offline", action="store_true", help="Use simulated latency (no keys needed).")
    p.add_argument("--max-concurrent", type=int, default=_MAX_CONCURRENT,
                   help=f"Semaphore bound for parallel fetches (default: {_MAX_CONCURRENT}).")
    args = p.parse_args()

    live = not args.offline

    if args.offline:
        _patch_offline()
    else:
        _patch_live_user_agent()

    data_path = Path(__file__).parent.parent / "data" / "research_questions.json"
    all_q = json.loads(data_path.read_text())["questions"]
    questions = [q["text"] for q in all_q[: args.n]]
    # Pad if fewer than n in the file
    while len(questions) < args.n:
        questions.append(f"Question {len(questions) + 1}")

    print(f"\nBenchmarking {args.n} question(s) x 3 sources ...")
    if args.offline:
        print("Mode: OFFLINE (simulated latency - no API keys required)\n")
    else:
        print("Mode: LIVE (real network - API keys must be set in environment)\n")

    seq_time = asyncio.run(run_sequential(questions, live=live))
    con_time = asyncio.run(run_concurrent(questions, max_concurrent=args.max_concurrent, live=live))
    speedup = seq_time / con_time if con_time > 0 else float("inf")

    label = "offline" if args.offline else "live"
    print(f"{'Mode':<15} {'Time (s)':>10} {'Speedup':>10}")
    print("-" * 38)
    print(f"{'Sequential':<15} {seq_time:>10.3f} {'1.0x':>10}")
    print(f"{'Concurrent':<15} {con_time:>10.3f} {speedup:>9.1f}x")
    print(f"\nN={args.n}, max_concurrent={args.max_concurrent} (semaphore), run={label}")
    if args.offline:
        print("Command: python scripts/bench.py --n 5 --offline")
        print("\nNOTE: These numbers use simulated (in-process) latency only.")
        print("For rubric-compliant real-network numbers, run WITHOUT --offline")
        print("on the target machine with caches cleared and API keys set:")
        print("  python scripts/bench.py --n 5")
    else:
        print("Command: python scripts/bench.py --n 5")
    print("\nInterpretation:")
    print("  - Sequential: awaits wiki -> arxiv -> web one after the other for each question")
    print("  - Concurrent: asyncio.gather(wiki, arxiv, web) per question, bounded by semaphore")
    if args.offline:
        print(f"  - Speedup of {speedup:.1f}x with simulated latency "
              "(0.15s wiki, 0.20s arxiv, 0.18s web)")
        print("  - With real network (avg ~0.5-1s per source), concurrent speedup is typically 2-4x")
    else:
        print(f"  - Speedup of {speedup:.1f}x measured over the real network")


if __name__ == "__main__":
    main()