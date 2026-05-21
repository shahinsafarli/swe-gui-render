#!/usr/bin/env python
"""
End-to-end demo: runs all 5 questions from data/research_questions.json.

Usage:
    python scripts/demo.py               # real providers (requires .env)
    python scripts/demo.py --offline     # no network, no API keys
    python scripts/demo.py --limit 2     # first 2 questions only
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Make sure src/ and ai/ are importable from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logging_config import setup_logging

setup_logging()

from src.core.researcher import run_research
from src.models import ResearchRequest
from src.services import cache as cache_store


def _offline_setup():
    """Inject fake AI modules for offline demo."""
    import ai as ai_mod
    from ai.schemas import Source
    from ai.providers.base import LLMProvider

    class _FakeLLM(LLMProvider):
        def complete(self, prompt: str, **kwargs) -> str:
            import re
            n = len(re.findall(r"^\[(\d+)\]", prompt, re.MULTILINE))
            cited = ", ".join(f"[{i}]" for i in range(1, min(n + 1, 4)))
            return (
                f"Based on the retrieved sources, this is a synthesized answer {cited}. "
                f"The evidence from multiple references supports this conclusion [1]."
            )

    _CANNED: dict[str, list[Source]] = {
        "photosynthesis": [
            Source(title="Photosynthesis", url="https://en.wikipedia.org/wiki/Photosynthesis",
                   snippet="Process by which plants convert light energy into chemical energy.", origin="wikipedia"),
            Source(title="Light-Dependent Reactions", url="https://arxiv.org/abs/1706.03762",
                   snippet="Overview of light reactions in chloroplasts.", origin="arxiv"),
            Source(title="How Plants Make Food", url="https://example.com/plants",
                   snippet="Detailed guide to plant energy production.", origin="web"),
        ],
        "transformer": [
            Source(title="Transformer model", url="https://en.wikipedia.org/wiki/Transformer_(ML)",
                   snippet="Deep learning architecture based on self-attention.", origin="wikipedia"),
            Source(title="Attention Is All You Need", url="https://arxiv.org/abs/1706.03762",
                   snippet="The original Transformer paper by Vaswani et al.", origin="arxiv"),
        ],
        "financial crisis": [
            Source(title="2008 Financial Crisis", url="https://en.wikipedia.org/wiki/2008_crisis",
                   snippet="Systemic collapse triggered by subprime mortgage failures.", origin="wikipedia"),
            Source(title="Crisis Causes Analysis", url="https://example.com/crisis",
                   snippet="Regulatory failures and excessive leverage.", origin="web"),
        ],
        "fusion": [
            Source(title="Fusion Power", url="https://en.wikipedia.org/wiki/Fusion_power",
                   snippet="Nuclear fusion as a potential clean energy source.", origin="wikipedia"),
            Source(title="Recent Fusion Advances", url="https://arxiv.org/abs/2310.12345",
                   snippet="Progress toward ignition and net energy gain.", origin="arxiv"),
        ],
        "crispr": [
            Source(title="CRISPR-Cas9", url="https://en.wikipedia.org/wiki/CRISPR",
                   snippet="Gene editing tool based on bacterial immune system.", origin="wikipedia"),
            Source(title="CRISPR mechanism", url="https://arxiv.org/abs/2001.09876",
                   snippet="Molecular details of Cas9 cleavage.", origin="arxiv"),
        ],
    }

    def _get_canned(question: str) -> list[Source]:
        q = question.lower()
        for kw, srcs in _CANNED.items():
            if kw in q:
                return srcs
        return [Source(title="Generic", url="https://example.com", snippet=f"Generic result for: {question}", origin="web")]

    fake_llm = _FakeLLM()

    async def _fake_wiki(query, **kwargs): return _get_canned(query)[:1]
    async def _fake_arxiv(query, **kwargs): return _get_canned(query)[1:2]
    async def _fake_web(query, **kwargs): return _get_canned(query)[2:3]

    ai_mod.fetch_wikipedia = _fake_wiki
    ai_mod.fetch_arxiv = _fake_arxiv
    ai_mod.fetch_web = _fake_web

    return fake_llm


def _render(result) -> str:
    lines = ["=" * 72]
    if result.cached:
        lines.append("(from cache)")
    lines.append(f"Q: {result.question}")
    lines.append("")
    lines.append(f"A: {result.answer}")
    if result.citations:
        lines.append("")
        lines.append("References:")
        for c in result.citations:
            lines.append(f"  [{c.index}] ({c.origin}) {c.title}")
            lines.append(f"      {c.url}")
    if result.sources_failed:
        lines.append(f"\n  [!] Sources failed: {', '.join(result.sources_failed)}")
    lines.append(f"\n  [elapsed: {result.elapsed_seconds:.2f}s]")
    return "\n".join(lines)


async def main(offline: bool, limit: int) -> None:
    data_path = Path(__file__).parent.parent / "data" / "research_questions.json"
    questions = json.loads(data_path.read_text())["questions"][:limit]

    llm = _offline_setup() if offline else None

    for q in questions:
        request = ResearchRequest(question=q["text"])
        result = await run_research(request, llm=llm)
        print(_render(result))
        print()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--offline", action="store_true", help="Run without real API calls.")
    p.add_argument("--limit", type=int, default=5, help="Max questions to process.")
    args = p.parse_args()

    cache_store.clear()
    asyncio.run(main(offline=args.offline, limit=args.limit))
