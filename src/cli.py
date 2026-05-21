"""
CLI entry point for the Async Research Assistant.

Usage:
    python -m researcher ask "What is photosynthesis?"
    python -m researcher ask "quantum computing basics" --sources wiki,arxiv
    python -m researcher ask "latest LLM research" --no-cache
    python -m researcher history --limit 5

    # Or directly (sys.path is fixed automatically):
    python src/cli.py ask "What is photosynthesis?"
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
from pathlib import Path

import click

# Ensure project root is on sys.path when run as `python src/cli.py`
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Windows: use SelectorEventLoop so psycopg async works correctly.
# Must be set BEFORE the first asyncio.run() call.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mGKHF]")


def _clean(text: str) -> str:
    """Remove ANSI escape sequences from a string."""
    return _ANSI_RE.sub("", text)


def _render_result(result) -> str:
    """Render a ResearchResult as human-readable text."""
    lines: list[str] = []

    if result.cached:
        lines.append("(from cache)")

    lines.append(f"Q: {_clean(result.question)}")
    lines.append("")
    lines.append(f"A: {_clean(result.answer)}")

    if result.citations:
        lines.append("")
        lines.append("References:")
        for c in result.citations:
            lines.append(f"  [{c.index}] ({_clean(c.origin)}) {_clean(c.title)}")
            lines.append(f"      {_clean(c.url)}")

    if result.sources_failed:
        lines.append("")
        lines.append(f"  [!] Sources that failed: {', '.join(result.sources_failed)}")

    lines.append("")
    lines.append(f"  [took {result.elapsed_seconds:.2f}s]")
    return "\n".join(lines)


def _parse_sources(sources_str: str | None):
    """Parse comma-separated source names into SourceFilter list or None."""
    if not sources_str:
        return None
    from src.models import SourceFilter
    result = []
    for part in sources_str.split(","):
        part = part.strip().lower()
        try:
            result.append(SourceFilter(part))
        except ValueError:
            raise click.BadParameter(
                f"Unknown source '{part}'. Valid: wiki, arxiv, web"
            )
    return result or None


@click.group()
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help="Logging verbosity.",
)
def cli(log_level: str) -> None:
    """Async Research Assistant — query Wikipedia, arXiv, and the web in parallel."""
    from src.logging_config import setup_logging

    setup_logging(level=log_level)


@cli.command("ask")
@click.argument("question")
@click.option(
    "--sources",
    default=None,
    help="Comma-separated subset: wiki,arxiv,web (default: all three).",
)
@click.option(
    "--no-cache",
    "no_cache",
    is_flag=True,
    default=False,
    help="Bypass the TTL cache and fetch fresh results.",
)
@click.option(
    "--json-output",
    "json_output",
    is_flag=True,
    default=False,
    help="Output results as JSON.",
)
def ask(question: str, sources: str | None, no_cache: bool, json_output: bool) -> None:
    """
    Ask a research question and receive a cited answer.

    Example:\n
        python -m researcher ask "What is photosynthesis?"\n
        python -m researcher ask "LLM context windows" --sources wiki,arxiv\n
        python -m researcher ask "fusion energy 2024" --no-cache
    """
    from src.models import ResearchRequest
    from src.core.researcher import run_research

    try:
        source_filters = _parse_sources(sources)
    except click.BadParameter as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)

    try:
        request = ResearchRequest(
            question=question,
            sources=source_filters,
            no_cache=no_cache,
        )
    except Exception as exc:
        click.echo(f"Validation error: {exc}", err=True)
        sys.exit(1)

    async def _run_ask():
        from src.storage.repository import init_db
        await init_db()
        return await run_research(request)

    try:
        result = asyncio.run(_run_ask())
    except Exception as exc:
        logger.error("ask_failed", extra={"error": str(exc)})
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    if json_output:
        click.echo(
            json.dumps(
                {
                    "question": result.question,
                    "answer": result.answer,
                    "citations": [c.model_dump() for c in result.citations],
                    "sources_failed": result.sources_failed,
                    "cached": result.cached,
                    "elapsed_seconds": result.elapsed_seconds,
                },
                indent=2,
            )
        )
    else:
        click.echo(_render_result(result))


@cli.command("history")
@click.option("--limit", default=10, show_default=True, help="Number of sessions to show.")
def history(limit: int) -> None:
    """Show the most recent research sessions stored in the database."""
    from src.storage.repository import init_db, repository

    async def _run():
        await init_db()
        return await repository.get_sessions(limit=limit)

    rows = asyncio.run(_run())
    if not rows:
        click.echo("No sessions found (database may not be configured).")
        return

    for row in rows:
        ts = row.get("created_at", "?")
        q = row.get("question", "")[:70]
        elapsed = row.get("elapsed_seconds", 0)
        cached_flag = " [cached]" if row.get("cached") else ""
        click.echo(f"[{row['id']}] {ts}{cached_flag} ({elapsed:.2f}s)")
        click.echo(f"     {q}")
        click.echo()


if __name__ == "__main__":
    cli()
