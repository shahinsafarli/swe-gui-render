"""CLI tests using Click's test runner — all offline."""
from __future__ import annotations

import json
import pytest
from click.testing import CliRunner

from src.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def no_db(monkeypatch):
    """Patch init_db to a no-op so CLI tests never touch a real database."""
    import src.storage.repository as repo_mod

    async def _noop():
        return None

    monkeypatch.setattr(repo_mod, "init_db", _noop)


def test_cli_help(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "ask" in result.output


def test_ask_help(runner):
    result = runner.invoke(cli, ["ask", "--help"])
    assert result.exit_code == 0


def test_ask_empty_question(runner):
    """Blank question should exit non-zero."""
    result = runner.invoke(cli, ["ask", "   "])
    assert result.exit_code != 0


def test_ask_bad_source(runner):
    """Unknown source name should exit non-zero."""
    result = runner.invoke(cli, ["ask", "test", "--sources", "tiktok"])
    assert result.exit_code != 0


def test_ask_runs_offline(runner, fake_llm, fake_web, monkeypatch):
    """ask command produces output for a valid question."""
    from src.core import researcher
    from src.models import ResearchResult, CitationRecord

    async def _fake_research(req, **kwargs):
        return ResearchResult(
            question=req.question,
            answer="Photosynthesis converts light [1].",
            citations=[CitationRecord(index=1, title="Wiki", url="https://wiki.org", origin="wikipedia")],
            elapsed_seconds=0.5,
        )

    monkeypatch.setattr(researcher, "run_research", _fake_research)

    result = runner.invoke(cli, ["ask", "What is photosynthesis?"])
    assert result.exit_code == 0
    assert "Photosynthesis" in result.output
    assert "[1]" in result.output


def test_ask_json_output(runner, monkeypatch):
    """--json-output flag produces valid JSON."""
    from src.core import researcher
    from src.models import ResearchResult

    async def _fake_research(req, **kwargs):
        return ResearchResult(
            question=req.question,
            answer="An answer [1].",
            citations=[],
            elapsed_seconds=0.1,
        )

    monkeypatch.setattr(researcher, "run_research", _fake_research)

    result = runner.invoke(cli, ["ask", "What is X?", "--json-output"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "question" in data
    assert "answer" in data


def test_ask_no_cache_flag(runner, monkeypatch):
    """--no-cache flag is forwarded to run_research."""
    from src.core import researcher
    from src.models import ResearchResult, ResearchRequest

    captured = {}

    async def _fake_research(req: ResearchRequest, **kwargs):
        captured["no_cache"] = req.no_cache
        return ResearchResult(
            question=req.question,
            answer="Answer.",
            elapsed_seconds=0.0,
        )

    monkeypatch.setattr(researcher, "run_research", _fake_research)

    runner.invoke(cli, ["ask", "test question", "--no-cache"])
    assert captured.get("no_cache") is True


def test_ask_init_db_called(runner, monkeypatch):
    """ask command calls init_db() on startup even when DB is unavailable."""
    from src.core import researcher
    from src.models import ResearchResult
    import src.storage.repository as repo_mod

    init_db_calls: list[int] = []

    async def _recording_init_db():
        init_db_calls.append(1)

    monkeypatch.setattr(repo_mod, "init_db", _recording_init_db)

    async def _fake_research(req, **kwargs):
        return ResearchResult(question=req.question, answer="A.", elapsed_seconds=0.0)

    monkeypatch.setattr(researcher, "run_research", _fake_research)

    runner.invoke(cli, ["ask", "test question"])
    assert len(init_db_calls) == 1, "init_db must be called exactly once per ask"


def test_render_result_strips_ansi(runner, monkeypatch):
    """_render_result must strip ANSI escape codes from all displayed fields."""
    from src.cli import _render_result
    from src.models import ResearchResult, CitationRecord

    ansi_answer = "\x1b[31mRed text answer [1].\x1b[0m"
    ansi_title = "\x1b[1mBold Title\x1b[0m"

    result = ResearchResult(
        question="Test?",
        answer=ansi_answer,
        citations=[CitationRecord(index=1, title=ansi_title, url="https://x.com", origin="web")],
        elapsed_seconds=0.1,
    )

    rendered = _render_result(result)

    assert "\x1b[" not in rendered, "ANSI escape codes must be stripped from rendered output"
    assert "Red text answer" in rendered
    assert "Bold Title" in rendered


def test_history_no_db(runner, monkeypatch):
    """history command shows a message when database is unavailable."""
    from src.storage import repository as repo_module

    async def _empty_sessions(limit=10):
        return []

    monkeypatch.setattr(repo_module.repository, "get_sessions", _empty_sessions)

    result = runner.invoke(cli, ["history"])
    assert result.exit_code == 0
    assert "No sessions" in result.output


def test_history_shows_sessions(runner, monkeypatch):
    """history command renders rows returned by the repository."""
    from src.storage import repository as repo_module

    async def _fake_sessions(limit=10):
        return [
            {
                "id": 1,
                "created_at": "2026-05-16T10:00:00+00:00",
                "question": "What is photosynthesis?",
                "elapsed_seconds": 1.23,
                "cached": False,
            }
        ]

    monkeypatch.setattr(repo_module.repository, "get_sessions", _fake_sessions)

    result = runner.invoke(cli, ["history", "--limit", "5"])
    assert result.exit_code == 0
    assert "photosynthesis" in result.output
    assert "[1]" in result.output


def test_history_init_db_called(runner, monkeypatch):
    """history command calls init_db() on startup."""
    from src.storage import repository as repo_module
    import src.storage.repository as repo_mod

    init_db_calls: list[int] = []

    async def _recording_init_db():
        init_db_calls.append(1)

    monkeypatch.setattr(repo_mod, "init_db", _recording_init_db)

    async def _empty_sessions(limit=10):
        return []

    monkeypatch.setattr(repo_module.repository, "get_sessions", _empty_sessions)

    runner.invoke(cli, ["history"])
    assert len(init_db_calls) == 1, "init_db must be called exactly once per history"
