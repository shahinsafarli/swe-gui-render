"""Tests for application-level Pydantic models — validation edge cases."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models import ResearchRequest, SourceFilter, ResearchResult


# ---- ResearchRequest validation --------------------------------------------


def test_valid_request():
    req = ResearchRequest(question="What is photosynthesis?")
    assert req.question == "What is photosynthesis?"
    assert req.sources is None
    assert req.no_cache is False


def test_strips_whitespace():
    req = ResearchRequest(question="  test question  ")
    assert req.question == "test question"


def test_rejects_empty():
    with pytest.raises(ValidationError):
        ResearchRequest(question="")


def test_rejects_whitespace_only():
    with pytest.raises(ValidationError):
        ResearchRequest(question="   ")


def test_rejects_too_long():
    with pytest.raises(ValidationError):
        ResearchRequest(question="x" * 501)


def test_max_length_accepted():
    ResearchRequest(question="x" * 500)


def test_source_filter_wiki():
    req = ResearchRequest(question="Q", sources=[SourceFilter.wiki])
    assert req.sources == [SourceFilter.wiki]


def test_source_filter_all():
    req = ResearchRequest(
        question="Q",
        sources=[SourceFilter.wiki, SourceFilter.arxiv, SourceFilter.web],
    )
    assert len(req.sources) == 3


def test_no_cache_default_false():
    req = ResearchRequest(question="Q")
    assert req.no_cache is False


def test_no_cache_true():
    req = ResearchRequest(question="Q", no_cache=True)
    assert req.no_cache is True


# ---- ResearchResult --------------------------------------------------------


def test_result_default_empty_lists():
    result = ResearchResult(question="Q", answer="A")
    assert result.citations == []
    assert result.sources_used == []
    assert result.sources_failed == []
    assert result.cached is False
