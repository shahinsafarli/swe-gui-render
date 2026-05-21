"""
Application-level data models for the Async Research Assistant.
Separate from ai/ schemas — the SE layer owns these.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class SourceFilter(str, Enum):
    """Which sources to query."""

    wiki = "wiki"
    arxiv = "arxiv"
    web = "web"


class ResearchRequest(BaseModel):
    """Validated user research request."""

    question: str = Field(..., min_length=1, max_length=500)
    sources: Optional[list[SourceFilter]] = None  # None = all three
    no_cache: bool = False

    @field_validator("question")
    @classmethod
    def _strip_question(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("question must not be blank")
        return stripped


class SourceRecord(BaseModel):
    """A source as stored/returned by the SE layer."""

    title: str
    url: str
    snippet: str
    origin: str  # wikipedia | arxiv | web


class CitationRecord(BaseModel):
    """A numbered citation in the final answer."""

    index: int
    title: str
    url: str
    origin: str


class ResearchResult(BaseModel):
    """Full result of one research request."""

    question: str
    answer: str
    citations: list[CitationRecord] = Field(default_factory=list)
    sources_used: list[SourceRecord] = Field(default_factory=list)
    sources_failed: list[str] = Field(default_factory=list)
    cached: bool = False
    elapsed_seconds: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ResearchSession(BaseModel):
    """A persisted research session (for history)."""

    id: int
    question: str
    answer: str
    citations_json: str
    sources_failed: list[str] = Field(default_factory=list)
    cached: bool = False
    elapsed_seconds: float = 0.0
    created_at: datetime


class ErrorResponse(BaseModel):
    """Structured error for CLI / API output."""

    error: str
    detail: Optional[str] = None
