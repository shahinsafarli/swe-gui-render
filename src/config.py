"""
Application configuration — reads from .env via pydantic-settings.
All other modules import from here; never read os.environ directly.

After loading, all settings are mirrored into os.environ so the provided
ai/ module (which uses os.getenv() directly) picks them up without
any modification to the ai/ package.
"""
import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM provider
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-6"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    google_api_key: str = ""

    # Database (optional — used for persistent cache / history)
    database_url: str = "postgresql://localhost/researcher"

    # App settings
    log_level: str = "INFO"
    max_concurrent: int = Field(default=5, ge=1, le=50)
    cache_ttl_seconds: int = Field(default=3600, ge=0)

    # Topic 4 — web search
    web_search_provider: str = "duckduckgo"
    tavily_api_key: str = ""
    serper_api_key: str = ""

    # Per-source timeouts (seconds)
    wikipedia_timeout: float = Field(default=10.0, gt=0)
    arxiv_timeout: float = Field(default=15.0, gt=0)
    web_timeout: float = Field(default=10.0, gt=0)

    # Max question length
    max_question_length: int = Field(default=500, ge=1)

    # Max sources passed to synthesizer
    max_sources: int = Field(default=9, ge=1)

    @property
    def db_url_is_placeholder(self) -> bool:
        """True when DATABASE_URL still contains template placeholders."""
        return "YOUR_DB" in self.database_url or "YOUR_" in self.database_url


def _mirror_to_environ(s: Settings) -> None:
    """Export settings into os.environ for the ai/ module.

    The provided ai/ package reads provider keys via os.getenv() directly
    (not via pydantic-settings).  Mirroring our loaded settings into os.environ
    ensures the ai/ factory functions see the correct values without us needing
    to modify the ai/ package.

    Only non-empty values are written to avoid clobbering env vars that were
    set externally to a non-empty value that we accidentally loaded as ''.
    """
    mapping = {
        "LLM_PROVIDER": s.llm_provider,
        "LLM_MODEL": s.llm_model,
        "ANTHROPIC_API_KEY": s.anthropic_api_key,
        "OPENAI_API_KEY": s.openai_api_key,
        "GOOGLE_API_KEY": s.google_api_key,
        "WEB_SEARCH_PROVIDER": s.web_search_provider,
        "TAVILY_API_KEY": s.tavily_api_key,
        "SERPER_API_KEY": s.serper_api_key,
    }
    for key, value in mapping.items():
        if value:  # only write if non-empty
            os.environ[key] = value


# Module-level singleton — import this everywhere
settings = Settings()

# Mirror into os.environ immediately so ai/ providers see the right keys.
_mirror_to_environ(settings)
