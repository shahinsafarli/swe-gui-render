# Async Research Assistant — AIENG Final Project (Topic 4)

A software-engineering layer around an async AI research pipeline.
The system queries **Wikipedia**, **arXiv**, and a **web search API** in
parallel, then synthesizes a single cited answer using an LLM.

**Course:** AI-ENG-110 Software Engineering · **Team:** _(fill in your team name)_  
**Due:** May 23, 2026 at 23:59 (UTC+4) · **Tag:** `v1.0-final`

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Technologies Used](#2-technologies-used)
3. [Architecture Overview](#3-architecture-overview)
4. [Setup & Installation](#4-setup--installation)
5. [Environment Configuration](#5-environment-configuration)
6. [Running the Project](#6-running-the-project)
7. [Database Initialization](#7-database-initialization)
8. [CLI Usage & Examples](#8-cli-usage--examples)
9. [Running Tests](#9-running-tests)
10. [Benchmark: Sequential vs Concurrent](#10-benchmark-sequential-vs-concurrent)
11. [Docker](#11-docker)
12. [Project Structure](#12-project-structure)
13. [Known Limitations](#13-known-limitations)

---

## 1. Project Overview

The user submits a research question. The system concurrently fetches sources
from three origins (Wikipedia, arXiv, web search), caches the results by a
canonicalized query key, deduplicates overlapping URLs, and calls an LLM to
produce a cited Markdown answer. Every AI call is wrapped with exponential
backoff retries, per-source timeouts, and graceful degradation — if one source
fails, the answer is still produced from the remaining ones.

---

## 2. Technologies Used

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| AI / LLM | Anthropic Claude / OpenAI GPT-4o-mini / Google Gemini (pluggable) |
| Web search | DuckDuckGo (no key) · Tavily · Serper |
| Async HTTP | httpx |
| Concurrency | asyncio, asyncio.gather, asyncio.Semaphore, asyncio.to_thread |
| Retries | tenacity (exponential backoff) |
| Validation | Pydantic v2 + pydantic-settings |
| CLI | Click |
| Database | PostgreSQL via psycopg3 (async, optional) |
| Testing | pytest + pytest-asyncio + pytest-cov |
| Containerization | Docker (multi-stage) + Docker Compose |
| CI | GitHub Actions |

---

## 3. Architecture Overview

```
User / CLI
    │  ResearchRequest  (validated Pydantic model)
    ▼
src/core/researcher.py          ← business logic
    ├── cache lookup (src/services/cache.py)
    └── src/concurrency/orchestrator.py
            ├── asyncio.gather(wiki, arxiv, web)
            ├── asyncio.Semaphore(MAX_CONCURRENT)   ← bounds parallelism
            └── per-source asyncio.timeout()
                    │
            src/services/ai_service.py  ← retries + logging + timeout
                    │
            ai/ module (provided, DO NOT MODIFY)
                    ├── fetch_wikipedia()
                    ├── fetch_arxiv()
                    ├── fetch_web()
                    └── synthesize()   ← offloaded via asyncio.to_thread
                    │
    ├── src/storage/repository.py      ← PostgreSQL session persistence
    └── ResearchResult  (Pydantic model returned to caller)
```

Key design decisions:

- **Every** `ai.*` call goes through `AIService`. No direct `ai.*` imports in
  business logic.
- The cache stores raw `Source` objects keyed by `(source_name, canonicalized_query)`.
  A `--no-cache` flag bypasses both read and write.
- `ai.synthesize` is synchronous (LLM call). It is offloaded with
  `asyncio.to_thread` so the event loop is never blocked.
- All source-fetch failures are caught per-source; partial results are still
  synthesized. Failed source names are surfaced in `ResearchResult.sources_failed`.

---

## 4. Setup & Installation

### Prerequisites

- Python 3.12+
- PostgreSQL 14+ _(optional — only needed for `researcher history`)_
- Docker + Docker Compose _(optional — for containerized runs)_

### Install

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_TEAM/async-research-assistant
# ↑ Replace YOUR_TEAM with your actual GitHub username/org

cd async-research-assistant

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install pinned dependencies
pip install -r requirements.txt
```

---

## 5. Environment Configuration

```bash
# Copy the template and fill in your values
cp .env.example .env
```

Open `.env` and replace the placeholder values. **Required** before any live run:

| Variable | Required? | Description |
|---|---|---|
| `LLM_PROVIDER` | Yes | `anthropic` \| `openai` \| `gemini` |
| `LLM_MODEL` | Yes | e.g. `claude-sonnet-4-6`, `gpt-4o-mini`, `gemini-2.0-flash` |
| `ANTHROPIC_API_KEY` | If anthropic | Your Anthropic API key (`sk-ant-...`) |
| `OPENAI_API_KEY` | If openai | Your OpenAI API key (`sk-...`) |
| `GOOGLE_API_KEY` | If gemini | Your Google AI key (`AIza...`) |
| `WEB_SEARCH_PROVIDER` | No | `duckduckgo` (default, no key needed) \| `tavily` \| `serper` |
| `TAVILY_API_KEY` | If tavily | Tavily API key |
| `DATABASE_URL` | No | `postgresql://USER:PASS@HOST:5432/DBNAME` — omit to skip history |
| `MAX_CONCURRENT` | No | Semaphore bound (default `5`) |
| `CACHE_TTL_SECONDS` | No | Cache TTL in seconds; `0` = never expire (default `3600`) |

> **Tip**: DuckDuckGo works out of the box with no API key. Use it for the
> offline demo and testing.

---

## 6. Running the Project

### Offline demo (no API keys, no network)

```bash
# Provided demo harness — uses fake providers
python demo_ai.py --offline

# Your SE-layer demo — runs all 5 research questions
python scripts/demo.py --offline
python scripts/demo.py --offline --limit 5   # same, explicit
```

### CLI — live mode (requires `.env` with valid keys)

```bash
# Ask a research question (all three sources)
python -m researcher ask "What is photosynthesis?"

# Restrict to specific sources
python -m researcher ask "quantum computing basics" --sources wiki,arxiv

# Bypass the cache
python -m researcher ask "latest LLM research" --no-cache

# Machine-readable JSON output (for piping / scripts)
python -m researcher ask "CRISPR gene editing" --json-output

# View session history (requires DATABASE_URL to be set)
python -m researcher history --limit 10

# Adjust log verbosity
python -m researcher --log-level DEBUG ask "AI safety"
```

---

## 7. Database Initialization

The database schema is created automatically on first use (`CREATE TABLE IF NOT EXISTS`).  
If you want to create it explicitly:

```bash
# From Python
python -c "
import asyncio
from src.storage.repository import init_db
asyncio.run(init_db())
print('Database initialized.')
"
```

**Table created:** `research_sessions`

```sql
CREATE TABLE IF NOT EXISTS research_sessions (
    id              SERIAL PRIMARY KEY,
    question        TEXT NOT NULL,
    answer          TEXT NOT NULL,
    citations_json  JSONB NOT NULL DEFAULT '[]',
    sources_failed  JSONB NOT NULL DEFAULT '[]',
    cached          BOOLEAN NOT NULL DEFAULT FALSE,
    elapsed_seconds FLOAT NOT NULL DEFAULT 0.0,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_research_sessions_created_at
    ON research_sessions (created_at DESC);
```

> **Note:** The `DATABASE_URL` variable must be set in `.env` for the
> repository to connect. If it is absent or wrong, all persistence calls
> are logged at WARNING level and the application continues without history.

---

## 8. CLI Usage & Examples

### `ask` command

```bash
python -m researcher ask QUESTION [OPTIONS]

Arguments:
  QUESTION   The research question (wrap in quotes if it contains spaces)

Options:
  --sources TEXT     Comma-separated subset: wiki,arxiv,web (default: all three)
  --no-cache         Bypass the TTL cache; always fetch fresh results
  --json-output      Print result as JSON instead of formatted text
  --log-level LEVEL  DEBUG | INFO | WARNING | ERROR  (default: INFO)
```

**Example output:**

```
Q: What is photosynthesis?

A: Photosynthesis is the process by which plants convert light energy into
chemical energy stored as glucose [1]. The two main stages are the
light-dependent reactions [1] and the Calvin cycle [2].

References:
  [1] (wikipedia) Photosynthesis
      https://en.wikipedia.org/wiki/Photosynthesis
  [2] (arxiv) Light-Dependent Reactions of Photosynthesis
      https://arxiv.org/abs/...

  [took 1.42s]
```

### `history` command

```bash
python -m researcher history [--limit N]

Options:
  --limit INTEGER   Number of recent sessions to display (default: 10)
```

---

## 9. Running Tests

```bash
# Full test suite with coverage
pytest --cov=src --cov-report=term-missing

# Smoke tests only (provided — MUST always pass)
pytest tests/test_ai_smoke.py -v

# Single test file
pytest tests/test_concurrency.py -v

# All tests — quiet output
pytest -q
```

**Coverage target:** ≥ 60% (actual: 81% as of v1.0-final).  
All tests are **fully offline** — AI module and HTTP layer are mocked.

---

## 10. Benchmark: Sequential vs Concurrent

### Offline simulation (no API keys required)

```bash
python scripts/bench.py --n 5 --offline
```

| Mode | Time (s) | Speedup |
|---|---|---|
| Sequential | 2.655 | 1.0× |
| Concurrent | 1.003 | 2.6× |

N=5 questions × 3 sources · simulated latency (wiki 0.15 s, arXiv 0.20 s, web 0.18 s)

### Real-network benchmark (spec-required)

To reproduce real-network numbers on your machine with caches cleared:

```bash
# 1. Fill in API keys
cp .env.example .env && $EDITOR .env   # set ANTHROPIC_API_KEY + TAVILY_API_KEY

# 2. Clear the in-process cache (automatic on each fresh run) and run
python scripts/bench.py --n 5
```

> **Note:** Real-network results are environment-dependent (ISP latency,
> API response time, rate-limiting). Typical observed speedup with live
> providers is **2–4×** (per-source latency ≈ 0.5–1 s). Results are not
> included in this repository because API keys are not available in the CI
> environment. Run the command above on your own machine to reproduce.

**Why the speedup?** Source fetches are I/O-bound. `asyncio.gather(wiki, arxiv, web)`
runs all three simultaneously so wall-clock time ≈ max(wiki, arXiv, web)
instead of their sum. A `Semaphore(MAX_CONCURRENT)` prevents unbounded
concurrency and rate-protects external APIs.

---

## 11. Docker

### Build & run (offline demo)

```bash
docker build --platform linux/amd64 -t researcher .
docker run researcher                          # offline demo (default CMD)
```

### Run with real providers

```bash
# Make sure .env is filled in
docker run --env-file .env researcher python -m researcher ask "What is AI?"
```

### Docker Compose (with PostgreSQL)

```bash
# Copy and fill in .env first (especially POSTGRES_USER, POSTGRES_PASSWORD)
cp .env.example .env

docker compose up --build                      # starts db + app
docker compose run app python scripts/demo.py --offline
docker compose down -v                         # stop + remove volumes
```

> **Replace** `POSTGRES_USER` and `POSTGRES_PASSWORD` in `.env` with
> strong credentials for any deployment beyond your local machine.

---

## 12. Project Structure

```
async-research-assistant/
├── ai/                         ← PROVIDED by course (DO NOT MODIFY)
│   ├── __init__.py             ← public API: fetch_*, synthesize, schemas
│   ├── providers/              ← LLM provider adapters (Anthropic / OpenAI / Gemini)
│   ├── schemas.py              ← Source, Citation, AnswerWithCitations
│   ├── sources.py              ← fetch_wikipedia, fetch_arxiv, fetch_web
│   └── synthesizer.py         ← synthesize()
│
├── src/
│   ├── __init__.py             ← logging setup
│   ├── config.py               ← pydantic-settings (reads .env)
│   ├── models.py               ← SE-layer Pydantic models
│   ├── services/
│   │   ├── ai_service.py       ← retries + timeouts + logging wrapper
│   │   └── cache.py            ← TTL in-memory cache (source, query) → sources
│   ├── core/
│   │   └── researcher.py       ← run_research() — full pipeline
│   ├── concurrency/
│   │   └── orchestrator.py     ← asyncio.gather + Semaphore + graceful degradation
│   ├── storage/
│   │   └── repository.py       ← PostgreSQL session persistence (psycopg3)
│   └── cli.py                  ← Click CLI (ask, history)
│
├── tests/
│   ├── conftest.py             ← FakeLLM, FakeWebSearch, no_real_ai autouse
│   ├── test_ai_smoke.py        ← PROVIDED — must not be deleted or weakened
│   ├── test_service.py         ← AIService tests (offline)
│   ├── test_cache.py           ← cache TTL, canonicalization, expiry
│   ├── test_concurrency.py     ← pipeline, semaphore, graceful degradation
│   ├── test_researcher.py      ← end-to-end pipeline (offline)
│   ├── test_cli.py             ← Click CliRunner tests
│   ├── test_models.py          ← Pydantic validation edge cases
│   └── test_storage.py         ← repository save/retrieve/error paths
│
├── scripts/
│   ├── demo.py                 ← end-to-end scripted demo (all 5 questions)
│   └── bench.py                ← sequential vs concurrent benchmark
│
├── data/
│   └── research_questions.json ← 5 sample questions for demo & bench
│
├── artefacts/                  ← sample run outputs (required for submission)
├── docs/
│   └── architecture.md         ← architecture diagram
├── report/
│   ├── report.tex              ← team report (fill in names, compile to PDF)
│   └── slides.tex              ← presentation slides
│
├── .github/
│   └── workflows/ci.yml        ← GitHub Actions CI (+2 bonus)
│
├── Dockerfile                  ← multi-stage build (+1 bonus)
├── docker-compose.yml          ← app + PostgreSQL
├── requirements.txt            ← ALL deps pinned to ==version
├── .env.example                ← copy to .env, fill in keys
├── .gitignore
├── pytest.ini
├── demo_ai.py                  ← PROVIDED offline demo harness
├── CONTRIBUTION_STATEMENT.md   ← fill in, sign, submit
└── README.md
```

---

## 13. Known Limitations

- **In-process cache only (ephemeral by design).** `src/services/cache.py`
  uses a plain Python dict as its backing store.  The cache is **wiped on
  every process restart** — running `python -m researcher ask` as a one-shot
  CLI command always starts cold.  It is also **not shared** between processes;
  running multiple workers behind a load balancer gives each worker its own
  private island.  If persistence or cross-process sharing matters, replace
  the `_store` dict with a Redis or filesystem JSON backend — the `get`/`set`/
  `clear` API is intentionally minimal to make that swap easy.
- **PostgreSQL is optional, not embedded.** History is silently skipped if
  `DATABASE_URL` is not set or the database is unreachable.
- **Connection pool is bounded to 5 connections.** `psycopg_pool.AsyncConnectionPool`
  caps simultaneous DB connections at `max_size=5` (configurable).  Under very
  heavy concurrent load, excess requests will queue inside the pool.
- **Token-bucket rate limiter is opt-in.** `src/services/rate_limiter.py`
  provides a `TokenBucketRateLimiter` with per-source defaults
  (Wikipedia: 10 req/s, arXiv: 2 req/s, web: 1 req/s). The limiter is not
  wired into `AIService` by default — callers that need strict rate control
  should wrap fetches with the module-level `wikipedia_limiter`,
  `arxiv_limiter`, or `web_limiter` instances (or construct their own). Very
  long questions may still consume more LLM tokens than expected; this limiter
  controls *call frequency*, not token budget.
- **DuckDuckGo is rate-limited.** Under burst traffic it can return empty
  results. Use Tavily or Serper for reliable production web search.
- **SQLite is not supported.** The repository uses psycopg3 SQL syntax
  (`%s` placeholders, `JSONB`, `SERIAL`, `TIMESTAMPTZ`) which is
  PostgreSQL-specific. Use the provided Docker Compose setup for a
  zero-config local PostgreSQL.

See `report/report.pdf` §7 for a full discussion of limitations and future work.
