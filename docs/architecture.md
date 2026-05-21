# Architecture — Async Research Assistant

## Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         User / CLI                          │
│             python -m researcher ask "question"             │
└─────────────────────┬───────────────────────────────────────┘
                      │ ResearchRequest (validated)
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  src/core/researcher.py                     │
│  - Cache lookup (src/services/cache.py)                     │
│  - Delegates to orchestrator for fresh fetches              │
│  - Calls ai_service.synthesize()                            │
│  - Assembles ResearchResult                                 │
└──────────┬───────────────────────────┬──────────────────────┘
           │                           │
           ▼                           ▼
┌──────────────────────┐   ┌──────────────────────────────────┐
│ src/services/        │   │ src/concurrency/orchestrator.py  │
│ cache.py             │   │                                  │
│  TTL in-memory cache │   │  asyncio.gather(wiki, arxiv, web)│
│  key=(source,query)  │   │  Semaphore(max_concurrent)       │
└──────────────────────┘   │  return_exceptions=True          │
                           │  per-source asyncio.timeout()    │
                           └──────────────┬───────────────────┘
                                          │
                          ┌───────────────┼──────────────┐
                          ▼               ▼              ▼
                   ┌────────────┐ ┌──────────────┐ ┌──────────┐
                   │ Wikipedia  │ │   arXiv      │ │   Web    │
                   │ REST API   │ │   Atom API   │ │  Search  │
                   └─────┬──────┘ └──────┬───────┘ └────┬─────┘
                         │               │               │
                         └───────────────┴───────────────┘
                                         │
                              ┌──────────▼──────────┐
                              │ src/services/        │
                              │ ai_service.py        │
                              │  Retries (tenacity)  │
                              │  Logging             │
                              │  Timeout handling    │
                              └──────────┬───────────┘
                                         │
                              ┌──────────▼──────────┐
                              │     ai/ module       │
                              │  (DO NOT MODIFY)     │
                              │  fetch_wikipedia()   │
                              │  fetch_arxiv()       │
                              │  fetch_web()         │
                              │  synthesize()        │
                              └─────────────────────┘
```

## Key design decisions

- **Why asyncio?** All three source fetches are I/O-bound (HTTP requests). asyncio.gather lets them run simultaneously; wall-clock time ≈ max(wiki, arxiv, web) instead of sum.
- **Why a Semaphore?** Prevents unbounded concurrency — polite to external APIs and prevents 429 errors.
- **Why per-source timeouts?** A slow Wikipedia response shouldn't block arXiv. asyncio.timeout() per coroutine isolates failures.
- **Why psycopg3 (async)?** Fits naturally in the asyncio event loop; no thread pool needed for DB.
- **Why pydantic-settings for config?** Type safety, validation, and automatic .env reading in one line. No stray os.environ calls.
