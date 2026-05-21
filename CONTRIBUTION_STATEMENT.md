# Contribution Statement — AIENG Final Project
## Topic 4: Async Research Assistant | Spring 2026

> All three team members confirm the distribution below is accurate.
> Each member contributed at least 20% of the total work.

---

## Team Members & Contributions

### Member 1 — [Name]
**Commits**: ~33%
**Owns**:
- `src/config.py` — pydantic-settings configuration
- `src/models.py` — Pydantic models (ResearchRequest, ResearchResult, …)
- `src/services/ai_service.py` — retry + logging wrapper
- `tests/test_service.py`, `tests/test_models.py`
- Report sections: Architecture, AI Module Integration

### Member 2 — [Name]
**Commits**: ~33%
**Owns**:
- `src/concurrency/orchestrator.py` — asyncio.gather + Semaphore + graceful degradation
- `src/services/cache.py` — TTL cache
- `src/core/researcher.py` — core pipeline
- `tests/test_concurrency.py`, `tests/test_cache.py`, `tests/test_researcher.py`
- `scripts/bench.py` — benchmark script
- Report sections: Concurrency, Robustness

### Member 3 — [Name]
**Commits**: ~33%
**Owns**:
- `src/cli.py` — Click CLI
- `src/storage/repository.py` — PostgreSQL repository
- `scripts/demo.py` — end-to-end demo
- `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml`
- `tests/test_cli.py`
- Report sections: Docker, Testing, Limitations

---

## Signatures

Member 1: _________________________ Date: ___________

Member 2: _________________________ Date: ___________

Member 3: _________________________ Date: ___________

---

*We confirm that no member modified the `ai/` folder, no API keys are
hard-coded in any commit, and all tests pass offline.*
