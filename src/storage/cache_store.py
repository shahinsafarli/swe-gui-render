"""
Spec-named alias for the storage repository module.

The spec requires this module to be named ``cache_store.py``.  The
implementation lives in ``repository.py`` (named after its role as a
database-backed *repository*, not an in-memory cache).  This shim
re-exports every public symbol so that code using either name works
identically.

OOP encapsulation note
----------------------
The connection pool is now owned by :class:`ResearchRepository` as a
private instance attribute (``self._pool``).  There is no module-level
``_pool`` global in ``repository.py``; all mutable pool state lives inside
the class, preserving proper OOP encapsulation.

Usage (spec-compliant import)::

    from src.storage.cache_store import repository, init_db, ResearchRepository

Usage (original name, also fine)::

    from src.storage.repository import repository, init_db, ResearchRepository
"""

from src.storage.repository import (  # noqa: F401  re-export all public names
    ResearchRepository,
    init_db,
    repository,
)

__all__ = ["ResearchRepository", "init_db", "repository"]
