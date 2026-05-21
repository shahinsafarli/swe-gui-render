"""
Async Research Assistant — SE layer package.

Ensures the project root is on sys.path so `from src.*` imports work
when the package is run via `python src/cli.py` (not just `python -m researcher`).
"""
from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
