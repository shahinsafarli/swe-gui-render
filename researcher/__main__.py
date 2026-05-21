"""Allows `python -m researcher` to invoke the CLI.

Also ensures the project root is on sys.path so both
`python -m researcher` and `python src/cli.py` work without PYTHONPATH.
"""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.cli import cli

if __name__ == "__main__":
    cli()
