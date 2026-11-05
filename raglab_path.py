"""Make src/ importable when running scripts from the repo root, with no install step."""
import sys
from pathlib import Path


def add_src() -> None:
    src = str(Path(__file__).resolve().parent / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
