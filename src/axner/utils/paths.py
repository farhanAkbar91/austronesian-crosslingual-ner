"""Repository path resolution."""

from __future__ import annotations

from pathlib import Path


def resolve_repo_root(start: Path | None = None) -> Path:
    """Walk upwards from ``start`` (default: CWD) until ``pyproject.toml`` is found.

    Falls back to the starting directory when no marker is found (e.g. when
    the package is installed standalone in Colab/Kaggle).
    """
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return current
