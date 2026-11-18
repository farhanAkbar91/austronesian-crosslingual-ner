"""Software environment capture for reproducibility records."""

from __future__ import annotations

import platform
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

_TRACKED_PACKAGES = (
    "torch", "transformers", "datasets", "accelerate", "numpy", "pandas",
    "pyarrow", "pyyaml", "tqdm", "austronesian-crosslingual-ner",
)


def _package_versions() -> dict:
    versions = {}
    for package in _TRACKED_PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def _git_state(repo_root) -> dict:
    if repo_root is None:
        return {"available": False}
    root = Path(repo_root)
    if not (root / ".git").exists():
        return {"available": False}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        return {"available": True, "commit": commit, "dirty": bool(dirty)}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"available": False}


def capture_environment_info(repo_root=None) -> dict:
    """Capture versions, platform, CUDA availability, and git state.

    ``repo_root`` defaults to the repository root when it contains a .git
    directory; pass ``None`` explicitly with ``repo_root=False`` semantics by
    using :func:`capture_environment_info` from non-repo contexts.
    """
    if repo_root is None:
        from axner.utils.paths import resolve_repo_root

        repo_root = resolve_repo_root()
    info = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": _package_versions(),
        "git": _git_state(repo_root),
    }
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        info["cuda_available"] = None
    return info
