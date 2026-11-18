"""YAML configuration loading and merging.

Configuration is organised in base files (project/data/model/training/
evaluation) plus one file per experiment. An experiment file contains an
``experiment`` section and MAY contain overlay sections (``data``, ``model``,
``training``, ``evaluation``) that deep-merge over the base values. Experiment
values always win.

The merged dictionary is the single source of truth handed to the runner and
snapshot into each run directory.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from axner.utils.paths import resolve_repo_root

BASE_CONFIG_FILES = {
    "project": "project.yaml",
    "data": "data.yaml",
    "model": "model.yaml",
    "training": "training.yaml",
    "evaluation": "evaluation.yaml",
}

MERGEABLE_SECTIONS = ("project", "data", "model", "training", "evaluation")


def load_yaml(path: Path | str) -> dict:
    """Load a YAML file as a dict; fail loudly on missing or non-mapping files."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"config file {path} must contain a YAML mapping, got {type(data).__name__}")
    return data


def deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge ``overlay`` into a copy of ``base``.

    Dicts merge recursively; lists and scalars are replaced by the overlay's.
    """
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_base_configs(config_dir: Path | str | None = None) -> dict:
    """Load all base configuration files from the configs/ directory.

    Each file wraps its content in its section name (``project.yaml`` starts
    with ``project:``); the wrapper is unwrapped so callers address sections
    directly (``cfg["project"]["label_scheme"]``).
    """
    config_dir = Path(config_dir) if config_dir else resolve_repo_root() / "configs"
    sections: dict = {}
    for section, filename in BASE_CONFIG_FILES.items():
        data = load_yaml(config_dir / filename)
        if set(data.keys()) == {section}:
            data = data[section]
        sections[section] = data
    return sections


def _base_config_dir(config_dir, experiment_path: Path) -> Path:
    """Locate the directory holding project.yaml for base configuration loading."""
    candidates = []
    if config_dir:
        candidates.append(Path(config_dir))
    candidates.append(experiment_path.parent.parent)
    candidates.append(resolve_repo_root() / "configs")
    for candidate in candidates:
        if (candidate / "project.yaml").exists():
            return candidate
    return resolve_repo_root() / "configs"


def load_experiment_config(path_or_name: Path | str, config_dir: Path | str | None = None) -> dict:
    """Load an experiment configuration merged over the base configurations.

    ``path_or_name`` is either a path to an experiment YAML or a bare
    experiment id whose file ``<id>.yaml`` lives in ``configs/experiments/``
    (or in ``config_dir`` when given).
    """
    path = Path(path_or_name)
    if path.suffix:  # explicit file path
        experiment_path = path
        base_dir = _base_config_dir(config_dir, experiment_path)
    else:  # bare experiment id
        search_dirs = []
        if config_dir:
            search_dirs.append(Path(config_dir))
        search_dirs.append(resolve_repo_root() / "configs" / "experiments")
        experiment_path = next(
            (d / f"{path.name}.yaml" for d in search_dirs if (d / f"{path.name}.yaml").exists()),
            None,
        )
        if experiment_path is None:
            raise FileNotFoundError(
                f"experiment config {path_or_name!r} not found in {[str(d) for d in search_dirs]}"
            )
        base_dir = experiment_path.parent.parent

    merged = load_base_configs(base_dir)
    experiment_cfg = load_yaml(experiment_path)
    for section in MERGEABLE_SECTIONS:
        if section in experiment_cfg:
            merged[section] = deep_merge(merged[section], experiment_cfg[section])
    merged["experiment"] = copy.deepcopy(experiment_cfg.get("experiment", {}))
    merged["_meta"] = {
        "experiment_path": str(experiment_path),
        "config_dir": str(base_dir),
    }
    return merged
