"""Run directories, manifests, and serialization helpers.

Every training/evaluation run gets a unique, traceable directory::

    outputs/<experiment_id>/<timestamp>-<label>-<uuid6>/
        config/      resolved config snapshot + environment info
        data/        sampling indices, audits, statistics, duplicates
        checkpoints/ model checkpoints (never committed to git)
        logs/        trainer log history
        predictions/ sentence/entity level prediction exports
        metrics/     validation/test metrics, bootstrap, aggregates

``manifest.json`` ties the directory together: experiment id, seeds, dataset
version, model id, sampling digests, and environment info.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path

import yaml

RUN_SUBDIRS = ("config", "data", "checkpoints", "logs", "predictions", "metrics")
_SAFE_LABEL = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize(label: str) -> str:
    return _SAFE_LABEL.sub("-", label).strip("-") or "run"


def create_run_directory(base_dir, experiment_id: str, label: str | None = None) -> Path:
    """Create a unique run directory with the standard subdirectory layout."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{timestamp}-{_sanitize(label or 'run')}-{uuid.uuid4().hex[:6]}"
    run_dir = Path(base_dir) / experiment_id / run_name
    if run_dir.exists():  # uuid collision safeguard
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    for sub in RUN_SUBDIRS:
        (run_dir / sub).mkdir()
    return run_dir


def run_id(run_dir) -> str:
    return Path(run_dir).name


def save_json(path, obj, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, ensure_ascii=False, default=str)
    return path


def load_json(path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_jsonl(records, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return path


def load_jsonl(path) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_yaml(obj, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(obj, fh, sort_keys=False, allow_unicode=True)
    return path


def load_yaml(path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def artifact_paths(run_dir) -> dict:
    """Standard artifact file locations for a run directory."""
    run_dir = Path(run_dir)
    return {
        "manifest": run_dir / "manifest.json",
        "config_resolved": run_dir / "config" / "config_resolved.yaml",
        "environment": run_dir / "config" / "environment.json",
        "sampling": run_dir / "data" / "sampling_indices.json",
        "data_audit": run_dir / "data" / "data_audit.json",
        "duplicates": run_dir / "data" / "duplicates.json",
        "dataset_stats": run_dir / "data" / "dataset_stats.json",
        "trainer_history": run_dir / "logs" / "trainer_history.json",
        "validation_metrics": run_dir / "metrics" / "validation_metrics.json",
        "validation_target_metrics": run_dir / "metrics" / "validation_target_{lang}.json",
        "test_metrics": run_dir / "metrics" / "test_metrics_{lang}.json",
        "bootstrap": run_dir / "metrics" / "bootstrap_{lang}.json",
        "sentence_level": run_dir / "predictions" / "test_{lang}_sentence_level.jsonl",
        "entity_level": run_dir / "predictions" / "test_{lang}_entity_level.jsonl",
    }


def write_manifest(run_dir, **fields) -> dict:
    """Create ``manifest.json``; existing content is overwritten."""
    manifest = {
        "schema_version": 1,
        "experiment_id": fields.get("experiment_id"),
        "run_id": run_id(run_dir),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **fields,
    }
    save_json(Path(run_dir) / "manifest.json", manifest)
    return manifest


def update_manifest(run_dir, **fields) -> dict:
    """Merge fields into the existing manifest (new keys win)."""
    path = Path(run_dir) / "manifest.json"
    manifest = load_json(path) if path.exists() else {}
    manifest.update(fields)
    save_json(path, manifest)
    return manifest


def load_manifest(run_dir) -> dict:
    return load_json(Path(run_dir) / "manifest.json")
