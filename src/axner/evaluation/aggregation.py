"""Aggregation of metrics across training seeds and run directories.

Seed-level aggregation quantifies VARIATION ACROSS TRAINING RUNS. It must
never be reported as test-set sampling uncertainty (that is the bootstrap's
job) and repeated seeds are not independent test datasets.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path

from axner.artifacts.manager import load_json

SEED_AGGREGATION_NOTE = (
    "statistics across training seeds (run-level variation); NOT test-set "
    "sampling uncertainty - use sentence-level bootstrap for that"
)


def aggregate_seeds(metric_dicts: list[dict], keys=None) -> dict:
    """Aggregate numeric metric fields over a list of per-seed metric dicts.

    ``keys`` restricts aggregation to the given metric names; default: every
    numeric (int/float, non-bool) field present in the first dict.
    """
    if not metric_dicts:
        raise ValueError("metric_dicts must be non-empty")
    if keys is None:
        first = metric_dicts[0]
        keys = [
            k for k, v in first.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        ]
    out: dict = {"n_seeds": len(metric_dicts), "note": SEED_AGGREGATION_NOTE, "metrics": {}}
    for key in keys:
        values = [d[key] for d in metric_dicts if key in d and isinstance(d[key], (int, float))]
        if not values:
            continue
        out["metrics"][key] = {
            "mean": round(statistics.fmean(values), 6),
            "std": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "n_values": len(values),
        }
    return out


def collect_run_metrics(run_dir) -> dict:
    """Load per-target test metrics from one run directory.

    Returns ``{target_language: metrics_dict}`` based on
    ``metrics/test_metrics_<lang>.json`` files.
    """
    metrics_dir = Path(run_dir) / "metrics"
    per_target = {}
    if metrics_dir.exists():
        for path in sorted(metrics_dir.glob("test_metrics_*.json")):
            lang = path.stem.removeprefix("test_metrics_")
            per_target[lang] = load_json(path)
    return per_target


def aggregate_run_directories(run_dirs) -> dict:
    """Aggregate runs (typically one per training seed) per (experiment, variant, target).

    Variant identity comes from the run manifest's ``variant_tags``
    (e.g. ``{"source": "id"}`` for EXP-1, ``{"composition": {...}}`` for
    EXP-2, ``{"arm": ..., "fewshot_size": 50}`` for EXP-3).
    """
    groups: dict = defaultdict(list)
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        try:
            manifest = load_json(run_dir / "manifest.json")
        except FileNotFoundError:
            continue
        variant_tags = manifest.get("variant_tags", {})
        variant_key = tuple(sorted(variant_tags.items()))
        for lang, metrics in collect_run_metrics(run_dir).items():
            groups[(manifest.get("experiment_id"), variant_key, lang)].append(metrics)

    out = {}
    for (experiment_id, variant_key, lang), metric_list in sorted(groups.items()):
        variant_tags = dict(variant_key)
        agg = aggregate_seeds(metric_list)
        out[f"{experiment_id}|{lang}|{variant_key}"] = {
            "experiment_id": experiment_id,
            "variant_tags": variant_tags,
            "target_language": lang,
            "aggregation": agg,
            "run_count": len(metric_list),
        }
    return out


def best_run_by(run_dirs, metrics_file: str, metric_key: str) -> tuple[Path, float] | None:
    """Find the run with the highest value of ``metric_key`` in ``metrics_file``.

    Used for EXP-3 checkpoint selection, which must rely on target-validation
    metrics only (never target test metrics).
    """
    best = None
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        path = run_dir / "metrics" / metrics_file
        try:
            metrics = load_json(path)
        except FileNotFoundError:
            continue
        value = metrics.get(metric_key)
        if value is None:
            continue
        if best is None or value > best[1]:
            best = (run_dir, float(value))
    return best
