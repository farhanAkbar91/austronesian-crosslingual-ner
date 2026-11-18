"""Aggregate metrics across runs (training seeds) for one or all experiments.

Seed aggregation reports run-level variation; it is NOT test-set sampling
uncertainty (use the bootstrap files for that).

Usage:
    python scripts/aggregate_results.py --experiment exp1_singlesource_zeroshot
    python scripts/aggregate_results.py            # aggregate every experiment
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from axner.artifacts.manager import save_json  # noqa: E402
from axner.evaluation.aggregation import aggregate_run_directories  # noqa: E402
from axner.utils.paths import resolve_repo_root  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default=None, help="experiment id (default: all)")
    parser.add_argument("--outputs-dir", default=str(resolve_repo_root() / "outputs"))
    parser.add_argument("--out", default=None,
                        help="output JSON path (default: outputs/aggregations/aggregate.json)")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    if args.experiment:
        experiment_dirs = [outputs_dir / args.experiment]
        if not experiment_dirs[0].exists():
            print(f"no outputs for experiment {args.experiment!r} under {outputs_dir}")
            return 1
    else:
        experiment_dirs = [p for p in sorted(outputs_dir.iterdir()) if p.is_dir()] if outputs_dir.exists() else []

    run_dirs = [
        run_dir
        for exp_dir in experiment_dirs
        for run_dir in (sorted(p for p in exp_dir.iterdir() if p.is_dir()) if exp_dir.exists() else [])
    ]
    if not run_dirs:
        print("no run directories found")
        return 1

    aggregated = aggregate_run_directories(run_dirs)
    out_path = Path(args.out) if args.out else outputs_dir / "aggregations" / "aggregate.json"
    save_json(out_path, aggregated)

    print(f"aggregated {len(run_dirs)} run(s) -> {out_path}\n")
    for key, entry in aggregated.items():
        f1 = entry["aggregation"]["metrics"].get("f1", {})
        variant = ", ".join(f"{k}={v}" for k, v in entry["variant_tags"].items()) or "-"
        print(f"  {entry['target_language']:>4}  f1={f1.get('mean', float('nan')):.4f} "
              f"(±{f1.get('std', 0):.4f}, n_seeds={f1.get('n_values', 0)})  {key.split('|')[0]}  [{variant}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
