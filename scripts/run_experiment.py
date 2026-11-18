"""Run one experiment run (one seed / one training configuration) end-to-end.

The full training campaign runs on Colab/Kaggle (see docs/experiment_protocol.md);
this script is the single entry point on both local (CPU smoke/dry runs) and
cloud (GPU training) environments.

Usage:
    python scripts/run_experiment.py --config configs/experiments/exp0_inlanguage_baseline.yaml --target jv --seed 42
    python scripts/run_experiment.py --config configs/experiments/exp1_singlesource_zeroshot.yaml --source id --seed 42
    python scripts/run_experiment.py --config configs/experiments/exp2a_multisource_id_ms.yaml --seed 42
    python scripts/run_experiment.py --config configs/experiments/exp3_fewshot_adaptation.yaml --arm fewshot --fewshot-size 50 --seed 42
    python scripts/run_experiment.py --config ... --dry-run   # data pipeline only, no model

Optional overrides:
    --output-base PATH   override outputs root (default: project.paths.outputs_dir)
    --run-label STR      label included in the run directory name
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from axner.experiments.runner import execute_run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="experiment config path or id")
    parser.add_argument("--seed", type=int, default=None, help="training seed for this run")
    parser.add_argument("--target", default=None, help="target language (target-training experiments)")
    parser.add_argument("--source", default=None, help="source language (matrix experiments)")
    parser.add_argument("--arm", default=None, choices=["fewshot", "target_only"],
                        help="EXP-3 arm")
    parser.add_argument("--fewshot-size", type=int, default=None, help="EXP-3 few-shot size")
    parser.add_argument("--output-base", default=None, help="outputs root override")
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="prepare and audit data, save artifacts, skip model/training")
    args = parser.parse_args()

    summary = execute_run(
        args.config,
        seed=args.seed,
        target=args.target,
        source=args.source,
        arm=args.arm,
        fewshot_size=args.fewshot_size,
        output_base=args.output_base,
        run_label=args.run_label,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
