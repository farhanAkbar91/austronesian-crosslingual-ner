"""CPU smoke test: validate the end-to-end pipeline on tiny data.

THIS IS A PIPELINE CHECK, NOT A RESEARCH RESULT. It uses a tiny randomly
initialized model and either synthetic data (fully offline, default) or a
tiny slice of real WikiANN data (downloads only that language's parquet).

Usage:
    python scripts/run_smoke_test.py                # offline synthetic smoke
    python scripts/run_smoke_test.py --real-data    # real WikiANN jv, tiny model
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from axner.config.loader import load_experiment_config  # noqa: E402
from axner.config.validation import validate_or_raise  # noqa: E402
from axner.experiments.runner import execute_run  # noqa: E402
from axner.utils.logging import get_logger  # noqa: E402

logger = get_logger("axner.scripts.smoke")

REAL_DATA_OVERRIDES = {
    "data": {
        "dataset": {"name": "wikiann"},
    },
    "experiment": {
        "targets": ["jv"],
        "train": {"language": "target", "size": 8},
        "validation": {"language": "target", "size": 8},
    },
    "evaluation": {
        "eval_target_validation": False,
        "bootstrap": {"n_resamples": 100},
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real-data", action="store_true",
                        help="use a tiny real WikiANN jv slice instead of synthetic data")
    parser.add_argument("--output-base", default=None)
    args = parser.parse_args()

    cfg = load_experiment_config("smoke_test", config_dir=REPO_ROOT / "configs" / "experiments")
    if args.real_data:
        from axner.config.loader import deep_merge

        cfg = deep_merge(cfg, REAL_DATA_OVERRIDES)
    cfg = validate_or_raise(cfg)

    mode = "real WikiANN jv (tiny)" if args.real_data else "synthetic (offline)"
    logger.info("smoke test mode: %s", mode)
    summary = execute_run(cfg, output_base=args.output_base, run_label="smoke")

    print("\n=== SMOKE TEST ", "PASS" if summary["status"] == "completed" else "FAIL", " ===", sep="")
    print(json.dumps(summary, indent=2))
    print("\nThis validated the PIPELINE only; the metrics are from a tiny random "
          "model and are not research results.")
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
