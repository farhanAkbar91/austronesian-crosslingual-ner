"""Export one run directory as a transfer bundle (zip) for Colab/Kaggle <-> local moves.

Usage:
    python scripts/export_artifacts.py --run-dir outputs/exp1_singlesource_zeroshot/<run_id>
    python scripts/export_artifacts.py --run-dir ... --out bundles/run.zip --include-checkpoints
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from axner.artifacts.transfer import export_bundle  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", default=None, help="bundle zip path (default: alongside the run dir)")
    parser.add_argument("--include-checkpoints", action="store_true",
                        help="include model checkpoints (needed for EXP-3 init on another machine)")
    args = parser.parse_args()

    bundle = export_bundle(args.run_dir, out_path=args.out,
                           include_checkpoints=args.include_checkpoints)
    size_mb = bundle.stat().st_size / 1e6
    print(f"bundle written: {bundle} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
