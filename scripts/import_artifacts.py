"""Import a run bundle (zip) exported from Colab/Kaggle (or another machine) into
the local outputs tree, verifying its manifest.

Usage:
    python scripts/import_artifacts.py --bundle bundles/<run_id>.bundle.zip
    python scripts/import_artifacts.py --bundle ... --include-checkpoints-first ...
    (checkpoints are included at EXPORT time with --include-checkpoints)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from axner.artifacts.transfer import import_bundle  # noqa: E402
from axner.utils.paths import resolve_repo_root  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", required=True, help="path to a run bundle zip")
    parser.add_argument("--dest", default=str(resolve_repo_root() / "outputs"),
                        help="outputs root (default: <repo>/outputs)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_dir = import_bundle(args.bundle, args.dest, overwrite=args.overwrite)
    print(f"imported run: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
