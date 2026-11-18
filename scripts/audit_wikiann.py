"""Audit WikiANN language configurations: metadata, split sizes, label schema,
IOB2 validity, and duplicate detection. Downloads per-language parquet files
(a few MB per language) into the data cache.

Usage:
    python scripts/audit_wikiann.py --languages jv su [--splits train validation test]
    python scripts/audit_wikiann.py --languages id ms tl en jv su --out data/audits
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from axner.artifacts.manager import save_json  # noqa: E402
from axner.data.wikiann import audit_languages  # noqa: E402
from axner.utils.logging import get_logger  # noqa: E402

logger = get_logger("axner.scripts.audit")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", required=True,
                        help="WikiANN language codes (e.g. jv su id ms tl en)")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "audits"))
    parser.add_argument("--cache-dir", default=str(REPO_ROOT / "data" / "hf_cache"))
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()

    report = audit_languages(
        args.languages, splits=tuple(args.splits),
        cache_dir=args.cache_dir, revision=args.revision,
    )

    out_dir = Path(args.out)
    for lang, lang_report in report["languages"].items():
        save_json(out_dir / f"{lang}.json", lang_report)
    save_json(out_dir / "summary.json", {"summary": report["summary"]})

    print(f"\naudit written to {out_dir}")
    header = ["language"] + [f"{s}_rows" for s in args.splits] + ["iob2_invalid"]
    print("  " + "  ".join(f"{h:>12}" for h in header))
    for row in report["summary"]:
        print("  " + "  ".join(f"{row.get(h, '-')!s:>12}" for h in header))
    return 0


if __name__ == "__main__":
    sys.exit(main())
