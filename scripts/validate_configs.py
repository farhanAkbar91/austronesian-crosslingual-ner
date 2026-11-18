"""Validate all project and experiment configurations; exit non-zero on failure.

Base configuration files are validated together under a minimal stub
experiment (they have no experiment section of their own); each experiment
configuration is validated as loaded and merged over the base files.

Usage:
    python scripts/validate_configs.py [--config-dir configs]
"""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from axner.config.loader import BASE_CONFIG_FILES, load_base_configs, load_experiment_config  # noqa: E402
from axner.config.validation import validate_config  # noqa: E402
from axner.utils.paths import resolve_repo_root  # noqa: E402


def _stub_experiment_cfg(base: dict) -> dict:
    cfg = deepcopy(base)
    cfg["experiment"] = {
        "id": "config_validation_stub",
        "description": "stub experiment used only to validate base configuration files",
        "training_seeds": [42],
        "sampling_seed": 42,
        "targets": ["jv"],
        "train": {"language": "target", "size": 100},
        "validation": {"language": "target", "size": None},
    }
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default=str(resolve_repo_root() / "configs"))
    args = parser.parse_args()
    config_dir = Path(args.config_dir)

    failures = 0

    print("== base configuration files (validated under a stub experiment) ==")
    for filename in BASE_CONFIG_FILES.values():
        path = config_dir / filename
        if not path.exists():
            print(f"  FAIL  {filename}: file not found")
            failures += 1
            continue
        try:
            problems = validate_config(_stub_experiment_cfg(load_base_configs(config_dir)))
            if problems:
                failures += 1
                print(f"  FAIL  {filename}")
                for problem in problems:
                    print(f"        - {problem}")
            else:
                print(f"  PASS  {filename}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {filename}: {exc}")

    print("== experiment configurations ==")
    experiments_dir = config_dir / "experiments"
    experiment_files = sorted(experiments_dir.glob("*.yaml")) if experiments_dir.exists() else []
    if not experiment_files:
        print("  (no experiment configs found)")
    for path in experiment_files:
        try:
            cfg = load_experiment_config(path, config_dir=config_dir)
            problems = validate_config(cfg)
            if problems:
                failures += 1
                print(f"  FAIL  {path.name}")
                for problem in problems:
                    print(f"        - {problem}")
            else:
                print(f"  PASS  {path.name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {path.name}: {exc}")

    print()
    if failures:
        print(f"validation FAILED for {failures} file(s)")
        return 1
    print("all configurations valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
