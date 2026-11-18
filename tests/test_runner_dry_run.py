"""Runner integration (dry-run path): config -> run directory -> data artifacts.

The dry-run path needs no torch: it exercises configuration resolution,
run-directory creation, data preparation, audits, and artifact writing.
"""

from pathlib import Path

from axner.artifacts.manager import load_json, load_manifest, load_yaml
from axner.experiments.runner import execute_run, resolve_run_point, variant_tags


def test_resolve_run_point_exp0_target(merged_config):
    point = resolve_run_point(merged_config, seed=43, target="su")
    assert point["seed"] == 43
    assert point["train_langs"] == ["su"]
    assert point["eval_langs"] == ["su"]  # test_language_mode = train
    assert point["val_langs"] == ["su"]


def test_resolve_run_point_source_matrix(merged_config):
    cfg = {
        **merged_config,
        "experiment": {
            **merged_config["experiment"],
            "matrix": {"sources": ["id", "ms"], "targets": ["jv", "su"]},
            "train": {"language": "source", "size": 10000},
            "validation": {"language": "source", "size_per_language": 10},
        },
    }
    point = resolve_run_point(cfg, source="ms")
    assert point["train_langs"] == ["ms"]
    assert point["eval_langs"] == ["jv", "su"]
    assert point["val_langs"] == ["ms"]
    assert point["target_validation_langs"] == ["jv", "su"]


def test_resolve_run_point_rejects_unknown_source(merged_config):
    from axner.errors import ConfigError

    import pytest

    cfg = {
        **merged_config,
        "experiment": {
            **merged_config["experiment"],
            "matrix": {"sources": ["id", "ms"], "targets": ["jv"]},
            "train": {"language": "source", "size": 10000},
        },
    }
    with pytest.raises(ConfigError):
        resolve_run_point(cfg, source="en")


def test_variant_tags_for_run_modes(merged_config):
    tags_target = variant_tags(merged_config, resolve_run_point(merged_config, target="jv"))
    assert tags_target["train_language"] == "jv"

    cfg = {
        **merged_config,
        "experiment": {
            **merged_config["experiment"],
            "matrix": {"sources": ["id"], "targets": ["jv"]},
            "train": {"language": "source", "size": 10000},
        },
    }
    tags_source = variant_tags(cfg, resolve_run_point(cfg, source="id"))
    assert tags_source == {"source": "id", "train_size": 10000}


def test_execute_run_dry_run_creates_artifacts(synthetic_config, tmp_path):
    summary = execute_run(synthetic_config, output_base=tmp_path, run_label="dryrun", dry_run=True)
    assert summary["status"] == "dry_run_completed"
    assert summary["train_rows"] == 32
    assert summary["test_rows"] == {"jv": 8}

    run_dir = Path(summary["run_dir"])
    for subdirectory in ("config", "data", "checkpoints", "logs", "predictions", "metrics"):
        assert (run_dir / subdirectory).is_dir()
    manifest = load_manifest(run_dir)
    assert manifest["status"] == "dry_run_completed"
    assert manifest["experiment_id"] == "smoke_test"
    assert manifest["training_seed"] == 42

    # resolved config snapshot matches the validated config
    snapshot = load_yaml(run_dir / "config" / "config_resolved.yaml")
    assert snapshot["experiment"]["id"] == "smoke_test"

    # data artifacts exist and are consistent
    sampling = load_json(run_dir / "data" / "sampling_indices.json")
    assert len(sampling) == 3  # train, validation, test
    stats = load_json(run_dir / "data" / "dataset_stats.json")
    assert stats["train"]["n_sentences"] == 32
    audit = load_json(run_dir / "data" / "data_audit.json")
    assert audit["train"]["scan"]["n_invalid_examples"] == 0
