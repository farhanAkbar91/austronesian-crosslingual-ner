"""Configuration validation: synthetic cases and shipped YAML files."""

from pathlib import Path

import pytest

from axner.config.loader import deep_merge, load_base_configs, load_experiment_config
from axner.config.validation import validate_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def _mutate(cfg, path, value):
    cfg = deep_merge(cfg, {})
    node = cfg
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return cfg


def test_valid_merged_config_passes(merged_config):
    assert validate_config(merged_config) == []


def test_missing_section_is_reported(merged_config):
    cfg = {k: v for k, v in merged_config.items() if k != "evaluation"}
    problems = validate_config(cfg)
    assert any("missing required configuration section 'evaluation'" in p for p in problems)


def test_unknown_language_is_rejected(merged_config):
    cfg = _mutate(merged_config, ["project", "languages", "targets"], ["jv", "xx"])
    problems = validate_config(cfg)
    assert any("supported languages" in p for p in problems)


def test_composition_total_must_match_exactly(merged_config):
    cfg = deep_merge(merged_config, {"experiment": {
        "train": {"language": "source", "size": 10000, "composition": {"id": 5000, "ms": 4999}},
    }})
    problems = validate_config(cfg)
    assert any("9999 != 10000" in p for p in problems)


def test_composition_and_matrix_are_exclusive(merged_config):
    cfg = deep_merge(merged_config, {"experiment": {
        "matrix": {"sources": ["id"], "targets": ["jv"]},
        "train": {"language": "source", "size": 10000, "composition": {"id": 10000}},
    }})
    problems = validate_config(cfg)
    assert any("mutually exclusive" in p for p in problems)


def test_zero_shot_matrix_must_be_disjoint(merged_config):
    cfg = deep_merge(merged_config, {"experiment": {
        "matrix": {"sources": ["jv"], "targets": ["jv"]},
        "train": {"language": "source", "size": 100},
    }})
    problems = validate_config(cfg)
    assert any("disjoint" in p for p in problems)


def test_empty_training_seeds_rejected(merged_config):
    cfg = _mutate(merged_config, ["experiment", "training_seeds"], [])
    problems = validate_config(cfg)
    assert any("training_seeds" in p for p in problems)


def test_duplicate_training_seeds_rejected(merged_config):
    cfg = _mutate(merged_config, ["experiment", "training_seeds"], [42, 42])
    problems = validate_config(cfg)
    assert any("unique" in p for p in problems)


def test_bootstrap_n_resamples_below_100_rejected(merged_config):
    cfg = _mutate(merged_config, ["evaluation", "bootstrap", "n_resamples"], 10)
    problems = validate_config(cfg)
    assert any("n_resamples" in p for p in problems)


def test_non_canonical_label_names_rejected(merged_config):
    cfg = _mutate(merged_config, ["data", "label_names"], ["O", "B-PER"])
    problems = validate_config(cfg)
    assert any("canonical label list" in p for p in problems)


def test_train_size_exceeding_known_availability_rejected(merged_config):
    cfg = deep_merge(merged_config, {"experiment": {
        "targets": ["jv"],
        "train": {"language": "target", "size": 1000},
        "validation": {"language": "target", "size": None},
    }})
    problems = validate_config(cfg)
    assert any("exceeds recorded WikiANN training size" in p for p in problems)


def test_train_size_required_without_fewshot_or_synthetic(merged_config):
    cfg = _mutate(merged_config, ["experiment", "train", "size"], None)
    problems = validate_config(cfg)
    assert any("experiment.train.size must be set" in p for p in problems)


def test_bad_learning_rate_rejected(merged_config):
    cfg = _mutate(merged_config, ["training", "hyperparameters", "learning_rate"], 5)
    problems = validate_config(cfg)
    assert any("learning_rate" in p for p in problems)


def test_init_source_must_differ_from_targets(merged_config):
    cfg = deep_merge(merged_config, {"experiment": {
        "targets": ["id"],
        "init_from": {"experiment": "exp1_singlesource_zeroshot", "source": "id",
                      "selection_metric": "target_validation_entity_f1", "checkpoint": None},
    }})
    problems = validate_config(cfg)
    assert any("must not be a target language" in p for p in problems)


@pytest.mark.parametrize("experiment_id", [
    "exp0_inlanguage_baseline",
    "exp1_singlesource_zeroshot",
    "exp2a_multisource_id_ms",
    "exp2b_multisource_id_ms_en",
    "exp3_fewshot_adaptation",
    "smoke_test",
])
def test_shipped_experiment_configs_validate(experiment_id):
    cfg = load_experiment_config(experiment_id, config_dir=REPO_ROOT / "configs" / "experiments")
    problems = validate_config(cfg)
    assert problems == [], "\n".join(problems)


def test_base_configs_validate_under_stub():
    # Mirrors scripts/validate_configs.py: base files are validated under a
    # minimal stub experiment (they define no experiment section themselves).
    from copy import deepcopy

    cfg = deepcopy(load_base_configs(REPO_ROOT / "configs"))
    cfg["experiment"] = {
        "id": "config_validation_stub",
        "description": "stub experiment used only to validate base configuration files",
        "training_seeds": [42],
        "sampling_seed": 42,
        "targets": ["jv"],
        "train": {"language": "target", "size": 100},
        "validation": {"language": "target", "size": None},
    }
    assert validate_config(cfg) == []
