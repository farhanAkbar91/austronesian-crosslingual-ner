"""Shared pytest fixtures.

All fixtures are synthetic and CPU-only: no dataset downloads, no GPU, and
(the torch-marked tests aside) no torch requirement.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from axner.constants import LABEL_NAMES
from axner.data.synthetic import build_synthetic_rows, synthetic_to_hf
from axner.models.offline import WhitespaceTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def label_names() -> list[str]:
    return list(LABEL_NAMES)


@pytest.fixture
def stub_tokenizer() -> WhitespaceTokenizer:
    return WhitespaceTokenizer(vocab_size=256, max_length=32)


@pytest.fixture
def synthetic_rows() -> list[dict]:
    return build_synthetic_rows(40, seed=7)


@pytest.fixture
def synthetic_dataset(synthetic_rows):
    return synthetic_to_hf(synthetic_rows)


@pytest.fixture
def merged_config() -> dict:
    """A valid merged configuration mirroring configs/*.yaml (EXP-0 style)."""
    return {
        "project": {
            "name": "austronesian-crosslingual-ner",
            "version": "0.1.0",
            "description": "test config",
            "label_scheme": "iob2",
            "entity_types": ["PER", "LOC", "ORG"],
            "languages": {"sources": ["id", "ms", "tl", "en"], "targets": ["jv", "su"]},
            "known_max_train_sizes": {"id": 20000, "ms": 20000, "tl": 10000, "en": 20000, "jv": 100, "su": 100},
            "known_max_sizes_verified": True,
            "paths": {"outputs_dir": "outputs", "data_dir": "data"},
        },
        "data": {
            "dataset": {"name": "wikiann", "revision": None, "cache_dir": "data/hf_cache"},
            "splits": {"train": "train", "validation": "validation", "test": "test"},
            "label_names": list(LABEL_NAMES),
            "sampling": {"method": "uniform_no_replacement", "seed": 42},
            "invalid_annotations": {"policy": "report_only"},
            "duplicates": {"policy": "report_only"},
        },
        "model": {
            "name": "FacebookAI/xlm-roberta-base",
            "max_length": 192,
            "subword_policy": "first_subtoken_label",
            "tiny_random": False,
            "cache_dir": "data/hf_cache",
        },
        "training": {
            "enabled": True,
            "hyperparameters": {
                "learning_rate": 2e-5, "batch_size": 16, "epochs": 3,
                "weight_decay": 0.01, "warmup_ratio": 0.1, "max_grad_norm": 1.0,
            },
            "eval_batch_size": 32,
            "fp16": False,
            "deterministic": True,
            "logging_steps": 50,
            "device": "auto",
            "checkpoint": {"selection_metric": "entity_f1", "save_total_limit": 2},
        },
        "evaluation": {
            "validation_language": "source",
            "validation_size_per_language": 1000,
            "test_language_mode": "all_targets",
            "eval_target_validation": True,
            "metrics": ["entity_precision", "entity_recall", "entity_f1", "per_type_f1"],
            "bootstrap": {"n_resamples": 1000, "seed": 42, "confidence_level": 0.95},
            "export": {"sentence_level": True, "entity_level": True},
        },
        "experiment": {
            "id": "exp0_inlanguage_baseline",
            "description": "test",
            "training_seeds": [42, 43, 44],
            "sampling_seed": 42,
            "targets": ["jv", "su"],
            "train": {"language": "target", "size": 100},
            "validation": {"language": "target", "size": None},
            "test_language_mode": "train",
            "training": {"epochs": 10},
        },
    }


@pytest.fixture
def synthetic_config(merged_config: dict) -> dict:
    """A valid offline synthetic smoke-test configuration (tiny random model)."""
    cfg = {
        "data": {
            "dataset": {"name": "synthetic", "revision": None, "cache_dir": "data/hf_cache"},
            "synthetic": {"n_train": 32, "n_validation": 8, "n_test": 8, "seed": 42},
        },
        "model": {**merged_config["model"], "tiny_random": True, "max_length": 32},
        "training": {
            **merged_config["training"],
            "hyperparameters": {
                "learning_rate": 5e-3, "batch_size": 8, "epochs": 1,
                "weight_decay": 0.0, "warmup_ratio": 0.0, "max_grad_norm": 1.0,
            },
            "eval_batch_size": 8,
            "logging_steps": 1,
            "device": "cpu",
            "checkpoint": {"selection_metric": "entity_f1", "save_total_limit": 1},
        },
        "evaluation": {
            "validation_language": "target",
            "validation_size_per_language": None,
            "test_language_mode": "train",
            "eval_target_validation": False,
            "metrics": ["entity_f1"],
            "bootstrap": {"n_resamples": 100, "seed": 42, "confidence_level": 0.95},
            "export": {"sentence_level": True, "entity_level": True},
        },
        "experiment": {
            "id": "smoke_test",
            "description": "offline pipeline validation",
            "training_seeds": [42],
            "sampling_seed": 42,
            "targets": ["jv"],
            "train": {"language": "target", "size": None},
            "validation": {"language": "target", "size": None},
            "test_language_mode": "train",
        },
    }
    from axner.config.loader import deep_merge

    return deep_merge(merged_config, cfg)
