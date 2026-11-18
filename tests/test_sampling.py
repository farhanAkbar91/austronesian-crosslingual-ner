"""Reproducible sampling and seed derivation."""

import pytest

from axner.data.sampling import (
    indices_sha256,
    language_sampling_seed,
    sample_indices,
    sample_split,
)
from axner.errors import SamplingError
from axner.utils.seed import derive_seed


def test_same_seed_gives_same_indices():
    a = sample_indices(1000, 50, seed=42)
    b = sample_indices(1000, 50, seed=42)
    assert a == b


def test_different_seed_gives_different_indices():
    a = sample_indices(1000, 50, seed=42)
    b = sample_indices(1000, 50, seed=43)
    assert a != b


def test_indices_are_sorted_and_sized():
    indices = sample_indices(1000, 100, seed=7)
    assert len(indices) == 100
    assert indices == sorted(indices)
    assert all(0 <= i < 1000 for i in indices)
    assert len(set(indices)) == 100  # no replacement


def test_full_availability_returns_all_indices():
    assert sample_indices(10, 10, seed=1) == list(range(10))


def test_requesting_more_than_available_raises():
    with pytest.raises(SamplingError):
        sample_indices(5, 10, seed=1)


def test_unknown_method_raises():
    with pytest.raises(SamplingError):
        sample_indices(10, 5, seed=1, method="shuffled_greedily")


def test_sample_split_record_fields(synthetic_dataset):
    indices, record, sampled = sample_split(synthetic_dataset, 10, seed=42, lang="jv", split="train")
    assert sampled.num_rows == 10
    assert record["language"] == "jv"
    assert record["n_requested"] == 10
    assert record["n_available"] == synthetic_dataset.num_rows
    assert record["seed"] == 42
    assert record["indices_sha256"] == indices_sha256(indices)


def test_derive_seed_is_stable_and_sensitive():
    assert derive_seed(42, "sample", "id", "train") == derive_seed(42, "sample", "id", "train")
    assert derive_seed(42, "sample", "id", "train") != derive_seed(42, "sample", "ms", "train")
    assert derive_seed(42, "sample", "id") != derive_seed(43, "sample", "id")
    value = derive_seed(0, "anything")
    assert 0 <= value < 2**31 - 1


def test_language_sampling_seeds_differ_per_language():
    seeds = {language_sampling_seed(42, lang) for lang in ("id", "ms", "tl", "en", "jv", "su")}
    assert len(seeds) == 6
