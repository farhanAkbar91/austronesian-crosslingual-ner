"""Dataset split isolation and duplicate detection.

Design invariant under test: train/val/test index sets come from DIFFERENT
WikiANN splits, so sampled index sets are disjoint by construction. Sampling
twice from the SAME underlying split could overlap, which the pipeline never
does; the duplicate detector surfaces any residual content-level leakage.
"""

import datasets as hf_datasets

from axner.data.sampling import sample_indices
from axner.data.stats import (
    cross_split_duplicate_count,
    dataset_statistics,
    duplicates_report,
    find_within_split_duplicates,
)


def _two_disjoint_splits(rows):
    """First 70 rows act as 'train split', last 30 as 'test split'."""
    train_split = hf_datasets.Dataset.from_list(rows[:70])
    test_split = hf_datasets.Dataset.from_list(rows[70:])
    return train_split, test_split


def test_sampled_indices_from_disjoint_splits_are_disjoint(synthetic_rows):
    from axner.data.synthetic import build_synthetic_rows

    rows = build_synthetic_rows(120, seed=8)
    train_split, test_split = _two_disjoint_splits(rows)
    train_indices = sample_indices(70, 50, seed=42)
    test_indices = sample_indices(30, 20, seed=99)
    train_rows = {tuple(train_split[i]["tokens"]) for i in train_indices}
    test_rows = {tuple(test_split[i]["tokens"]) for i in test_indices}
    assert len(train_indices) == 50 and len(test_indices) == 20
    assert not (train_rows & test_rows)


def test_within_split_duplicates_detected():
    row = {"tokens": ["a", "b"], "ner_tags": [0, 1]}
    ds = hf_datasets.Dataset.from_list([row, dict(row), {"tokens": ["c"], "ner_tags": [0]}])
    report = find_within_split_duplicates(ds)
    assert report["n_duplicate_groups"] == 1
    assert report["n_duplicate_rows"] == 1
    assert report["examples"][0]["occurrences"] == 2


def test_cross_split_duplicates_detected(synthetic_rows):
    shared = synthetic_rows[0]
    ds_a = hf_datasets.Dataset.from_list(synthetic_rows[1:15] + [shared])
    ds_b = hf_datasets.Dataset.from_list(synthetic_rows[15:25] + [dict(shared)])
    report = cross_split_duplicate_count(ds_a, ds_b, "train", "test")
    assert report["n_shared_sequences"] == 1


def test_no_false_cross_split_duplicates(synthetic_rows):
    ds_a = hf_datasets.Dataset.from_list(synthetic_rows[:10])
    ds_b = hf_datasets.Dataset.from_list(synthetic_rows[10:20])
    report = cross_split_duplicate_count(ds_a, ds_b, "train", "test")
    assert report["n_shared_sequences"] == 0


def test_duplicates_report_aggregates_splits(synthetic_rows):
    ds_a = hf_datasets.Dataset.from_list(synthetic_rows[:20])
    ds_b = hf_datasets.Dataset.from_list(synthetic_rows[20:40])
    report = duplicates_report({"train": ds_a, "test": ds_b})
    assert set(report["within_split"]) == {"train", "test"}
    assert "train__test" in report["cross_split"]
    assert report["cross_split"]["train__test"]["n_shared_sequences"] == 0


def test_dataset_statistics_shape(synthetic_dataset, label_names):
    stats = dataset_statistics(synthetic_dataset, label_names, "train")
    assert stats["name"] == "train"
    assert stats["n_sentences"] == synthetic_dataset.num_rows
    assert stats["total_tokens"] == sum(len(r["tokens"]) for r in synthetic_dataset)
    assert set(stats["mentions_by_type"]) <= {"PER", "LOC", "ORG"}
    assert stats["total_mentions"] == sum(stats["mentions_by_type"].values())
