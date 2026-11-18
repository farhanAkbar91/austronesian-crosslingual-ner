"""IOB2 validation and label-schema checks."""

import datasets as hf_datasets

from axner.data.validation import (
    apply_invalid_policy,
    check_label_schema,
    scan_iob2_issues,
    validate_example,
    validate_iob2_names,
)


def test_valid_sequence_passes():
    tags = ["O", "B-PER", "I-PER", "O", "B-LOC", "I-LOC", "B-LOC", "I-LOC", "O"]
    assert validate_iob2_names(tags) == []


def test_i_tag_at_start_is_invalid():
    issues = validate_iob2_names(["I-PER", "I-PER", "O"])
    assert len(issues) == 1
    assert "position 0" in issues[0]


def test_i_tag_after_o_is_invalid():
    issues = validate_iob2_names(["O", "I-LOC", "I-LOC"])
    assert len(issues) == 1
    assert "position 1" in issues[0]


def test_i_tag_after_different_type_is_invalid():
    issues = validate_iob2_names(["B-PER", "I-LOC"])
    assert len(issues) == 1
    assert "I-LOC" in issues[0]


def test_unknown_tag_name_is_invalid():
    issues = validate_iob2_names(["B-DATE"])
    assert issues and "unknown tag name" in issues[0]


def test_validate_example_reports_out_of_range_ids(label_names):
    issues = validate_example([0, 99], label_names)
    assert any("out of range" in issue for issue in issues)


def test_validate_example_valid_ids_pass(label_names):
    assert validate_example([0, 1, 2, 3, 4, 5, 6], label_names) == []


def test_check_label_schema_detects_order_and_set_mismatch(label_names):
    ds = hf_datasets.Dataset.from_dict({"ner_tags": [[0]]})
    ds = ds.cast_column("ner_tags", hf_datasets.Sequence(
        hf_datasets.ClassLabel(names=["O", "B-LOC", "I-LOC", "B-PER", "I-PER", "B-ORG", "I-ORG"])))
    schema = check_label_schema(ds, label_names)
    assert schema["same_set"] is True
    assert schema["same_order"] is False


def test_scan_synthetic_data_has_no_invalid_examples(synthetic_dataset, label_names):
    scan = scan_iob2_issues(synthetic_dataset, label_names)
    assert scan["n_examples"] == synthetic_dataset.num_rows
    assert scan["n_invalid_examples"] == 0
    assert scan["issue_counts"] == {}


def test_scan_detects_injected_invalid_row(label_names):
    good = {"tokens": ["a", "b"], "ner_tags": [0, 1]}
    bad = {"tokens": ["c", "d"], "ner_tags": [2, 2]}  # I-PER, I-PER from nothing
    ds = hf_datasets.Dataset.from_list([good, bad])
    scan = scan_iob2_issues(ds, label_names)
    assert scan["n_invalid_examples"] == 1
    assert scan["invalid_examples"][0]["index"] == 1


def test_policy_drop_removes_invalid_rows_only(label_names):
    good = {"tokens": ["a", "b"], "ner_tags": [0, 1]}
    bad = {"tokens": ["c", "d"], "ner_tags": [2, 2]}
    ds = hf_datasets.Dataset.from_list([good, bad, good])
    scan = scan_iob2_issues(ds, label_names)
    dropped_ds, applied = apply_invalid_policy(ds, scan, "drop", split_name="train")
    assert applied == {"policy": "drop", "dropped": 1}
    assert dropped_ds.num_rows == 2
    kept_ds, applied = apply_invalid_policy(ds, scan, "report_only")
    assert applied == {"policy": "report_only", "dropped": 0}
    assert kept_ds.num_rows == 3


def test_policy_rejects_unknown_policy(label_names):
    import pytest

    from axner.errors import DataValidationError

    ds = hf_datasets.Dataset.from_list([{"tokens": ["a"], "ner_tags": [0]}])
    scan = scan_iob2_issues(ds, label_names)
    with pytest.raises(DataValidationError):
        apply_invalid_policy(ds, scan, "silently_fix")
