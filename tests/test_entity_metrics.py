"""Entity-level metric calculation with exact span matching."""

import pytest

from axner.evaluation.entity_metrics import (
    compute_entity_metrics,
    error_summary,
    extract_spans,
    per_type_metrics,
)


def test_extract_spans_simple():
    spans = extract_spans(["B-PER", "I-PER", "O", "B-LOC"])
    assert spans == [("PER", 0, 1), ("LOC", 3, 3)]


def test_extract_spans_empty():
    assert extract_spans(["O", "O"]) == []
    assert extract_spans([]) == []


def test_extract_spans_recovery_for_invalid_predictions():
    # Predictions may violate IOB2; the documented recovery starts a new span.
    spans = extract_spans(["I-PER", "I-PER", "O", "I-LOC"])
    assert spans == [("PER", 0, 1), ("LOC", 3, 3)]


def test_extract_spans_unknown_tag_raises():
    with pytest.raises(ValueError):
        extract_spans(["B-DATE"])


def test_compute_entity_metrics_hand_computed():
    gold = [["B-PER", "O", "B-LOC"]]
    pred = [["B-PER", "O", "O"]]
    metrics = compute_entity_metrics(gold, pred)
    assert metrics["tp"] == 1 and metrics["fp"] == 0 and metrics["fn"] == 1
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == round(2 / 3, 6)


def test_compute_entity_metrics_multiple_sentences():
    gold = [["B-PER"], ["B-LOC"], ["O"]]
    pred = [["B-PER"], ["O"], ["B-ORG"]]
    metrics = compute_entity_metrics(gold, pred)
    assert metrics["tp"] == 1 and metrics["fp"] == 1 and metrics["fn"] == 1
    assert metrics["n_sentences"] == 3


def test_compute_entity_metrics_type_mismatch_is_not_a_match():
    gold = [["B-PER", "I-PER"]]
    pred = [["B-LOC", "I-LOC"]]
    metrics = compute_entity_metrics(gold, pred)
    assert metrics["tp"] == 0 and metrics["fp"] == 1 and metrics["fn"] == 1


def test_compute_entity_metrics_no_entities():
    metrics = compute_entity_metrics([["O"]], [["O"]])
    assert metrics["f1"] == 0.0 and metrics["tp"] == 0


def test_compute_entity_metrics_length_mismatch_raises():
    with pytest.raises(ValueError):
        compute_entity_metrics([["O"]], [])


def test_per_type_metrics_isolated_by_type():
    gold = [["B-PER", "B-LOC"]]
    pred = [["B-PER", "O"]]
    per_type = per_type_metrics(gold, pred)
    assert per_type["PER"]["tp"] == 1 and per_type["PER"]["f1"] == 1.0
    assert per_type["LOC"]["fn"] == 1 and per_type["LOC"]["recall"] == 0.0
    assert per_type["ORG"]["tp"] == 0


def test_error_summary_type_confusion():
    summary = error_summary([["B-PER"]], [["B-LOC"]])
    assert summary["type_confusions"] == 1
    assert summary["type_confusion_matrix"] == {"PER->LOC": 1}
    assert summary["spurious"] == 0 and summary["missed"] == 0


def test_error_summary_boundary_error():
    # gold span 0-1, pred span 0-0: same type, overlapping, not exact
    summary = error_summary([["B-PER", "I-PER"]], [["B-PER", "O"]])
    assert summary["boundary_errors"] == 1
    assert summary["boundary_examples"][0] == {"gold": ("PER", 0, 1), "pred": ("PER", 0, 0)}


def test_error_summary_spurious_and_missed():
    summary = error_summary([["B-PER"], ["O"]], [["O"], ["B-ORG"]])
    assert summary["missed"] == 1
    assert summary["spurious"] == 1
    assert summary["type_confusions"] == 0 and summary["boundary_errors"] == 0
