"""Prediction record building and JSONL serialization round trips."""

import numpy as np

from axner.constants import IGNORE_INDEX
from axner.data.alignment import align_dataset
from axner.evaluation.export import (
    build_entity_records,
    build_sentence_records,
    export_jsonl,
    load_jsonl,
)


def _aligned_with_predictions(synthetic_dataset, stub_tokenizer, label_names, tmp_path):
    aligned = align_dataset(synthetic_dataset, stub_tokenizer, max_length=32)
    n = aligned.num_rows
    max_len = max(len(row["input_ids"]) for row in aligned)

    label_ids = np.full((n, max_len), IGNORE_INDEX, dtype=np.int64)
    for i, row in enumerate(aligned):
        label_ids[i, : len(row["labels"])] = row["labels"]

    # Half the sentences predicted exactly as gold; half predicted all-O.
    predictions = np.zeros((n, max_len), dtype=np.int64)  # class id 0 == "O"
    for i, row in enumerate(aligned):
        if i % 2 == 0:
            predictions[i, : len(row["labels"])] = row["labels"]

    return aligned, predictions, label_ids


def test_sentence_records_align_with_kept_words(synthetic_dataset, stub_tokenizer, label_names, tmp_path):
    aligned, predictions, label_ids = _aligned_with_predictions(
        synthetic_dataset, stub_tokenizer, label_names, tmp_path
    )
    records = build_sentence_records(aligned, predictions, label_ids, label_names)
    assert len(records) == aligned.num_rows
    for record, row in zip(records, aligned):
        assert len(record["gold_tags"]) == len(record["kept_word_ids"])
        assert len(record["pred_tags"]) == len(record["kept_word_ids"])
        assert record["n_words_kept"] == len(record["kept_word_ids"])
        assert record["tokens"] == [row["tokens"][w] for w in row["kept_word_ids"]]


def test_prediction_export_round_trip(synthetic_dataset, stub_tokenizer, label_names, tmp_path):
    aligned, predictions, label_ids = _aligned_with_predictions(
        synthetic_dataset, stub_tokenizer, label_names, tmp_path
    )
    records = build_sentence_records(aligned, predictions, label_ids, label_names)
    path = export_jsonl(records, tmp_path / "predictions" / "sentence_level.jsonl")
    loaded = load_jsonl(path)
    assert loaded == records


def test_entity_records_kinds(synthetic_dataset, stub_tokenizer, label_names, tmp_path):
    aligned, predictions, label_ids = _aligned_with_predictions(
        synthetic_dataset, stub_tokenizer, label_names, tmp_path
    )
    records = build_sentence_records(aligned, predictions, label_ids, label_names)
    entity_records = build_entity_records(records)

    for i, sentence in enumerate(records):
        sentence_entities = [e for e in entity_records if e["sentence_id"] == sentence["sentence_id"]]
        if i % 2 == 0:
            # predicted exactly as gold: every gold span is a tp, no fps
            assert all(e["kind"] == "tp" for e in sentence_entities)
        else:
            # predicted all-O: every gold span is an fn
            assert all(e["kind"] == "fn" for e in sentence_entities)


def test_entity_records_surface_join(synthetic_dataset, stub_tokenizer, label_names, tmp_path):
    aligned, predictions, label_ids = _aligned_with_predictions(
        synthetic_dataset, stub_tokenizer, label_names, tmp_path
    )
    records = build_sentence_records(aligned, predictions, label_ids, label_names)
    entity_records = build_entity_records(records)
    by_sentence = {}
    for record in records:
        by_sentence[record["sentence_id"]] = record
    for entity in entity_records:
        sentence = by_sentence[entity["sentence_id"]]
        assert entity["surface"] == " ".join(sentence["tokens"][entity["start"]:entity["end"] + 1])
