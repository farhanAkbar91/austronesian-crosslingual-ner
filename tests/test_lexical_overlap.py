"""Lexical overlap analysis: normalization, inventory, seen/similar/unseen."""

import datasets as hf_datasets

from axner.analysis.lexical_overlap import (
    build_inventory,
    classify_mentions,
    extract_mentions,
    overlap_report,
)
from axner.analysis.normalization import char_ngram_set, normalize_surface


def test_normalize_surface_pipeline():
    assert normalize_surface("  Ápple   Pie ") == "ápple pie"
    assert normalize_surface("JAKARTA") == "jakarta"
    assert normalize_surface("Kutha\r\nkrajan") == "kutha krajan"
    # NFC composes combining marks: "e" + combining acute -> precomposed é
    assert normalize_surface("e\u0301tude") == "étude"


def test_char_ngram_set_boundary_markers():
    grams = char_ngram_set("abc", n_range=(2, 3))
    # marked as ^abc$ -> bigrams/trigrams over the marked string
    assert "^a" in grams and "c$" in grams and "^ab" in grams


def test_extract_mentions_and_inventory(label_names):
    rows = [
        {"tokens": ["Budi", "ing", "Jakarta"], "ner_tags": [1, 0, 5], "sentence_id": "s0"},
        {"tokens": ["Budi", "saka", "Solo"], "ner_tags": [1, 0, 5], "sentence_id": "s1"},
    ]
    ds = hf_datasets.Dataset.from_list(rows)
    mentions = extract_mentions(ds, label_names)
    assert len(mentions) == 4
    assert mentions[0]["surface"] == "Budi" and mentions[0]["type"] == "PER"

    inventory = build_inventory(ds, label_names)
    assert inventory["PER"]["budi"] == 2
    assert inventory["LOC"]["jakarta"] == 1
    assert inventory["LOC"]["solo"] == 1


def test_classify_seen_similar_unseen(label_names):
    source_rows = [
        {"tokens": ["Budi", "ing", "Bank-Nusa"], "ner_tags": [1, 0, 3]},
        {"tokens": ["Jakarta", "kutha"], "ner_tags": [5, 0]},
    ]
    source_ds = hf_datasets.Dataset.from_list(source_rows)
    inventory = build_inventory(source_ds, label_names)

    target_rows = [
        # exact (normalized, type) match -> seen
        {"tokens": ["budi"], "ner_tags": [1], "sentence_id": "t0"},
        # near surface, same type -> similar (high char n-gram overlap)
        {"tokens": ["BankNusa"], "ner_tags": [3], "sentence_id": "t1"},
        # nothing close, different words -> unseen
        {"tokens": ["KorpsLogistik"], "ner_tags": [3], "sentence_id": "t2"},
        # same surface as a source entity but different type -> unseen (type-aware)
        {"tokens": ["Budi"], "ner_tags": [5], "sentence_id": "t3"},
    ]
    target_ds = hf_datasets.Dataset.from_list(target_rows)
    mentions = extract_mentions(target_ds, label_names)

    result = classify_mentions(mentions, inventory, threshold=0.5, n_range=(2, 3, 4),
                               entity_type_aware=True)
    assert result["mentions"]["total"] == 4
    assert result["mentions"]["seen"] == 1
    assert result["mentions"]["similar"] == 1
    assert result["mentions"]["unseen"] == 2
    assert result["by_type"]["PER"]["seen"] == 1
    assert result["by_type"]["LOC"]["unseen"] == 1


def test_type_aware_off_ignores_entity_type(label_names):
    source_ds = hf_datasets.Dataset.from_list(
        [{"tokens": ["Budi", "saka"], "ner_tags": [1, 0]}]
    )
    inventory = build_inventory(source_ds, label_names)
    target_mentions = [{"surface": "Budi", "type": "LOC", "sentence_id": "t0", "start": 0, "end": 0}]
    aware = classify_mentions(target_mentions, inventory, entity_type_aware=True)
    assert aware["mentions"]["seen"] == 0
    not_aware = classify_mentions(target_mentions, inventory, entity_type_aware=False)
    assert not_aware["mentions"]["seen"] == 1


def test_similarity_threshold_controls_similar_band(label_names):
    source_ds = hf_datasets.Dataset.from_list(
        [{"tokens": ["Jakarta", "selatan"], "ner_tags": [5, 0]}]
    )
    inventory = build_inventory(source_ds, label_names)
    # "Jakrta" is NOT an exact normalized match of "jakarta"; it has moderate
    # char n-gram overlap, so the threshold decides similar vs unseen.
    mention = [{"surface": "Jakrta", "type": "LOC", "sentence_id": "t0", "start": 0, "end": 0}]
    assert classify_mentions(mention, inventory, threshold=0.9)["mentions"]["unseen"] == 1
    assert classify_mentions(mention, inventory, threshold=0.3)["mentions"]["similar"] == 1


def test_overlap_report_end_to_end(label_names):
    source_ds = hf_datasets.Dataset.from_list([
        {"tokens": ["Budi", "ing", "Jakarta"], "ner_tags": [1, 0, 5]},
    ])
    target_ds = hf_datasets.Dataset.from_list([
        {"tokens": ["Budi", "lunga", "Solo"], "ner_tags": [1, 0, 5]},
        {"tokens": ["Wati", "nari"], "ner_tags": [1, 0]},
    ])
    report = overlap_report(target_ds, source_ds, label_names, threshold=0.5)
    mentions = report["mentions"]
    assert mentions["seen"] + mentions["similar"] + mentions["unseen"] == mentions["total"] == 3
    assert report["inventory"]["n_unique_entities"]["PER"] == 1
    assert report["inventory"]["n_total_mentions"]["LOC"] == 1
