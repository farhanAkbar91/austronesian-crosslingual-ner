"""Multi-source dataset composition."""

from collections import Counter

from axner.data.sampling import compose_multisource


def _part_rows(rows, lang):
    import datasets as hf_datasets

    ds = hf_datasets.Dataset.from_list(rows)
    return ds


def test_composition_totals_and_counts(synthetic_rows):
    import datasets as hf_datasets

    rows_a = synthetic_rows[:20]
    rows_b = synthetic_rows[20:40]
    parts = [
        ("id", hf_datasets.Dataset.from_list(rows_a), 15),
        ("ms", hf_datasets.Dataset.from_list(rows_b), 10),
    ]
    combined, record = compose_multisource(parts, base_seed=42, shuffle=True)

    assert combined.num_rows == 25
    assert record["total"] == 25
    counts = Counter(combined["lang"])
    assert counts == {"id": 15, "ms": 10}
    assert record["shuffle"] is True
    assert record["shuffle_seed"] is not None
    assert len(record["sources"]) == 2
    for source in record["sources"]:
        assert source["n_sampled"] == source["n_requested"]
        assert source["seed"] is not None
        assert source["indices_sha256"]


def test_composition_is_deterministic(synthetic_rows):
    import datasets as hf_datasets

    def build():
        parts = [
            ("id", hf_datasets.Dataset.from_list(synthetic_rows[:20]), 15),
            ("ms", hf_datasets.Dataset.from_list(synthetic_rows[20:40]), 10),
        ]
        return compose_multisource(parts, base_seed=42, shuffle=True)

    combined_a, record_a = build()
    combined_b, record_b = build()
    assert combined_a["tokens"] == combined_b["tokens"]
    assert combined_a["lang"] == combined_b["lang"]
    assert record_a == record_b


def test_composition_without_shuffle_keeps_language_blocks(synthetic_rows):
    import datasets as hf_datasets

    parts = [
        ("ms", hf_datasets.Dataset.from_list(synthetic_rows[:20]), 10),
        ("id", hf_datasets.Dataset.from_list(synthetic_rows[20:40]), 10),
    ]
    combined, record = compose_multisource(parts, base_seed=42, shuffle=False)
    assert record["shuffle"] is False
    assert record["shuffle_seed"] is None
    langs = combined["lang"]
    # sorted-language order: id block first, then ms
    assert all(lang == "id" for lang in langs[:10])
    assert all(lang == "ms" for lang in langs[10:])


def test_composition_preserves_row_content(synthetic_rows):
    import datasets as hf_datasets

    rows = synthetic_rows[:10]
    parts = [("id", hf_datasets.Dataset.from_list(rows), 10)]
    combined, _ = compose_multisource(parts, base_seed=42, shuffle=False)
    assert combined["tokens"] == [row["tokens"] for row in rows]
    assert combined["ner_tags"] == [row["ner_tags"] for row in rows]
