"""Token-label alignment and subword handling (offline stub tokenizer)."""

from axner.constants import IGNORE_INDEX
from axner.data.alignment import (
    align_dataset,
    align_tokens_labels,
    truncation_stats,
    word_level_tags_from_labels,
)


def test_first_subtoken_gets_word_tag(stub_tokenizer):
    # "Bank-Nusa" splits into two subtokens sharing word id 1.
    aligned = align_tokens_labels(stub_tokenizer, ["Budi", "Bank-Nusa", "ing"], [1, 3, 0], max_length=32)
    assert aligned["labels"] == [IGNORE_INDEX, 1, 3, IGNORE_INDEX, 0, IGNORE_INDEX]
    assert aligned["input_ids"][0] == stub_tokenizer.bos_token_id
    assert aligned["input_ids"][-1] == stub_tokenizer.eos_token_id
    assert aligned["kept_word_ids"] == [0, 1, 2]
    assert aligned["n_words"] == 3
    assert aligned["n_words_kept"] == 3


def test_special_tokens_are_ignored(stub_tokenizer):
    aligned = align_tokens_labels(stub_tokenizer, ["saka"], [0], max_length=32)
    assert aligned["labels"][0] == IGNORE_INDEX
    assert aligned["labels"][-1] == IGNORE_INDEX
    assert aligned["labels"][1] == 0


def test_truncation_excludes_cut_words(stub_tokenizer):
    aligned = align_tokens_labels(
        stub_tokenizer, ["Budi", "Bank-Nusa", "ing"], [1, 3, 0], max_length=4
    )
    # sequence cut to [BOS, Budi, Bank, Nusa] -> words 0 and 1 kept, word 2 dropped
    assert aligned["n_words_kept"] == 2
    assert aligned["kept_word_ids"] == [0, 1]
    assert aligned["labels"] == [IGNORE_INDEX, 1, 3, IGNORE_INDEX]


def test_word_level_decode_round_trip(stub_tokenizer, label_names):
    tokens = ["Budi", "Bank-Nusa", "saka", "Jakarta"]
    tags = [1, 3, 0, 5]
    aligned = align_tokens_labels(stub_tokenizer, tokens, tags, max_length=32)
    decoded = word_level_tags_from_labels(aligned["labels"], label_names)
    assert decoded == [label_names[t] for t in tags]


def test_align_dataset_keeps_export_columns(stub_tokenizer, synthetic_dataset):
    aligned = align_dataset(synthetic_dataset, stub_tokenizer, max_length=32)
    for column in ("input_ids", "attention_mask", "labels", "kept_word_ids", "tokens"):
        assert column in aligned.column_names
    for column in ("ner_tags", "langs", "spans"):
        assert column not in aligned.column_names
    row = aligned[0]
    assert len(row["input_ids"]) == len(row["labels"]) == len(row["attention_mask"])
    assert len(row["kept_word_ids"]) == row["n_words_kept"]


def test_truncation_stats(synthetic_dataset, stub_tokenizer):
    aligned = align_dataset(synthetic_dataset, stub_tokenizer, max_length=32)
    stats = truncation_stats(aligned)
    assert stats["n_examples"] == aligned.num_rows
    assert stats["n_truncated_examples"] == 0
    assert stats["fraction_words_kept"] == 1.0

    truncated = align_dataset(synthetic_dataset, stub_tokenizer, max_length=8)
    stats = truncation_stats(truncated)
    assert stats["n_truncated_examples"] > 0
    assert 0.0 < stats["fraction_words_kept"] < 1.0
