"""Offline helpers for CPU smoke tests.

``WhitespaceTokenizer`` implements the minimal fast-tokenizer API used by
``axner.data.alignment`` (call + ``word_ids``) without any downloads, and
``build_tiny_model`` builds a tiny randomly initialized BERT. These exist to
validate the PIPELINE end-to-end on CPU; they are not research models and
produce no meaningful quality numbers.
"""

from __future__ import annotations

import hashlib


class OfflineBatch:
    """Minimal stand-in for ``transformers.BatchEncoding`` (single example)."""

    def __init__(self, data: dict, word_ids_list: list):
        self._data = data
        self._word_ids = word_ids_list

    def __getitem__(self, key):
        return self._data[key]

    def __contains__(self, key):
        return key in self._data

    def keys(self):
        return self._data.keys()

    def word_ids(self, batch_index: int = 0) -> list:
        if batch_index != 0:
            raise IndexError("OfflineBatch holds a single example (batch_index=0 only)")
        return list(self._word_ids)


class WhitespaceTokenizer:
    """Deterministic whitespace tokenizer: [BOS] word ids [EOS].

    Word ids are derived from an MD5 hash (stable across processes,
    independent of PYTHONHASHSEED) into ``[3, vocab_size)``. Words containing
    "-" are split into multiple subtokens sharing one word id, which exercises
    the non-first-subtoken masking path in alignment.
    """

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __init__(self, vocab_size: int = 512, max_length: int = 32):
        if vocab_size <= 3:
            raise ValueError("vocab_size must leave room for special tokens")
        self.vocab_size = vocab_size
        self.max_length = max_length

    def _word_id(self, word: str) -> int:
        digest = hashlib.md5(word.encode("utf-8")).digest()
        return 3 + int.from_bytes(digest[:8], "big") % (self.vocab_size - 3)

    def _subtokens(self, word: str) -> list[str]:
        return word.split("-") if "-" in word else [word]

    def save_pretrained(self, output_dir) -> None:
        """No-op: the HF Trainer calls this when saving checkpoints."""
        return None

    def __call__(self, words, is_split_into_words: bool = False, truncation: bool = False,
                 max_length: int | None = None, **kwargs) -> OfflineBatch:
        if isinstance(words, str):
            words = words.split() if is_split_into_words else [words]
        words = list(words)
        limit = max_length or self.max_length

        sub_ids: list[int] = []
        word_ids: list[int | None] = []
        for word_index, word in enumerate(words):
            for part in self._subtokens(word):
                sub_ids.append(self._word_id(part))
                word_ids.append(word_index)
        sequence = [self.bos_token_id, *sub_ids, self.eos_token_id]
        sequence_word_ids: list[int | None] = [None, *word_ids, None]

        if truncation and len(sequence) > limit:
            sequence = sequence[:limit]
            sequence_word_ids = sequence_word_ids[:limit]

        return OfflineBatch(
            {"input_ids": sequence, "attention_mask": [1] * len(sequence)},
            sequence_word_ids,
        )


def build_tiny_model(label_names, seed: int = 0, vocab_size: int = 512,
                     hidden_size: int = 32, num_hidden_layers: int = 2,
                     num_attention_heads: int = 4, intermediate_size: int = 64,
                     max_position_embeddings: int = 64):
    """Build a tiny, randomly initialized BERT token classifier (CPU smoke tests)."""
    import torch
    from transformers import BertConfig, BertForTokenClassification

    torch.manual_seed(seed)
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        num_labels=len(label_names),
        id2label={i: name for i, name in enumerate(label_names)},
        label2id={name: i for i, name in enumerate(label_names)},
    )
    return BertForTokenClassification(config)
