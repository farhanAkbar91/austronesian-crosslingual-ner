"""Prediction export at sentence and entity level.

Exports are JSONL with one record per sentence (tokens, gold tags, predicted
tags at word level for kept words) and one record per entity (spans with
``kind`` = tp / fp / fn). Files round-trip through :func:`load_jsonl`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from axner.constants import IGNORE_INDEX
from axner.evaluation.entity_metrics import extract_spans


def build_sentence_records(eval_ds, predictions, label_ids, label_names) -> list[dict]:
    """Build word-level sentence records from aligned data and raw predictions.

    ``predictions`` are logits of shape (n_sentences, seq_len, n_labels); they
    are converted with argmax. If already class ids (2-D), they pass through.
    """
    predictions = np.asarray(predictions)
    pred_ids = np.argmax(predictions, axis=-1) if predictions.ndim == 3 else predictions
    records = []
    for i, row in enumerate(eval_ds):
        mask = label_ids[i] != IGNORE_INDEX
        gold_names = [label_names[t] for t, keep in zip(label_ids[i], mask) if keep]
        pred_names = [label_names[t] for t, keep in zip(pred_ids[i], mask) if keep]
        kept = row["kept_word_ids"]
        if len(gold_names) != len(kept):
            raise ValueError(
                f"sentence {row.get('sentence_id', i)}: {len(gold_names)} gold labels for "
                f"{len(kept)} kept words - alignment/export mismatch"
            )
        records.append({
            "sentence_id": row.get("sentence_id", f"row-{i}"),
            "lang": row.get("lang"),
            "tokens": [row["tokens"][w] for w in kept],
            "kept_word_ids": list(kept),
            "n_words": row.get("n_words"),
            "n_words_kept": len(kept),
            "gold_tags": gold_names,
            "pred_tags": pred_names,
        })
    return records


def build_entity_records(sentence_records) -> list[dict]:
    """Flatten sentence records into per-entity records tagged tp/fp/fn."""
    entity_records = []
    for sentence in sentence_records:
        gold_spans = extract_spans(sentence["gold_tags"])
        pred_spans = extract_spans(sentence["pred_tags"])
        gset, pset = set(gold_spans), set(pred_spans)
        seen_pred = set()
        for idx, (etype, start, end) in enumerate(gold_spans):
            kind = "tp" if (etype, start, end) in pset else "fn"
            entity_records.append({
                "sentence_id": sentence["sentence_id"],
                "lang": sentence["lang"],
                "entity_index": idx,
                "type": etype,
                "start": start,
                "end": end,
                "surface": " ".join(sentence["tokens"][start:end + 1]),
                "gold_type": etype,
                "pred_type": etype if kind == "tp" else None,
                "kind": kind,
            })
        for etype, start, end in pred_spans:
            if (etype, start, end) not in gset:
                entity_records.append({
                    "sentence_id": sentence["sentence_id"],
                    "lang": sentence["lang"],
                    "entity_index": None,
                    "type": etype,
                    "start": start,
                    "end": end,
                    "surface": " ".join(sentence["tokens"][start:end + 1]),
                    "gold_type": None,
                    "pred_type": etype,
                    "kind": "fp",
                })
    return entity_records


def export_jsonl(records, path) -> Path:
    """Write records to a JSONL file (UTF-8, one JSON object per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def load_jsonl(path) -> list[dict]:
    """Read a JSONL file written by :func:`export_jsonl`."""
    records = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def export_predictions(eval_ds, predictions, label_ids, label_names,
                       sentence_level_path=None, entity_level_path=None) -> dict:
    """Build and export sentence/entity-level prediction files for one target."""
    sentence_records = build_sentence_records(eval_ds, predictions, label_ids, label_names)
    paths = {}
    if sentence_level_path:
        paths["sentence_level"] = export_jsonl(sentence_records, sentence_level_path)
    if entity_level_path:
        paths["entity_level"] = export_jsonl(build_entity_records(sentence_records), entity_level_path)
    return {"records": sentence_records, "paths": {k: str(v) for k, v in paths.items()}}
