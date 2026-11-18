"""Entity-level metrics with exact span matching.

A span is ``(type, start, end)`` with inclusive token positions. An entity
counts as correct only when type AND both boundaries match exactly.

Gold tag sequences are validated IOB2 by construction (see
``axner.data.validation``). Model PREDICTIONS can violate IOB2, so
:func:`extract_spans` applies a documented decode-time recovery: an
``I-<TYPE>`` that does not continue an open ``<TYPE>`` span starts a new
span. This policy affects only predicted sequences, never gold annotations.
"""

from __future__ import annotations

from collections import Counter

from axner.constants import ENTITY_TYPES


def extract_spans(tags) -> list[tuple[str, int, int]]:
    """Extract ``(type, start, end)`` spans from an IOB2 tag-name sequence.

    Invalid continuations (``I-X`` without an open ``X`` span) start a new
    span instead of raising, which keeps model predictions scoreable; gold
    data should never hit this path because it is validated upstream. Tags
    outside the supported entity types raise ``ValueError``.
    """
    allowed_types = set(ENTITY_TYPES)
    spans: list[tuple[str, int, int]] = []
    open_type: str | None = None
    open_start = -1
    for i, tag in enumerate(tags):
        if tag == "O":
            if open_type is not None:
                spans.append((open_type, open_start, i - 1))
                open_type = None
        elif tag.startswith("B-"):
            itype = tag[2:]
            if itype not in allowed_types:
                raise ValueError(f"unknown entity type in tag {tag!r} at position {i}")
            if open_type is not None:
                spans.append((open_type, open_start, i - 1))
            open_type, open_start = itype, i
        elif tag.startswith("I-"):
            itype = tag[2:]
            if itype not in allowed_types:
                raise ValueError(f"unknown entity type in tag {tag!r} at position {i}")
            if open_type != itype:
                if open_type is not None:
                    spans.append((open_type, open_start, i - 1))
                open_type, open_start = itype, i
            # else: valid continuation of the open span
        else:
            raise ValueError(f"unknown tag name {tag!r} at position {i}")
    if open_type is not None:
        spans.append((open_type, open_start, len(tags) - 1))
    return spans


def _counts(gold_seqs, pred_seqs) -> tuple[int, int, int]:
    tp = fp = fn = 0
    for gold, pred in zip(gold_seqs, pred_seqs):
        gset = set(extract_spans(gold))
        pset = set(extract_spans(pred))
        tp += len(gset & pset)
        fp += len(pset - gset)
        fn += len(gset - pset)
    return tp, fp, fn


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def compute_entity_metrics(gold_seqs, pred_seqs) -> dict:
    """Micro-averaged entity precision/recall/F1 over exact span matches."""
    if len(gold_seqs) != len(pred_seqs):
        raise ValueError(f"gold/prediction count mismatch: {len(gold_seqs)} vs {len(pred_seqs)}")
    tp, fp, fn = _counts(gold_seqs, pred_seqs)
    out = _prf(tp, fp, fn)
    out.update({"tp": tp, "fp": fp, "fn": fn, "n_sentences": len(gold_seqs)})
    return out


def per_type_metrics(gold_seqs, pred_seqs, types=ENTITY_TYPES) -> dict:
    """Per-entity-type precision/recall/F1 (micro over that type's spans)."""
    results = {}
    for etype in types:
        tp = fp = fn = 0
        for gold, pred in zip(gold_seqs, pred_seqs):
            gset = {s for s in extract_spans(gold) if s[0] == etype}
            pset = {s for s in extract_spans(pred) if s[0] == etype}
            tp += len(gset & pset)
            fp += len(pset - gset)
            fn += len(gset - pset)
        entry = _prf(tp, fp, fn)
        entry.update({"tp": tp, "fp": fp, "fn": fn})
        results[etype] = entry
    return results


def error_summary(gold_seqs, pred_seqs, types=ENTITY_TYPES) -> dict:
    """Coarse error taxonomy over non-matching spans.

    Classification per sentence, greedy over unmatched spans:
      * ``type_confusions``: identical boundaries, different type
        (counted in ``type_confusion_matrix`` as ``gold->pred``);
      * ``boundary_errors``: same type, overlapping boundaries, not exact;
      * ``spurious``: predicted spans matching no gold span;
      * ``missed``: gold spans matched by no prediction.
    """
    n_type_confusions = 0
    n_boundary_errors = 0
    n_spurious = 0
    n_missed = 0
    confusion_matrix: Counter = Counter()
    boundary_examples = []

    for gold, pred in zip(gold_seqs, pred_seqs):
        gspans = extract_spans(gold)
        pspans = extract_spans(pred)
        gset, pset = set(gspans), set(pspans)
        fp = list(pset - gset)
        fn = list(gset - pset)
        used_fn = set()

        # 1) type confusions: exact boundary match, wrong type
        remaining_fp = []
        for p in fp:
            match = next((g for g in fn if g[1] == p[1] and g[2] == p[2]), None)
            if match is not None:
                n_type_confusions += 1
                confusion_matrix[f"{match[0]}->{p[0]}"] += 1
                used_fn.add(match)
            else:
                remaining_fp.append(p)

        # 2) boundary errors: same type with overlapping (non-exact) boundaries
        remaining_fn = [g for g in fn if g not in used_fn]
        for p in remaining_fp:
            match = next(
                (g for g in remaining_fn
                 if g[0] == p[0] and p[1] <= g[2] and g[1] <= p[2]),
                None,
            )
            if match is not None:
                n_boundary_errors += 1
                if len(boundary_examples) < 10:
                    boundary_examples.append({"gold": match, "pred": p})
                remaining_fn.remove(match)
            else:
                n_spurious += 1
        n_missed += len(remaining_fn)

    return {
        "type_confusions": n_type_confusions,
        "type_confusion_matrix": dict(confusion_matrix),
        "boundary_errors": n_boundary_errors,
        "boundary_examples": boundary_examples,
        "spurious": n_spurious,
        "missed": n_missed,
        "entity_types": list(types),
    }
