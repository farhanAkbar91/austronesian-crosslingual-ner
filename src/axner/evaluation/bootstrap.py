"""Sentence-level bootstrap for test-set uncertainty.

Two distinct resamplings must not be conflated (docs/methodology.md):

* :func:`bootstrap_ci` resamples TEST SENTENCES with replacement and
  quantifies uncertainty from test-set sampling for one system.
* :func:`paired_bootstrap` resamples the same sentences for two systems and
  quantifies the uncertainty of their metric difference.

Seed-level variation across training runs is handled separately by
``axner.evaluation.aggregation`` and must never be presented as test-set
uncertainty.
"""

from __future__ import annotations

import numpy as np

from axner.evaluation.entity_metrics import extract_spans


def _per_sentence_counts(gold_seqs, pred_seqs):
    """Per-sentence (tp, fp, fn) arrays from exact span matching."""
    tp = np.zeros(len(gold_seqs), dtype=np.int64)
    fp = np.zeros(len(gold_seqs), dtype=np.int64)
    fn = np.zeros(len(gold_seqs), dtype=np.int64)
    for i, (gold, pred) in enumerate(zip(gold_seqs, pred_seqs)):
        gset = set(extract_spans(gold))
        pset = set(extract_spans(pred))
        tp[i] = len(gset & pset)
        fp[i] = len(pset - gset)
        fn[i] = len(gset - pset)
    return tp, fp, fn


def _f1_from_counts(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def bootstrap_ci(gold_seqs, pred_seqs, n_resamples: int = 1000, seed: int = 42,
                 confidence_level: float = 0.95) -> dict:
    """Percentile CI for entity micro-F1 over sentence-level resampling."""
    rng = np.random.default_rng(seed)
    tp, fp, fn = _per_sentence_counts(gold_seqs, pred_seqs)
    n = len(gold_seqs)
    observed = _f1_from_counts(tp.sum(), fp.sum(), fn.sum())

    scores = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = rng.integers(0, n, n)
        scores[b] = _f1_from_counts(tp[idx].sum(), fp[idx].sum(), fn[idx].sum())

    alpha = (1 - confidence_level) / 2
    lo, hi = np.quantile(scores, [alpha, 1 - alpha])
    return {
        "metric": "entity_micro_f1",
        "observed": round(float(observed), 6),
        "ci_low": round(float(lo), 6),
        "ci_high": round(float(hi), 6),
        "confidence_level": confidence_level,
        "n_resamples": n_resamples,
        "n_sentences": n,
        "seed": seed,
        "method": "percentile bootstrap over test sentences (with replacement)",
        "interpretation": (
            "uncertainty attributable to test-set sampling; NOT variation across "
            "training seeds (see axner.evaluation.aggregation for seed variation)"
        ),
    }


def paired_bootstrap(gold_seqs, pred_a, pred_b, n_resamples: int = 1000, seed: int = 42,
                     confidence_level: float = 0.95) -> dict:
    """Paired bootstrap for the F1 difference between two systems on one test set.

    Both systems are resampled on the SAME sentence indices per replicate, so
    the CI reflects the difference, not each system's absolute score.
    ``p_value`` is an approximate two-sided bootstrap p-value.
    """
    rng = np.random.default_rng(seed)
    tp_a, fp_a, fn_a = _per_sentence_counts(gold_seqs, pred_a)
    tp_b, fp_b, fn_b = _per_sentence_counts(gold_seqs, pred_b)
    n = len(gold_seqs)
    observed = _f1_from_counts(tp_a.sum(), fp_a.sum(), fn_a.sum()) - _f1_from_counts(
        tp_b.sum(), fp_b.sum(), fn_b.sum()
    )

    deltas = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = rng.integers(0, n, n)
        f_a = _f1_from_counts(tp_a[idx].sum(), fp_a[idx].sum(), fn_a[idx].sum())
        f_b = _f1_from_counts(tp_b[idx].sum(), fp_b[idx].sum(), fn_b[idx].sum())
        deltas[b] = f_a - f_b

    alpha = (1 - confidence_level) / 2
    lo, hi = np.quantile(deltas, [alpha, 1 - alpha])
    p_value = 2 * min(float(np.mean(deltas <= 0)), float(np.mean(deltas >= 0)))
    return {
        "metric": "delta_entity_micro_f1 (system_a - system_b)",
        "observed_delta": round(float(observed), 6),
        "ci_low": round(float(lo), 6),
        "ci_high": round(float(hi), 6),
        "confidence_level": confidence_level,
        "n_resamples": n_resamples,
        "n_sentences": n,
        "seed": seed,
        "p_value_approx": round(min(p_value, 1.0), 6),
        "method": "paired percentile bootstrap over test sentences (shared resample indices)",
    }
