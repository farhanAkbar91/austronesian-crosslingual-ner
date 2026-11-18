"""Bootstrap confidence intervals (sentence-level and paired)."""

from axner.evaluation.bootstrap import bootstrap_ci, paired_bootstrap

GOLD = [["B-PER", "O"], ["B-LOC", "B-LOC"], ["O", "O"], ["B-ORG", "I-ORG"], ["B-PER"]]


def test_perfect_predictions_have_degenerate_ci():
    ci = bootstrap_ci(GOLD, GOLD, n_resamples=200, seed=42)
    assert ci["observed"] == 1.0
    assert ci["ci_low"] == 1.0 and ci["ci_high"] == 1.0


def test_bootstrap_is_deterministic_per_seed():
    pred_a = [["B-PER", "O"], ["O", "B-LOC"], ["O", "O"], ["B-ORG", "O"], ["O"]]
    first = bootstrap_ci(GOLD, pred_a, n_resamples=200, seed=7)
    second = bootstrap_ci(GOLD, pred_a, n_resamples=200, seed=7)
    assert first == second


def test_bootstrap_ci_bounds_are_ordered():
    pred = [["O"] for _ in GOLD]
    ci = bootstrap_ci(GOLD, pred, n_resamples=200, seed=42)
    assert 0.0 <= ci["ci_low"] <= ci["ci_high"] <= 1.0
    assert ci["n_sentences"] == len(GOLD)
    assert ci["n_resamples"] == 200


def test_bootstrap_reported_observed_matches_micro_f1():
    pred = [["B-PER", "O"], ["B-LOC", "B-LOC"], ["B-ORG", "O"], ["O", "O"], ["B-PER"]]
    ci = bootstrap_ci(GOLD, pred, n_resamples=100, seed=1)
    # tp = 4 (PER s0, LOC s1 twice, PER s4), fp = 1 (ORG s2), fn = 1 (ORG s3)
    assert ci["observed"] == round(2 * 4 / (2 * 4 + 1 + 1), 6)


def test_paired_bootstrap_detects_better_system():
    pred_better = [["B-PER", "O"], ["B-LOC", "B-LOC"], ["O", "O"], ["B-ORG", "I-ORG"], ["B-PER"]]
    pred_worse = [["O", "O"], ["O", "O"], ["O", "O"], ["O", "O"], ["O"]]
    result = paired_bootstrap(GOLD, pred_better, pred_worse, n_resamples=200, seed=42)
    assert result["observed_delta"] > 0
    assert result["ci_low"] > 0  # strictly better on every resample
    assert result["p_value_approx"] <= 0.05


def test_paired_bootstrap_shares_resample_indices():
    # Identical systems must yield a degenerate delta distribution.
    pred = [["B-PER", "O"], ["O", "B-LOC"], ["O", "O"], ["B-ORG", "O"], ["O"]]
    result = paired_bootstrap(GOLD, pred, pred, n_resamples=200, seed=42)
    assert result["observed_delta"] == 0.0
    assert result["ci_low"] == 0.0 and result["ci_high"] == 0.0
