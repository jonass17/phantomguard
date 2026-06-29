import numpy as np
import pytest

from phantomguard import (
    sharpe_ratio, probabilistic_sharpe_ratio, deflated_sharpe_ratio,
    expected_max_sharpe, min_track_record_length, bootstrap_sharpe_ci,
    walk_forward_splits, purged_kfold, positive_fold_fraction,
    lookahead_flag, stale_value_flag, evaluate,
)


def test_sharpe_basic():
    r = np.array([0.01, 0.01, 0.01, 0.01])
    # zero variance -> defined as 0.0, not inf
    assert sharpe_ratio(r) == 0.0
    r2 = np.array([0.02, -0.01, 0.03, 0.01])
    assert sharpe_ratio(r2) == pytest.approx(r2.mean() / r2.std(ddof=1))


def test_psr_monotonic_in_sample_size():
    rng = np.random.default_rng(1)
    small = rng.normal(0.05, 1.0, 50)
    big = np.concatenate([small, rng.normal(0.05, 1.0, 2000)])
    # more evidence at the same effect size -> higher PSR
    assert probabilistic_sharpe_ratio(big) >= probabilistic_sharpe_ratio(small)


def test_psr_bounds():
    rng = np.random.default_rng(2)
    p = probabilistic_sharpe_ratio(rng.normal(0.01, 0.01, 300))
    assert 0.0 <= p <= 1.0


def test_psr_constant_series_is_half_not_nan():
    # zero-variance returns: no edge -> PSR resolves to 0.5, never NaN
    p = probabilistic_sharpe_ratio([0.02, 0.02, 0.02, 0.02])
    assert np.isfinite(p)
    assert p == pytest.approx(0.5)


def test_expected_max_sharpe_grows_with_trials():
    a = expected_max_sharpe(0.01, 10)
    b = expected_max_sharpe(0.01, 1000)
    assert b > a > 0
    assert expected_max_sharpe(0.01, 1) == 0.0


def test_dsr_below_psr_under_many_trials():
    rng = np.random.default_rng(3)
    r = rng.normal(0.001, 0.01, 600)
    psr = probabilistic_sharpe_ratio(r)
    dsr = deflated_sharpe_ratio(r, n_trials=500, sr_variance=0.02)
    # deflation can only lower (or equal) the probability vs the zero benchmark
    assert dsr <= psr + 1e-9


def test_min_trl_infinite_when_no_edge():
    r = np.array([0.01, -0.01, 0.01, -0.01, 0.01, -0.01])
    assert min_track_record_length(r) == float("inf")


def test_bootstrap_ci_orders():
    rng = np.random.default_rng(4)
    point, lo, hi = bootstrap_sharpe_ci(rng.normal(0.05, 1.0, 400), n_boot=2000)
    assert lo <= point <= hi


def test_walk_forward_is_causal():
    for train, test in walk_forward_splits(100, n_splits=4):
        assert train.max() < test.min()  # training strictly precedes test


def test_purged_kfold_embargo_excludes_neighbours():
    folds = list(purged_kfold(100, n_splits=5, embargo=0.1))
    assert len(folds) == 5
    for train, test in folds:
        assert set(train).isdisjoint(set(test))


def test_positive_fold_fraction():
    assert positive_fold_fraction([1, -1, 1, 1]) == 0.75


def test_lookahead_flag_detects_leak():
    rng = np.random.default_rng(5)
    contemp = rng.normal(0, 1, 500)
    signal = contemp + rng.normal(0, 0.01, 500)   # signal == this bar's return
    forward = rng.normal(0, 1, 500)               # unrelated to the future
    flagged, _ = lookahead_flag(signal, forward, contemp)
    assert flagged


def test_stale_value_flag():
    x = np.r_[np.ones(60), np.arange(40)]   # 60% frozen
    flagged, _ = stale_value_flag(x, max_repeat_frac=0.2)
    assert flagged


def test_evaluate_passes_clean_edge():
    rng = np.random.default_rng(6)
    real = rng.normal(0.0015, 0.008, 1500)
    v = evaluate(real, n_trials=4, sr_variance=0.005,
                 fold_pnls=[0.1, 0.05, 0.08, 0.07, 0.09], periods_per_year=252)
    assert v.passed, str(v)


def test_evaluate_rejects_phantom():
    rng = np.random.default_rng(7)
    phantom = rng.normal(0.0003, 0.012, 400)
    v = evaluate(phantom, n_trials=1000, sr_variance=0.05,
                 fold_pnls=[0.1, -0.05, -0.02, 0.03, -0.01], periods_per_year=252)
    assert not v.passed
