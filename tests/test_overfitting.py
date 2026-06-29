import numpy as np
import pytest

from phantomguard import pbo_cscv


def test_pbo_noise_is_near_half():
    # 30 strategies of pure noise: selecting the in-sample best has no OOS skill,
    # so PBO should sit near 0.5.
    rng = np.random.default_rng(0)
    M = rng.normal(0, 1, size=(1000, 30))
    res = pbo_cscv(M, n_blocks=10)
    assert 0.30 <= res.pbo <= 0.70, res.pbo


def test_pbo_real_edge_is_low():
    # One column carries a genuine, persistent edge; the rest are noise.
    # The in-sample champion is reliably the real one and it survives OOS,
    # so PBO collapses toward zero.
    rng = np.random.default_rng(1)
    M = rng.normal(0, 1, size=(1500, 25))
    M[:, 0] += 0.25                      # persistent drift, Sharpe ~0.25/obs
    res = pbo_cscv(M, n_blocks=12)
    assert res.pbo < 0.10, res.pbo


def test_pbo_orders_weak_vs_strong_edge():
    # A weaker edge must yield a HIGHER PBO than a stronger one.
    rng = np.random.default_rng(7)
    base = rng.normal(0, 1, size=(1500, 25))
    weak = base.copy();   weak[:, 0] += 0.10
    strong = base.copy(); strong[:, 0] += 0.40
    assert pbo_cscv(weak, n_blocks=12).pbo > pbo_cscv(strong, n_blocks=12).pbo


def test_pbo_bounds_and_shapes():
    rng = np.random.default_rng(2)
    M = rng.normal(0, 1, size=(800, 8))
    res = pbo_cscv(M, n_blocks=8)
    assert 0.0 <= res.pbo <= 1.0
    assert 0.0 <= res.prob_oos_loss <= 1.0
    assert res.logits.shape == res.oos_ranks.shape
    assert res.n_splits == res.logits.size
    assert np.all((res.oos_ranks > 0) & (res.oos_ranks < 1))


def test_pbo_rejects_bad_input():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError):
        pbo_cscv(rng.normal(0, 1, (100, 1)), n_blocks=8)   # need >= 2 strategies
    with pytest.raises(ValueError):
        pbo_cscv(rng.normal(0, 1, (100, 5)), n_blocks=7)   # n_blocks must be even
    with pytest.raises(ValueError):
        pbo_cscv(rng.normal(0, 1, (10, 5)), n_blocks=8)    # too few obs per block
