"""Tests for the remove-the-best-group concentration check.

Constructions where the truth is known by design:
1. Edge spread evenly across days -> must NOT flag.
2. All profit manufactured into a single day, rest slightly negative
   -> must flag a sign flip.
"""
import numpy as np
import pytest

from phantomguard.concentration import concentration_check


def test_broad_edge_not_flagged():
    rng = np.random.default_rng(1)
    # 20 days x 20 trades, genuine positive mean everywhere.
    v = rng.normal(0.3, 1.0, 400)
    days = np.repeat(np.arange(20), 20)
    r = concentration_check(v, days)
    assert not r.sign_flip
    assert not r.concentrated
    assert r.mean_ex_best > 0
    assert r.positive_groups >= 15


def test_one_day_wonder_is_caught():
    rng = np.random.default_rng(2)
    days = np.repeat(np.arange(10), 30)
    v = rng.normal(-0.05, 0.5, 300)     # slightly negative everywhere...
    v[days == 3] += 2.0                  # ...except one glorious day
    r = concentration_check(v, days)
    assert r.mean_all > 0                # headline looks good
    assert r.sign_flip                   # but it is one day's story
    assert r.concentrated
    assert r.best_group == 3
    assert any("SIGN FLIP" in n for n in r.notes)


def test_share_warning_without_flip():
    # Positive everywhere, but one day carries most of the money.
    days = np.repeat(np.arange(5), 10)
    v = np.full(50, 0.1)
    v[days == 2] = 3.0
    r = concentration_check(v, days)
    assert not r.sign_flip
    assert r.mean_ex_best > 0
    assert r.best_group_share > 0.5
    assert r.concentrated
    assert any("fragile" in n for n in r.notes)


def test_validation_errors():
    with pytest.raises(ValueError):
        concentration_check([1.0, 2.0], [1])         # length mismatch
    with pytest.raises(ValueError):
        concentration_check([1.0, 2.0], [1, 1])      # only one group
