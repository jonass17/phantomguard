"""Tests for the look-ahead cheat probe and the extra ratios.

Probe tests build tiny synthetic pipelines where we KNOW whether the harness
is wired correctly, then check the probe reaches the right verdict.
Ratio tests check hand-computable values.
"""
import numpy as np
import pytest

from phantomguard.probes import lookahead_cheat_probe
from phantomguard.ratios import (
    calmar_ratio,
    downside_deviation,
    max_drawdown,
    sortino_ratio,
)


# ---------------------------------------------------------------- probes
def _make_market(seed=0, n=2000):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, n)  # bar returns, no real edge


def test_correct_harness_passes():
    rets = _make_market()

    def run(shift):
        # Honest (shift=0): trade today on YESTERDAY's sign (roll +1).
        # Leak (shift=-1): the roll becomes 0 -> the signal IS today's return.
        sig = np.sign(np.roll(rets, 1 + shift))
        sig[:2] = 0
        sig[-2:] = 0
        return float((sig * rets).mean())

    r = lookahead_cheat_probe(run, min_improvement=0.1)
    assert r.harness_ok                       # the peek explodes -> wiring is fine
    assert r.metric_leak > 10 * abs(r.metric_base)


def test_broken_harness_is_caught():
    rets = _make_market(seed=1)

    def run(shift):
        # BUG: the shift is ignored -- the signal never actually moves.
        sig = np.sign(np.roll(rets, +1))
        return float((sig * rets).mean())

    r = lookahead_cheat_probe(run, min_improvement=0.1)
    assert not r.harness_ok
    assert any("LEAK DID NOT EXPLODE" in n for n in r.notes)


def test_latency_fragility_is_reported():
    rng = np.random.default_rng(2)
    n = 4000
    innovations = rng.normal(0, 1, n)

    def run(shift):
        # An edge that exists ONLY at zero lag: signal = today's innovation.
        sig = np.roll(innovations, shift)
        pnl = sig * innovations                # decays to ~0 at any other lag
        return float(pnl.mean())

    r = lookahead_cheat_probe(run, min_improvement=0.2)
    assert r.fragile_to_delay


# ---------------------------------------------------------------- ratios
def test_max_drawdown_hand_computed():
    # equity: 1.1 -> 0.99 ; drawdown = (1.1 - 0.99) / 1.1 = 0.1
    assert max_drawdown([0.10, -0.10]) == pytest.approx(0.1)
    # additive: +1, -3 -> equity 1, -2 ; drawdown = 3
    assert max_drawdown([1.0, -3.0], compound=False) == pytest.approx(3.0)


def test_downside_deviation_ignores_upside():
    # only the -0.2 counts: sqrt(mean([0,0,0.04])) = sqrt(0.04/3)
    assert downside_deviation([0.1, 0.3, -0.2]) == pytest.approx(np.sqrt(0.04 / 3))


def test_sortino_no_downside_is_inf_not_glory():
    assert sortino_ratio([0.01, 0.02, 0.03]) == float("inf")


def test_sortino_beats_sharpe_for_positive_skew():
    from phantomguard import sharpe_ratio
    # lumpy wins, small losses: classic positive skew
    r = np.array([-0.01] * 20 + [0.5, 0.4, 0.6])
    assert sortino_ratio(r) > sharpe_ratio(r)


def test_calmar_sanity():
    rng = np.random.default_rng(3)
    r = rng.normal(0.001, 0.01, 252)          # ~ +28%/yr, small drawdowns
    c = calmar_ratio(r, periods_per_year=252)
    assert c > 0
    # all-negative series must have negative calmar
    assert calmar_ratio([-0.01] * 50, periods_per_year=252) < 0
