"""Tests for decay_check and cost_ladder -- known-truth constructions."""
import numpy as np
import pytest

from phantomguard.decay import cost_ladder, decay_check


# ---------------------------------------------------------------- decay
def test_stable_edge_not_flagged():
    rng = np.random.default_rng(1)
    v = rng.normal(0.3, 1.0, 400)          # same edge early and late
    r = decay_check(v)
    assert not r.decaying
    assert r.mean_late > 0


def test_dying_edge_is_caught():
    rng = np.random.default_rng(2)
    early = rng.normal(0.5, 0.5, 200)       # golden past...
    late = rng.normal(-0.1, 0.5, 200)       # ...dead present
    r = decay_check(np.concatenate([early, late]))
    assert r.decaying
    assert any("DECAY" in n for n in r.notes)
    assert r.mean_all > 0                    # headline still looks fine!


def test_order_labels_sort_before_split():
    rng = np.random.default_rng(3)
    early = rng.normal(0.5, 0.5, 100)
    late = rng.normal(-0.2, 0.5, 100)
    v = np.concatenate([late, early])        # deliberately shuffled halves
    order = np.concatenate([np.arange(100, 200), np.arange(100)])
    r = decay_check(v, order=order)          # order labels restore chronology
    assert r.decaying
    assert r.mean_early > r.mean_late


def test_decay_check_needs_enough_data():
    with pytest.raises(ValueError):
        decay_check([1.0] * 5)


# ---------------------------------------------------------------- cost ladder
def test_breakeven_is_the_mean():
    v = np.array([2.0, 4.0, 6.0])            # mean 4 -> break-even cost 4
    r = cost_ladder(v, costs=[0, 1])
    assert r.breakeven_cost == pytest.approx(4.0)


def test_edge_dying_on_the_ladder_is_flagged():
    rng = np.random.default_rng(4)
    v = rng.normal(1.0, 1.0, 2000)           # strong edge, breakeven ~1.0
    r = cost_ladder(v, costs=[0, 0.5, 2.0])
    lo0 = r.ladder[0]["ci_lo"]
    assert lo0 > 0                            # significant at zero cost
    assert r.ladder[-1]["mean"] < 0           # dead at cost 2
    assert any("survives" in n for n in r.notes)


def test_cluster_ladder_uses_cluster_ci():
    rng = np.random.default_rng(5)
    base = rng.normal(0.5, 1.0, 100)
    v = np.repeat(base, 5) + rng.normal(0, 0.01, 500)
    clusters = np.repeat(np.arange(100), 5)
    r_iid = cost_ladder(v, costs=[0], n_boot=4000)
    r_clu = cost_ladder(v, costs=[0], clusters=clusters, n_boot=4000)
    width_iid = r_iid.ladder[0]["ci_hi"] - r_iid.ladder[0]["ci_lo"]
    width_clu = r_clu.ladder[0]["ci_hi"] - r_clu.ladder[0]["ci_lo"]
    assert width_clu > width_iid * 1.5        # duplication must widen the honest CI
