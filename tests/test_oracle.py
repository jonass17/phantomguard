"""Tests for the oracle controls (power probes against phantom negatives)."""
import numpy as np
import pytest

from phantomguard.audit import audit
from phantomguard.oracle import oracle_control, oracle_probe


# ---------------------------------------------------------------------------
# oracle_control: trade lists
# ---------------------------------------------------------------------------

def test_powered_case_detects_claimed_edge():
    # 2000 independent trades, noise sd 1: an edge of 0.5/trade is easily
    # inside the audit's reach.
    rng = np.random.default_rng(0)
    v = rng.normal(0.0, 1.0, 2000)
    r = oracle_control(v, claimed_effect=0.5,
                       clusters=np.arange(2000), groups=np.arange(2000) // 100)
    assert r.powered
    assert r.detected
    assert r.injected_ci[0] > 0
    assert r.mde < 0.5
    assert r.margin > 1.0
    assert r.ci_kind == "cluster"


def test_underpowered_brutal_clustering():
    # 200 trades but only 4 independent clusters: a tiny claimed edge cannot
    # be established here no matter how real it is.
    rng = np.random.default_rng(1)
    shared = rng.normal(0.0, 1.0, 4)
    v = np.repeat(shared, 50) + rng.normal(0, 0.05, 200)
    ts = np.repeat(np.arange(4), 50)
    r = oracle_control(v, claimed_effect=0.1, clusters=ts, groups=ts)
    assert not r.powered
    assert r.mde > 0.1
    assert any("UNDERPOWERED" in n for n in r.notes)


def test_underpowered_small_n():
    rng = np.random.default_rng(2)
    v = rng.normal(0.0, 1.0, 12)
    r = oracle_control(v, claimed_effect=0.05)
    assert not r.powered
    assert r.ci_kind == "iid"
    assert r.n_clusters is None


def test_mde_monotone_sweep():
    rng = np.random.default_rng(3)
    v = rng.normal(0.0, 1.0, 500)
    r = oracle_control(v, claimed_effect=0.2, clusters=np.arange(500))
    los = [lo for _, lo, _ in r.sweep]
    dets = [det for _, _, det in r.sweep]
    # CI_lo strictly increases with the injected effect...
    assert all(b > a for a, b in zip(los, los[1:]))
    # ...so detection is monotone: once detected, always detected.
    assert dets == sorted(dets)
    # and the MDE is exactly the detection boundary of the sweep.
    for eff, _, det in r.sweep:
        assert det == (eff > r.mde)


def test_mde_consistency_with_powered():
    rng = np.random.default_rng(4)
    v = rng.normal(0.0, 1.0, 800)
    r = oracle_control(v, claimed_effect=1.0, clusters=np.arange(800))
    assert r.powered == (r.claimed_effect > r.mde)


def test_claimed_effect_must_be_positive():
    v = np.random.default_rng(5).normal(0, 1, 100)
    with pytest.raises(ValueError):
        oracle_control(v, claimed_effect=0.0)
    with pytest.raises(ValueError):
        oracle_control(v, claimed_effect=-0.5)


# ---------------------------------------------------------------------------
# audit() integration
# ---------------------------------------------------------------------------

def test_audit_attaches_oracle_and_flags_unpowered_negative():
    # Pure noise with brutal clustering: verdict is negative AND the oracle
    # says a real edge of the claimed size would have gone undetected.
    rng = np.random.default_rng(6)
    shared = rng.normal(0.0, 1.0, 5)
    v = np.repeat(shared, 40) + rng.normal(0, 0.05, 200)
    ts = np.repeat(np.arange(5), 40)
    r = audit(v, clusters=ts, groups=ts, oracle=0.1)
    assert r.verdict == "NOT ESTABLISHED"
    assert r.oracle is not None
    assert not r.oracle.powered
    assert any("UNPOWERED NEGATIVE" in f for f in r.flags)
    assert "power artifact" in str(r)


def test_audit_negative_without_oracle_hints_at_it():
    rng = np.random.default_rng(7)
    v = rng.normal(0.0, 1.0, 100)
    r = audit(v, clusters=np.arange(100), groups=np.arange(100) // 10)
    assert r.verdict == "NOT ESTABLISHED"
    assert r.oracle is None
    assert "oracle" in str(r)  # the report tells you the negative is unproven


def test_audit_powered_negative_is_a_real_finding():
    # Big clean sample, no edge: the negative verdict stands AND the oracle
    # confirms a claimed edge would have been seen -- no UNPOWERED flag.
    rng = np.random.default_rng(8)
    v = rng.normal(0.0, 1.0, 3000)
    r = audit(v, clusters=np.arange(3000), groups=np.arange(3000) // 150,
              oracle=0.5)
    assert r.verdict == "NOT ESTABLISHED"
    assert r.oracle is not None and r.oracle.powered
    assert not any("UNPOWERED NEGATIVE" in f for f in r.flags)


# ---------------------------------------------------------------------------
# oracle_probe: model pipelines
# ---------------------------------------------------------------------------

def _make_binary_problem(n=400, k=3, base_rate=0.3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, k))
    y = (rng.random(n) < base_rate).astype(float)
    return X, y


def test_oracle_probe_powered_pipeline():
    # A pipeline that actually uses its features scores AUC 1.0 when the
    # target itself is appended as the last column.
    X, y = _make_binary_problem()
    r = oracle_probe(lambda X_, y_: X_[:, -1], X, y)
    assert r.metric == "auc"
    assert r.score_oracle == pytest.approx(1.0)
    assert r.powered
    # the honest baseline (last real feature) is near chance
    assert r.score_baseline is not None and r.score_baseline < 0.6


def test_oracle_probe_underpowered_pipeline():
    # A pipeline that ignores its inputs (the min_child_samples failure mode:
    # the model cannot split, so every score is the prior) must FAIL the probe.
    X, y = _make_binary_problem(seed=1)
    r = oracle_probe(lambda X_, y_: np.zeros(len(X_)), X, y)
    assert not r.powered
    assert r.score_oracle == pytest.approx(0.5)
    assert any("CANNOT SEE ITS OWN TARGET" in n for n in r.notes)
    assert any("CONSTANT" in n for n in r.notes)


def test_oracle_probe_rare_target_warning():
    X, y = _make_binary_problem(n=1000, base_rate=0.0, seed=2)
    y[:15] = 1.0  # 15 positives, 1.5% base rate
    r = oracle_probe(lambda X_, y_: X_[:, -1], X, y)
    assert r.n_positive == 15
    assert any("rare target" in n for n in r.notes)


def test_oracle_probe_continuous_target_uses_spearman():
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (200, 2))
    y = rng.normal(0, 1, 200)
    r = oracle_probe(lambda X_, y_: X_[:, -1], X, y)
    assert r.metric == "spearman"
    assert r.score_oracle == pytest.approx(1.0)
    assert r.powered


def test_oracle_probe_validates_shapes():
    X, y = _make_binary_problem()
    with pytest.raises(ValueError):
        oracle_probe(lambda X_, y_: X_[:, -1], X, y[:-1])
    with pytest.raises(ValueError):
        oracle_probe(lambda X_, y_: X_[:5, -1], X, y)  # wrong score length
