"""Tests for the one-call audit()."""
import numpy as np
import pytest

from phantomguard.audit import audit, audit_csv


def test_phantom_is_killed():
    # Co-firing trades with zero true edge: audit must say NOT ESTABLISHED.
    rng = np.random.default_rng(1)
    shared = rng.normal(0.0, 1.0, 120)
    v = np.repeat(shared, 5) + rng.normal(0, 0.05, 600)
    ts = np.repeat(np.arange(120), 5)
    days = ts // 24
    r = audit(v, clusters=ts, groups=days)
    assert r.verdict == "NOT ESTABLISHED"
    assert r.ci_kind == "cluster"
    assert not r.significant


def test_broad_real_edge_at_most_suspect():
    # Independent trades, genuine strong edge, spread across days.
    rng = np.random.default_rng(2)
    v = rng.normal(0.5, 1.0, 800)
    ts = np.arange(800)              # every trade its own timestamp
    days = ts // 40
    r = audit(v, clusters=ts, groups=days)
    assert r.significant
    assert r.verdict in ("NO RED FLAGS", "SUSPECT")
    assert r.verdict != "NOT ESTABLISHED"


def test_missing_labels_are_flagged():
    rng = np.random.default_rng(3)
    r = audit(rng.normal(0.5, 1.0, 300))
    assert any("no cluster labels" in f for f in r.flags)
    assert any("no group labels" in f for f in r.flags)
    assert r.verdict in ("SUSPECT", "NOT ESTABLISHED")  # never a clean bill


def test_one_day_wonder_killed_even_if_significant():
    rng = np.random.default_rng(4)
    days = np.repeat(np.arange(10), 30)
    v = rng.normal(-0.05, 0.3, 300)
    v[days == 3] += 2.0              # windfall day makes the headline positive
    ts = np.arange(300)
    r = audit(v, clusters=ts, groups=days)
    if r.concentration.sign_flip:
        assert r.verdict == "NOT ESTABLISHED"


def test_audit_csv_roundtrip(tmp_path):
    p = tmp_path / "trades.csv"
    p.write_text(
        "pnl,entry_ts,day\n"
        "1.0,t1,d1\n-0.5,t1,d1\n0.8,t2,d1\n-0.2,t3,d2\n0.4,t3,d2\nbad,t4,d2\n",
        encoding="utf-8",
    )
    r = audit_csv(p, "pnl", cluster_col="entry_ts", group_col="day")
    assert r.n_obs == 5              # the unparseable row is skipped, not zeroed
    assert r.ci_kind == "cluster"


def test_audit_csv_missing_column(tmp_path):
    p = tmp_path / "trades.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        audit_csv(p, "pnl")


def test_single_group_does_not_crash():
    rng = np.random.default_rng(6)
    v = rng.normal(0.1, 1.0, 50)
    r = audit(v, clusters=np.arange(50), groups=["d1"] * 50)
    assert r.concentration is None
    assert any("single group" in f for f in r.flags)
