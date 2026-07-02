"""Tests for the cluster-aware bootstrap.

The two key behaviours, verified against constructions where the truth is
known by design:
1. When every observation is its own cluster, the cluster CI must agree with
   the IID CI (clustering machinery adds nothing when there is nothing to add).
2. When observations are near-duplicates within clusters, the IID CI must be
   exposed as too narrow (widening factor >> 1) and the significance flip
   must be caught.
"""
import numpy as np
import pytest

from phantomguard.clustering import (
    cluster_bootstrap_ci,
    diagnose_clustering,
    effective_clusters,
    iid_bootstrap_ci,
)


def test_independent_data_cluster_ci_matches_iid():
    rng = np.random.default_rng(1)
    v = rng.normal(0.0, 1.0, 400)
    clusters = np.arange(400)  # every trade its own timestamp
    _, ilo, ihi = iid_bootstrap_ci(v, seed=2)
    _, clo, chi = cluster_bootstrap_ci(v, clusters, seed=2)
    # Same information -> intervals agree to within bootstrap noise.
    assert abs(clo - ilo) < 0.05
    assert abs(chi - ihi) < 0.05
    d = diagnose_clustering(v, clusters, seed=2)
    assert not d.anti_conservative
    assert d.n_clusters == 400


def test_duplicated_trades_widen_ci_and_flag():
    # 80 timestamps, each firing 5 near-identical trades (5 assets co-firing).
    rng = np.random.default_rng(3)
    base = rng.normal(0.3, 1.0, 80)
    v = np.repeat(base, 5) + rng.normal(0, 0.01, 400)  # 5 costumes, 1 information
    clusters = np.repeat(np.arange(80), 5)

    d = diagnose_clustering(v, clusters, seed=4)
    assert d.n_obs == 400
    assert d.n_clusters == 80
    assert d.max_cluster_size == 5
    # Duplicating each draw 5x should widen the honest CI by roughly sqrt(5).
    assert d.widening > 1.6
    assert d.anti_conservative
    assert any("anti-conservative" in n for n in d.notes)


def test_significance_flip_is_caught():
    # Weak true effect + heavy duplication: IID says "significant",
    # clusters say "not so fast".
    rng = np.random.default_rng(5)
    base = rng.normal(0.22, 1.0, 60)
    v = np.repeat(base, 6) + rng.normal(0, 0.01, 360)
    clusters = np.repeat(np.arange(60), 6)
    d = diagnose_clustering(v, clusters, seed=6)
    _, ilo, _ = iid_bootstrap_ci(v, seed=6)
    if ilo > 0 and d.cluster_ci[0] <= 0:  # construction achieved the flip
        assert any("SIGNIFICANCE FLIP" in n for n in d.notes)
    # Regardless of the flip, the widening itself must be flagged.
    assert d.anti_conservative


def test_seed_reproducibility():
    rng = np.random.default_rng(7)
    v = rng.normal(0, 1, 100)
    clusters = np.repeat(np.arange(25), 4)
    a = cluster_bootstrap_ci(v, clusters, seed=42)
    b = cluster_bootstrap_ci(v, clusters, seed=42)
    assert a == b


def test_effective_clusters():
    assert effective_clusters([1, 1, 2, 3, 3, 3]) == 3


def test_validation_errors():
    with pytest.raises(ValueError):
        cluster_bootstrap_ci([1.0, 2.0], [1])          # length mismatch
    with pytest.raises(ValueError):
        cluster_bootstrap_ci([np.nan, np.inf], [1, 2])  # no finite data
