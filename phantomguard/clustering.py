"""Cluster-aware bootstrap: honest confidence intervals for dependent trades.

The IID bootstrap assumes every observation is an independent draw. Trading
data routinely violates this: when one signal fires across 5 correlated assets
at the same timestamp, those 5 trades are one piece of information wearing
five costumes -- they win or lose together. Resampling them independently
produces a confidence interval that is too narrow, which turns noise into
"significance". This is one of the most common ways dead strategies get
deployed.

The fix is the cluster (block) bootstrap: resample whole clusters -- e.g. all
trades sharing an entry timestamp -- so that whatever happened together stays
together. The honest effective sample size is closer to the number of
clusters than the number of trades.

References
----------
Cameron, Gelbach & Miller (2008), "Bootstrap-Based Improvements for Inference
    with Clustered Errors", Review of Economics and Statistics.
Efron & Tibshirani (1993), "An Introduction to the Bootstrap", ch. 8
    (the failure of the IID bootstrap under dependence).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# If the cluster CI is this much wider than the IID CI, the IID interval is
# materially anti-conservative and should not be trusted.
WIDENING_WARN = 1.15


def _validate(values, clusters):
    v = np.asarray(values, dtype=float).ravel()
    c = np.asarray(clusters).ravel()
    if v.size != c.size:
        raise ValueError(f"values ({v.size}) and clusters ({c.size}) must have the same length")
    mask = np.isfinite(v)
    v, c = v[mask], c[mask]
    if v.size < 2:
        raise ValueError("need at least 2 finite observations")
    return v, c


def effective_clusters(clusters) -> int:
    """Number of distinct clusters -- the honest upper bound on independent
    information in the sample."""
    return int(np.unique(np.asarray(clusters).ravel()).size)


def cluster_bootstrap_ci(values, clusters, n_boot: int = 10000,
                         alpha: float = 0.05, seed: int = 0):
    """Cluster bootstrap CI for the mean of ``values``.

    Clusters (e.g. entry timestamps) are resampled with replacement as whole
    units. Returns ``(point, lo, hi)``.

    Notes
    -----
    Resampling ``n_clusters`` clusters per replicate keeps the amount of
    *independent* information per replicate constant, which is the quantity
    that actually drives the width of an honest CI.
    """
    v, c = _validate(values, clusters)
    rng = np.random.default_rng(seed)

    # Group observation values by cluster once, up front.
    uniq, inv = np.unique(c, return_inverse=True)
    k = uniq.size
    groups = [v[inv == i] for i in range(k)]
    sums = np.array([g.sum() for g in groups])
    counts = np.array([g.size for g in groups], dtype=float)

    # Each replicate: draw k clusters with replacement; the replicate mean is
    # (sum of drawn cluster sums) / (sum of drawn cluster sizes).
    draw = rng.integers(0, k, size=(n_boot, k))
    boot_means = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)

    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(v.mean()), lo, hi


def iid_bootstrap_ci(values, n_boot: int = 10000, alpha: float = 0.05,
                     seed: int = 0):
    """Plain IID bootstrap CI for the mean -- shown for comparison only.
    Use ``cluster_bootstrap_ci`` whenever observations can co-fire."""
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size < 2:
        raise ValueError("need at least 2 finite observations")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boot_means = v[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(v.mean()), lo, hi


@dataclass
class ClusterDiagnosis:
    """Side-by-side IID vs cluster bootstrap, with a verdict."""
    n_obs: int
    n_clusters: int
    max_cluster_size: int
    point: float
    iid_ci: tuple[float, float]
    cluster_ci: tuple[float, float]
    widening: float               # cluster CI width / IID CI width
    anti_conservative: bool       # True -> the IID CI should not be trusted
    notes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "ClusterDiagnosis",
            f"  observations      : {self.n_obs}",
            f"  distinct clusters : {self.n_clusters}"
            f"  (max cluster size {self.max_cluster_size})",
            f"  point estimate    : {self.point:+.6g}",
            f"  IID bootstrap CI  : [{self.iid_ci[0]:+.6g}, {self.iid_ci[1]:+.6g}]",
            f"  cluster CI        : [{self.cluster_ci[0]:+.6g}, {self.cluster_ci[1]:+.6g}]",
            f"  widening factor   : {self.widening:.2f}x",
        ]
        lines += [f"  ! {n}" for n in self.notes]
        return "\n".join(lines)


def diagnose_clustering(values, clusters, n_boot: int = 10000,
                        alpha: float = 0.05, seed: int = 0) -> ClusterDiagnosis:
    """Compare the naive IID CI with the cluster-honest CI and flag the gap.

    Parameters
    ----------
    values : array-like
        Per-trade PnL (or returns).
    clusters : array-like
        Cluster label per trade -- typically the entry timestamp. Trades that
        share a label are treated as one unit of information.

    The verdict to act on: if ``anti_conservative`` is True, any significance
    claim based on the IID interval is inflated; use the cluster interval.
    """
    v, c = _validate(values, clusters)
    uniq, counts = np.unique(c, return_counts=True)

    point, ilo, ihi = iid_bootstrap_ci(v, n_boot=n_boot, alpha=alpha, seed=seed)
    _, clo, chi = cluster_bootstrap_ci(v, c, n_boot=n_boot, alpha=alpha, seed=seed)

    iid_width = ihi - ilo
    widening = float((chi - clo) / iid_width) if iid_width > 0 else float("inf")

    notes = []
    dup_ratio = uniq.size / v.size
    if dup_ratio < 0.9:
        notes.append(
            f"only {uniq.size}/{v.size} independent timestamps -- "
            f"effective n is much smaller than the trade count"
        )
    anti = widening > WIDENING_WARN
    if anti:
        notes.append(
            f"IID CI is anti-conservative (cluster CI {widening:.2f}x wider); "
            f"do not base a significance claim on the IID interval"
        )
    if ilo > 0 >= clo:
        notes.append(
            "SIGNIFICANCE FLIP: IID CI excludes 0 but the cluster CI does not -- "
            "this 'edge' may be an artifact of dependent trades"
        )

    return ClusterDiagnosis(
        n_obs=int(v.size),
        n_clusters=int(uniq.size),
        max_cluster_size=int(counts.max()),
        point=point,
        iid_ci=(ilo, ihi),
        cluster_ci=(clo, chi),
        widening=widening,
        anti_conservative=anti,
        notes=notes,
    )
