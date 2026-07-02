"""One-call audit: run every applicable check over a trade list.

``audit()`` is the front door: hand it per-trade PnL plus whatever labels you
have (entry timestamps, days/markets), and it runs the full battery --
bootstrap significance (cluster-aware when timestamps are given),
concentration, and risk ratios -- then aggregates every red flag into one
verdict.

The verdict vocabulary is deliberately asymmetric:

- ``NOT ESTABLISHED`` -- at least one finding kills the significance claim.
- ``SUSPECT``         -- significant on its face, but structural warnings
                          remain; do not deploy on this evidence alone.
- ``NO RED FLAGS``    -- nothing here caught it. This is NOT "the edge is
                          real": an audit can only ever find guilt, never
                          prove innocence. Out-of-sample and forward tests
                          remain your burden.
"""
from __future__ import annotations

import csv as _csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .clustering import diagnose_clustering, iid_bootstrap_ci, ClusterDiagnosis
from .concentration import concentration_check, ConcentrationResult
from .decay import decay_check, DecayResult
from .ratios import max_drawdown, sortino_ratio


@dataclass
class AuditReport:
    """Aggregated result of the full check battery."""
    n_obs: int
    mean: float
    ci: tuple[float, float]        # the HONEST interval (cluster CI if available)
    ci_kind: str                   # "cluster" or "iid"
    significant: bool              # honest CI excludes zero
    clustering: ClusterDiagnosis | None
    concentration: ConcentrationResult | None
    decay: DecayResult | None
    sortino: float
    max_drawdown_additive: float
    flags: list[str] = field(default_factory=list)
    verdict: str = ""

    def __str__(self) -> str:
        lines = [
            f"PhantomGuard audit -- verdict: {self.verdict}",
            "",
            f"  trades            : {self.n_obs}",
            f"  mean PnL          : {self.mean:+.6g}",
            f"  {self.ci_kind} 95% CI     : [{self.ci[0]:+.6g}, {self.ci[1]:+.6g}]"
            f"  -> {'excludes 0' if self.significant else 'includes 0'}",
            f"  sortino (per-obs) : {self.sortino:+.3g}",
            f"  max drawdown      : {self.max_drawdown_additive:.6g} (additive)",
        ]
        if self.clustering is not None:
            lines.append(
                f"  clusters          : {self.clustering.n_clusters}/{self.n_obs}"
                f" independent timestamps (CI widening {self.clustering.widening:.2f}x)"
            )
        if self.concentration is not None:
            c = self.concentration
            share = (f"{c.best_group_share:.0%} of PnL"
                     if np.isfinite(c.best_group_share) else "n/a (total PnL <= 0)")
            lines.append(
                f"  concentration     : best group {c.best_group!r} carries"
                f" {share}; ex-best mean {c.mean_ex_best:+.4g}"
            )
        if self.decay is not None:
            lines.append(
                f"  decay             : early {self.decay.mean_early:+.4g} -> "
                f"late {self.decay.mean_late:+.4g} (trend rho {self.decay.spearman_rho:+.2f})"
            )
        if self.flags:
            lines.append("")
            lines.append("  red flags:")
            lines += [f"    ! {f}" for f in self.flags]
        else:
            lines.append("")
            lines.append("  no red flags found (this does not prove the edge is real --")
            lines.append("  it only means this battery could not kill it)")
        return "\n".join(lines)


def audit(values, clusters=None, groups=None, n_boot: int = 10000,
          alpha: float = 0.05, seed: int = 0) -> AuditReport:
    """Run every applicable check over per-trade PnL.

    Parameters
    ----------
    values : array-like
        Per-trade PnL (or returns).
    clusters : array-like, optional
        Entry timestamp (or any co-firing label) per trade. Enables the
        cluster bootstrap; the HONEST interval is then the cluster CI.
    groups : array-like, optional
        Day / market / event label per trade. Enables the concentration
        check. Passing calendar days is the recommended minimum.

    Give it everything you have: every omitted label silently disables the
    check that could have caught your phantom.
    """
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size < 2:
        raise ValueError("need at least 2 finite observations")

    flags: list[str] = []

    # --- honest significance -------------------------------------------
    clu = None
    if clusters is not None:
        clu = diagnose_clustering(values, clusters, n_boot=n_boot, alpha=alpha, seed=seed)
        ci, kind = clu.cluster_ci, "cluster"
        flags += clu.notes
    else:
        _, lo, hi = iid_bootstrap_ci(v, n_boot=n_boot, alpha=alpha, seed=seed)
        ci, kind = (lo, hi), "iid"
        flags.append(
            "no cluster labels supplied -- if any trades can co-fire "
            "(same signal, several assets), this CI is too narrow"
        )
    significant = ci[0] > 0 or ci[1] < 0

    # --- concentration ---------------------------------------------------
    conc = None
    if groups is not None:
        distinct = len(set(np.asarray(groups).ravel().tolist()))
        if distinct >= 2:
            conc = concentration_check(values, groups)
            flags += conc.notes
        else:
            flags.append(
                "all trades fall into a single group (e.g. one day) -- "
                "concentration is untestable; the sample cannot establish "
                "an edge across regimes"
            )
    else:
        flags.append(
            "no group labels (days/markets) supplied -- one-day-wonder "
            "concentration cannot be checked"
        )

    # --- decay -----------------------------------------------------------
    dec = None
    if v.size >= 10:
        # clusters (entry timestamps) double as the chronological order; without
        # them we assume the caller passed trades oldest-first.
        dec = decay_check(values if clusters is None else values,
                          order=clusters if clusters is not None else None)
        flags += dec.notes
        if clusters is None:
            flags.append(
                "decay check assumed chronological input order (no timestamps given)"
            )

    # --- verdict ---------------------------------------------------------
    mean = float(v.mean())
    killed = (not significant) or mean <= 0 or (conc is not None and conc.sign_flip)
    if not significant:
        flags.append("honest CI includes zero -- not distinguishable from noise")
    if mean <= 0:
        flags.append("mean PnL is not positive -- there is no edge claim to establish")
    if killed:
        verdict = "NOT ESTABLISHED"
    elif flags:
        verdict = "SUSPECT"
    else:
        verdict = "NO RED FLAGS"

    return AuditReport(
        n_obs=int(v.size),
        mean=float(v.mean()),
        ci=(float(ci[0]), float(ci[1])),
        ci_kind=kind,
        significant=bool(significant),
        clustering=clu,
        concentration=conc,
        decay=dec,
        sortino=float(sortino_ratio(v)),
        max_drawdown_additive=float(max_drawdown(v, compound=False)),
        flags=flags,
        verdict=verdict,
    )


def audit_csv(path, value_col: str, cluster_col: str | None = None,
              group_col: str | None = None, **kwargs) -> AuditReport:
    """``audit()`` straight from a CSV of trades.

    ``value_col`` names the PnL column; ``cluster_col`` (entry timestamp)
    and ``group_col`` (day/market) are optional but strongly recommended.
    """
    path = Path(path)
    values, clusters, groups = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        if reader.fieldnames is None or value_col not in reader.fieldnames:
            raise ValueError(f"column {value_col!r} not found in {path.name} "
                             f"(has: {reader.fieldnames})")
        for row in reader:
            try:
                values.append(float(row[value_col]))
            except (TypeError, ValueError):
                continue  # skip unparseable rows rather than invent zeros
            if cluster_col is not None:
                clusters.append(row[cluster_col])
            if group_col is not None:
                groups.append(row[group_col])
    return audit(
        values,
        clusters=clusters if cluster_col is not None else None,
        groups=groups if group_col is not None else None,
        **kwargs,
    )
