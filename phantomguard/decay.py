"""Decay check: is the edge already dying inside your own sample?

The most common life cycle of a real-but-doomed edge: it existed, others
found it too, and it is fading -- so the *early* part of your backtest is
profitable, the *late* part is flat or negative, and the average still looks
fine. You then deploy into the corpse. The only part of the sample that
forecasts tomorrow is the recent end.

Two complementary measurements, both order-based (no timestamps needed --
pass trades in chronological order, or pass sortable time labels):

- early/late split: mean PnL of the first half vs the second half.
- trend: Spearman rank correlation between time order and PnL. Rank-based,
  so a few fat outliers cannot fake or hide a drift.

As with every check in this library, a decay flag does not mean "tune it
away" -- it means the sample does not establish a *currently live* edge.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats as _ss


@dataclass
class DecayResult:
    """Early-vs-late comparison and time trend of per-trade PnL."""
    n_obs: int
    mean_all: float
    mean_early: float           # first half (chronological)
    mean_late: float            # second half -- the part that forecasts tomorrow
    late_share_positive: bool
    spearman_rho: float         # PnL vs time order; negative = fading
    spearman_p: float
    decaying: bool
    notes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "DecayResult",
            f"  observations : {self.n_obs}",
            f"  mean early   : {self.mean_early:+.6g}   (first half)",
            f"  mean late    : {self.mean_late:+.6g}   (second half -- what forecasts tomorrow)",
            f"  time trend   : rho {self.spearman_rho:+.3f} (p={self.spearman_p:.3f})",
        ]
        lines += [f"  ! {n}" for n in self.notes]
        return "\n".join(lines)


def decay_check(values, order=None) -> DecayResult:
    """Check whether the edge fades from the early to the late sample.

    Parameters
    ----------
    values : array-like
        Per-trade PnL in CHRONOLOGICAL order (oldest first), unless ``order``
        is given.
    order : array-like, optional
        Sortable time labels (timestamps, dates, sequence numbers). When
        given, trades are sorted by it first.

    Verdict logic (deliberately simple and pre-stated):
    ``decaying`` is True when the late-half mean is not positive while the
    full-sample mean is, OR the time trend is significantly negative
    (rho < 0, p < 0.05).
    """
    v = np.asarray(values, dtype=float).ravel()
    if order is not None:
        o = np.asarray(order).ravel()
        if o.size != v.size:
            raise ValueError(f"values ({v.size}) and order ({o.size}) must have the same length")
        v = v[np.argsort(o, kind="stable")]
    v = v[np.isfinite(v)]
    if v.size < 10:
        raise ValueError("need at least 10 observations for a meaningful decay check")

    half = v.size // 2
    early, late = v[:half], v[half:]
    rho, p = _ss.spearmanr(np.arange(v.size), v)

    notes = []
    late_positive = float(late.mean()) > 0
    if not late_positive and float(v.mean()) > 0:
        notes.append(
            f"DECAY: late half mean {late.mean():+.4g} is not positive while the "
            f"full-sample mean {v.mean():+.4g} is -- the profitable part of this "
            f"sample is the past, not the part that forecasts tomorrow"
        )
    trend_negative = rho < 0 and p < 0.05
    if trend_negative:
        notes.append(
            f"negative time trend (rho {rho:+.3f}, p={p:.3f}) -- "
            f"PnL per trade is drifting down inside the sample"
        )

    return DecayResult(
        n_obs=int(v.size),
        mean_all=float(v.mean()),
        mean_early=float(early.mean()),
        mean_late=float(late.mean()),
        late_share_positive=late_positive,
        spearman_rho=float(rho),
        spearman_p=float(p),
        decaying=bool((not late_positive and float(v.mean()) > 0) or trend_negative),
        notes=notes,
    )


@dataclass
class CostLadderResult:
    """Edge survival under increasing per-trade costs."""
    n_obs: int
    ladder: list[dict]          # per level: cost, mean, ci_lo, ci_hi
    breakeven_cost: float       # cost at which the mean hits zero
    notes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["CostLadderResult"]
        for step in self.ladder:
            sig = "CI_lo>0" if step["ci_lo"] > 0 else "includes 0" if step["ci_hi"] > 0 else "CI_hi<0"
            lines.append(
                f"  cost {step['cost']:>8.4g} : mean {step['mean']:+.6g}"
                f"  [{step['ci_lo']:+.6g}, {step['ci_hi']:+.6g}]  ({sig})"
            )
        lines.append(f"  break-even cost : {self.breakeven_cost:.6g} per trade")
        lines += [f"  ! {n}" for n in self.notes]
        return "\n".join(lines)


def cost_ladder(values, costs, clusters=None, n_boot: int = 10000,
                alpha: float = 0.05, seed: int = 0) -> CostLadderResult:
    """Recompute the mean and its CI under increasing per-trade costs.

    Parameters
    ----------
    values : array-like
        Per-trade PnL BEFORE the costs you want to stress (same unit as costs).
    costs : sequence of float
        Cost levels to subtract per trade, e.g. ``[0, 0.5, 1.0, 2.0]`` cents.
    clusters : array-like, optional
        Entry timestamps -- uses the honest cluster bootstrap when given.

    The single most useful line of the output is the break-even cost: compare
    it with your REAL spread+slippage+fees. An edge whose break-even sits
    below your real costs is dead no matter how pretty the raw mean looks.
    """
    from .clustering import cluster_bootstrap_ci, iid_bootstrap_ci

    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size < 2:
        raise ValueError("need at least 2 finite observations")
    costs = sorted(float(c) for c in costs)
    if not costs:
        raise ValueError("need at least one cost level")

    ladder = []
    for c in costs:
        shifted = v - c
        if clusters is not None:
            mean, lo, hi = cluster_bootstrap_ci(shifted, clusters, n_boot=n_boot,
                                                alpha=alpha, seed=seed)
        else:
            mean, lo, hi = iid_bootstrap_ci(shifted, n_boot=n_boot, alpha=alpha, seed=seed)
        ladder.append({"cost": c, "mean": mean, "ci_lo": lo, "ci_hi": hi})

    breakeven = float(v.mean())  # mean PnL per trade IS the cost that zeroes it
    notes = []
    surviving = [s for s in ladder if s["ci_lo"] > 0]
    if not surviving:
        notes.append(
            "no cost level has CI_lo > 0 -- the edge is not established even "
            "before realistic costs"
        )
    elif surviving[-1]["cost"] < costs[-1]:
        notes.append(
            f"edge survives (CI_lo>0) only up to cost {surviving[-1]['cost']:g} -- "
            f"compare with your REAL spread+slippage+fees before believing it"
        )
    return CostLadderResult(
        n_obs=int(v.size),
        ladder=ladder,
        breakeven_cost=breakeven,
        notes=notes,
    )
