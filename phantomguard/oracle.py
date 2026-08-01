"""Oracle controls: prove that your setup COULD have found the edge before
you believe it saying there is none.

Every other check in this library guards against the phantom POSITIVE: a
result that looks like an edge but is noise, leakage or concentration. This
module guards against the opposite failure, the phantom NEGATIVE: a pipeline
so weak that it could not have detected a real edge -- so its "nothing here"
is a statement about the pipeline, not about the market.

The two failure modes need two different controls:

- ``lookahead_cheat_probe`` (in :mod:`phantomguard.probes`) wires a crime
  into the pipeline and demands an explosion -> catches broken wiring that
  would FAKE a positive.
- The oracle controls here inject the ANSWER into the pipeline and demand
  near-perfect detection -> catch a setup too weak to produce a real
  negative. A pipeline that cannot see its own target cannot see anything.

Both controls are cheap, and a negative verdict without the second one is
not a finding. The classic real-world failure: a tree model with
``min_child_samples=50`` scored on a target with 30 positives reached only
AUC 0.57 *with the target itself as a feature* -- every negative that setup
ever produced was void, and the leakage probe alone would never have said so.

Two entry points:

- ``oracle_control(pnl, claimed_effect, ...)`` -- for trade lists / the
  ``audit()`` path: inject a synthetic edge of the claimed size and check
  that the (cluster-honest) audit detects it; also report the minimal
  detectable effect (MDE) at this n / clustering.
- ``oracle_probe(fit_score, X, y)`` -- for model pipelines: append the
  target itself as a feature and demand a near-perfect score.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
from scipy import stats as _ss


# ---------------------------------------------------------------------------
# A) trade lists: synthetic-edge injection + minimal detectable effect
# ---------------------------------------------------------------------------

@dataclass
class OracleControlResult:
    """Power verdict for a trade-list audit: could it detect the claimed edge?"""
    n_obs: int
    n_clusters: Optional[int]      # None when no cluster labels were given
    ci_kind: str                   # "cluster" or "iid" -- same path audit() uses
    claimed_effect: float          # mean PnL/trade a real edge would produce
    injected_ci: tuple[float, float]   # honest CI after injecting claimed_effect
    detected: bool                 # injected CI_lo > 0 -- the audit saw the edge
    powered: bool                  # == detected: the negative is interpretable
    mde: float                     # minimal detectable effect at this n/clustering
    margin: float                  # claimed_effect / mde (power headroom)
    injected_verdict: str          # full audit() verdict on the injected copy
    sweep: list[tuple[float, float, bool]] = field(default_factory=list)
    #      (effect, ci_lo at that effect, detected) -- monotone by construction
    notes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "OracleControl (synthetic-edge power probe)",
            f"  observations      : {self.n_obs}"
            + (f"  ({self.n_clusters} clusters)" if self.n_clusters is not None else ""),
            f"  claimed effect    : {self.claimed_effect:+.6g} per trade",
            f"  injected {self.ci_kind} CI : [{self.injected_ci[0]:+.6g}, "
            f"{self.injected_ci[1]:+.6g}]  -> "
            f"{'DETECTED' if self.detected else 'NOT DETECTED'}",
            f"  MDE (this sample) : {self.mde:.6g} per trade",
            f"  power margin      : {self.margin:.2f}x"
            if np.isfinite(self.margin) else "  power margin      : inf",
            "  effect sweep      :",
        ]
        for eff, lo, det in self.sweep:
            lines.append(f"      {eff:>10.4g} -> CI_lo {lo:+.4g}  "
                         f"{'detected' if det else 'missed'}")
        lines.append(f"  powered           : {self.powered}")
        lines += [f"  ! {n}" for n in self.notes]
        return "\n".join(lines)


def oracle_control(values, claimed_effect: float, clusters=None, groups=None,
                   effect_grid: Optional[Sequence[float]] = None,
                   n_boot: int = 10000, alpha: float = 0.05,
                   seed: int = 0) -> OracleControlResult:
    """Inject a synthetic edge of the claimed size and ask: would ``audit()``
    have caught it?

    Takes the same inputs as :func:`phantomguard.audit` plus
    ``claimed_effect`` -- the mean PnL per trade that a REAL edge of the
    claimed size would produce. The sample is centered (its own mean
    removed), the claimed effect is added on top, and the exact same honest
    audit path (cluster bootstrap when timestamps are given) runs on the
    copy. Detection means the honest CI lower bound clears zero.

    ``mde`` is the minimal detectable effect: the smallest mean shift that
    this sample -- with THIS n, THIS noise and THIS clustering -- would flag
    as significant. Because a constant shift moves every bootstrap replicate
    by exactly that constant (same seed, same draws), the MDE is exact, not
    a grid approximation; the ``sweep`` is reported for legibility.

    The verdict to act on: ``powered=False`` means a true edge of the
    claimed size would have gone UNDETECTED here -- any negative verdict
    from this sample is a power artifact, not a finding.
    """
    from .audit import audit  # lazy -- audit() and oracle_control() call each other

    claimed_effect = float(claimed_effect)
    if not np.isfinite(claimed_effect) or claimed_effect <= 0:
        raise ValueError(
            "claimed_effect must be a positive finite mean PnL per trade "
            "(the size a real edge would have; pass its absolute value)")

    v = np.asarray(values, dtype=float).ravel()
    mask = np.isfinite(v)
    v = v[mask]
    if v.size < 2:
        raise ValueError("need at least 2 finite observations")
    c = np.asarray(clusters).ravel()[mask] if clusters is not None else None
    g = np.asarray(groups).ravel()[mask] if groups is not None else None

    # Center, then shift: the injected sample keeps the real noise, tails and
    # clustering structure, but has a TRUE mean of exactly claimed_effect.
    injected = v - v.mean() + claimed_effect
    rep = audit(injected, clusters=c, groups=g,
                n_boot=n_boot, alpha=alpha, seed=seed)

    detected = rep.ci[0] > 0
    # Shift linearity: CI_lo(effect) = CI_lo(centered null) + effect exactly.
    null_lo = rep.ci[0] - claimed_effect
    mde = max(0.0, -null_lo)
    margin = claimed_effect / mde if mde > 0 else float("inf")

    if effect_grid is None:
        effect_grid = [f * claimed_effect
                       for f in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0)]
    sweep = []
    for eff in sorted(float(e) for e in effect_grid):
        lo = null_lo + eff
        sweep.append((eff, lo, lo > 0))

    notes: list[str] = []
    if not detected:
        notes.append(
            f"UNDERPOWERED: an injected TRUE edge of {claimed_effect:+.4g}/trade "
            f"does not reach CI_lo > 0 at this n/clustering (MDE {mde:.4g}) -- "
            f"a negative verdict from this sample is a power artifact, not a finding"
        )
    elif np.isfinite(margin) and margin < 1.5:
        notes.append(
            f"thin power margin: the claimed effect is only {margin:.2f}x the "
            f"MDE -- detection is borderline, treat a negative with care"
        )
    if detected and rep.verdict == "NOT ESTABLISHED":
        notes.append(
            "the CI detects the injected edge but the full audit still says "
            "NOT ESTABLISHED (concentration or another structural check) -- "
            "power is conditional on that check's failure mode"
        )

    return OracleControlResult(
        n_obs=int(v.size),
        n_clusters=(rep.clustering.n_clusters if rep.clustering is not None else None),
        ci_kind=rep.ci_kind,
        claimed_effect=claimed_effect,
        injected_ci=rep.ci,
        detected=bool(detected),
        powered=bool(detected),
        mde=float(mde),
        margin=float(margin),
        injected_verdict=rep.verdict,
        sweep=sweep,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# B) model pipelines: the target itself as a feature must score ~perfectly
# ---------------------------------------------------------------------------

@dataclass
class OracleProbeResult:
    """Verdict of the target-as-feature oracle probe."""
    n_obs: int
    n_positive: Optional[int]      # binary targets only
    metric: str                    # "auc" (binary y) or "spearman" (continuous y)
    score_oracle: float            # pipeline score WITH the target as a feature
    score_baseline: Optional[float]  # honest run, for contrast
    threshold: float
    powered: bool                  # True -> the pipeline can see its own target
    notes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "OracleProbeResult (target-as-feature power probe)",
            f"  observations   : {self.n_obs}"
            + (f"  ({self.n_positive} positive)" if self.n_positive is not None else ""),
            f"  oracle {self.metric:<8} : {self.score_oracle:.4f}"
            f"  (threshold {self.threshold:.2f})",
        ]
        if self.score_baseline is not None:
            lines.append(f"  honest {self.metric:<8} : {self.score_baseline:.4f}")
        lines.append(f"  powered        : {self.powered}")
        lines += [f"  ! {n}" for n in self.notes]
        return "\n".join(lines)


def _auc(y: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based (Mann-Whitney) AUC, tie-safe."""
    n_pos = int((y == 1).sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC needs both classes present in y")
    ranks = _ss.rankdata(scores)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def oracle_probe(
    fit_score: Callable[[np.ndarray, np.ndarray], np.ndarray],
    X, y,
    threshold: float = 0.95,
    run_baseline: bool = True,
) -> OracleProbeResult:
    """Append the target itself as a feature and demand a near-perfect score.

    Parameters
    ----------
    fit_score : callable
        ``fit_score(X, y) -> scores`` -- YOUR pipeline, exactly as configured
        for the real run (same model, same hyperparameters, same
        preprocessing), fitting on ``(X, y)`` and returning one predicted
        score per row. Only you know how your pipeline is wired; this
        function only injects the oracle column and judges the result.
    X : array-like, shape (n, k)
        The feature matrix of the real run. The probe appends ``y`` as an
        extra column, so the pipeline must accept k+1 features.
    y : array-like
        The target. Binary (0/1) targets are scored with AUC; anything else
        with the Spearman rank correlation between ``y`` and the scores.
    threshold : float
        ``score_oracle >= threshold`` counts as powered. Default 0.95: with
        the answer as a feature, anything materially below ~1.0 means the
        pipeline structurally cannot use even a perfect signal.

    Judgment
    --------
    ``powered=False`` means: this setup is too weak to detect ANY real
    signal -- every negative it produced is void, exactly like a smoke
    detector that stays silent in real smoke. Typical cause on rare targets:
    tree models whose ``min_child_samples`` (or equivalent) exceeds what the
    positive count can support -- the tree cannot split even on the answer.
    Fix the config (or switch to a logit for the probe), re-run, and only
    then read the honest negative.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y, dtype=float).ravel()
    if X.shape[0] != y.size:
        raise ValueError(f"X ({X.shape[0]} rows) and y ({y.size}) must match")
    if y.size < 4:
        raise ValueError("need at least 4 observations")

    binary = set(np.unique(y).tolist()) <= {0.0, 1.0}
    n_pos = int((y == 1).sum()) if binary else None

    X_oracle = np.column_stack([X, y])
    scores_o = np.asarray(fit_score(X_oracle, y), dtype=float).ravel()
    if scores_o.size != y.size:
        raise ValueError("fit_score must return one score per row")

    notes: list[str] = []

    def _score(scores: np.ndarray) -> float:
        if binary:
            return _auc(y, scores)
        rho = _ss.spearmanr(y, scores).statistic
        return float(rho) if np.isfinite(rho) else 0.0

    if np.allclose(scores_o, scores_o[0]):
        notes.append("pipeline returned a CONSTANT score with the target as a "
                     "feature -- it is not using its inputs at all")
    score_oracle = _score(scores_o)

    score_baseline = None
    if run_baseline:
        scores_b = np.asarray(fit_score(X, y), dtype=float).ravel()
        score_baseline = _score(scores_b)

    powered = score_oracle >= threshold
    if not powered:
        notes.append(
            f"PIPELINE CANNOT SEE ITS OWN TARGET: with the answer injected as "
            f"a feature the {'AUC' if binary else 'rank correlation'} is only "
            f"{score_oracle:.3f} < {threshold:.2f} -- this setup is too weak "
            f"to detect any real signal; every negative it produced is void"
        )
    if binary and n_pos is not None and (n_pos < 50 or n_pos < 0.02 * y.size):
        notes.append(
            f"rare target ({n_pos} positives, base rate {n_pos / y.size:.2%}): "
            f"tree models with min-samples-per-leaf constraints near or above "
            f"the positive count cannot split even on a leaked answer -- the "
            f"classic phantom-negative trap this probe exists for"
        )

    return OracleProbeResult(
        n_obs=int(y.size),
        n_positive=n_pos,
        metric="auc" if binary else "spearman",
        score_oracle=float(score_oracle),
        score_baseline=score_baseline,
        threshold=float(threshold),
        powered=bool(powered),
        notes=notes,
    )
