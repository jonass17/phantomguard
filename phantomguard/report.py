"""Tie the statistics, splits and gates into one verdict.

``evaluate`` is the front door: hand it the selected strategy's returns, how
many trials you ran, and (optionally) per-fold OOS PnL, and it returns a
``Verdict`` that says PASS only if every honest gate is cleared.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import stats
from .gates import Gates, DEFAULT_GATES
from .walkforward import positive_fold_fraction


@dataclass
class Verdict:
    passed: bool
    reasons: list = field(default_factory=list)   # why it failed / warnings
    metrics: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def __str__(self) -> str:
        head = "PASS ✅" if self.passed else "FAIL ❌"
        lines = [f"PhantomGuard verdict: {head}", "", "metrics:"]
        for k, v in self.metrics.items():
            lines.append(f"  {k:<22} {v}")
        if self.reasons:
            lines += ["", "gate failures:"] + [f"  - {r}" for r in self.reasons]
        if self.warnings:
            lines += ["", "warnings:"] + [f"  ! {w}" for w in self.warnings]
        return "\n".join(lines)


def evaluate(returns, n_trials: int, sr_trials=None, sr_variance=None,
             fold_pnls=None, periods_per_year: float = 1.0,
             gates: Gates = DEFAULT_GATES, n_boot: int = 10000) -> Verdict:
    """Run the full honest battery and return a PASS/FAIL ``Verdict``.

    Parameters mirror the stats module. ``sr_trials`` or ``sr_variance`` are
    required for the Deflated Sharpe gate -- without a trial count, DSR is
    meaningless and PhantomGuard refuses to fake it.
    """
    r = np.asarray(returns, dtype=float).ravel()
    r = r[np.isfinite(r)]

    psr = stats.probabilistic_sharpe_ratio(r)
    point, lo, hi = stats.bootstrap_sharpe_ci(
        r, n_boot=n_boot, periods_per_year=periods_per_year)
    ann_sr = stats.annualize_sharpe(stats.sharpe_ratio(r), periods_per_year)
    oos_pnl = float(r.sum())

    metrics = {
        "n_obs": r.size,
        "n_trials": n_trials,
        "sharpe_annual": round(ann_sr, 3),
        "PSR": round(psr, 4),
        "boot_CI_sharpe": (round(lo, 3), round(hi, 3)),
        "oos_pnl": round(oos_pnl, 6),
        "min_track_record_len": round(stats.min_track_record_length(r), 1),
    }

    reasons, warnings = [], []

    if oos_pnl <= gates.oos_pnl_min:
        reasons.append(f"OOS PnL {oos_pnl:.4f} <= {gates.oos_pnl_min}")
    if psr < gates.psr_min:
        reasons.append(f"PSR {psr:.3f} < {gates.psr_min}")
    if lo <= gates.ci_lower_min:
        reasons.append(f"bootstrap CI lower {lo:.3f} <= {gates.ci_lower_min}")

    # Deflated Sharpe -- only if we were given trial dispersion.
    if sr_trials is not None or sr_variance is not None:
        dsr = stats.deflated_sharpe_ratio(
            r, n_trials=n_trials, sr_trials=sr_trials, sr_variance=sr_variance)
        metrics["DSR"] = round(dsr, 4)
        if dsr < gates.dsr_min:
            reasons.append(f"DSR {dsr:.3f} < {gates.dsr_min} "
                           f"(n_trials={n_trials})")
    else:
        warnings.append("no sr_trials/sr_variance given -> DSR gate skipped; "
                        "an un-deflated result is not yet trustworthy")

    # Positive-fold consistency.
    if fold_pnls is not None:
        pf = positive_fold_fraction(fold_pnls)
        metrics["pos_folds"] = round(pf, 3)
        if pf < gates.pos_folds_min:
            reasons.append(f"positive folds {pf:.2f} < {gates.pos_folds_min}")
    else:
        warnings.append("no fold_pnls given -> positive-fold gate skipped")

    # Overfit smell tests (warnings, not hard fails).
    if ann_sr > gates.sharpe_overfit_warn:
        warnings.append(f"annual Sharpe {ann_sr:.2f} > "
                        f"{gates.sharpe_overfit_warn}: overfit suspect")

    passed = len(reasons) == 0
    return Verdict(passed=passed, reasons=reasons, metrics=metrics,
                   warnings=warnings)
