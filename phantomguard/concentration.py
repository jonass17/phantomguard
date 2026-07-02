"""Concentration check: does the edge survive without its best day?

A backtest whose entire profit comes from one day (or one week, one market,
one event) is not a strategy -- it is an anecdote. Regime luck placed one fat
outcome inside the sample window, and the average dressed it up as skill.
This is one of the most reliable phantom signatures: point estimates look
healthy, and removing a single group makes the whole thing negative.

The check groups trades (by day, by market, by whatever label you pass),
removes the single best group, and asks whether anything is left. It also
reports how much of the total PnL the best group carries.

This is a robustness diagnostic, not a licence to trim: if your result dies
without its best day, the conclusion is "not established", never "let me
exclude that day and keep the rest".

Longshot caveat
---------------
Some strategies are *supposed* to look like this: trend following, long
volatility, cheap-option buying -- many small losses, rare fat wins. For
those, this check will flag by design, and the flag does not mean "dead".
It means: your evidence rests on rare events, so you need a track record
long enough to contain SEVERAL of them at a stable rate. One windfall is
an anecdote; ten windfalls across years is a distribution. Until then the
honest verdict is still "not established" -- just for a different reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# If the best group carries more than this share of total profit, flag it.
SHARE_WARN = 0.5


def _validate(values, groups):
    v = np.asarray(values, dtype=float).ravel()
    g = np.asarray(groups).ravel()
    if v.size != g.size:
        raise ValueError(f"values ({v.size}) and groups ({g.size}) must have the same length")
    mask = np.isfinite(v)
    v, g = v[mask], g[mask]
    if v.size < 2:
        raise ValueError("need at least 2 finite observations")
    if np.unique(g).size < 2:
        raise ValueError("need at least 2 distinct groups (e.g. days) to test concentration")
    return v, g


@dataclass
class ConcentrationResult:
    """Outcome of the remove-the-best-group robustness check."""
    n_obs: int
    n_groups: int
    mean_all: float
    total_pnl: float
    best_group: object            # label of the most profitable group
    best_group_pnl: float
    best_group_share: float       # best group PnL / total PnL (if total > 0)
    mean_ex_best: float           # mean after removing the best group
    positive_groups: int
    concentrated: bool            # True -> result rests on one group
    sign_flip: bool               # True -> mean goes non-positive without best group
    notes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        share = (f"{self.best_group_share:.0%}" if np.isfinite(self.best_group_share)
                 else "n/a (total <= 0)")
        lines = [
            "ConcentrationResult",
            f"  observations       : {self.n_obs} in {self.n_groups} groups",
            f"  mean (all)         : {self.mean_all:+.6g}",
            f"  best group         : {self.best_group!r}"
            f"  (PnL {self.best_group_pnl:+.6g}, {share} of total)",
            f"  mean ex-best group : {self.mean_ex_best:+.6g}",
            f"  positive groups    : {self.positive_groups}/{self.n_groups}",
        ]
        lines += [f"  ! {n}" for n in self.notes]
        return "\n".join(lines)


def concentration_check(values, groups) -> ConcentrationResult:
    """Remove the single most profitable group and see what is left.

    Parameters
    ----------
    values : array-like
        Per-trade PnL (or returns).
    groups : array-like
        Group label per trade -- typically the calendar day of the trade,
        but any unit of shared fate works (market, event, week).

    The verdict to act on: if ``sign_flip`` is True, the sample does not
    establish an edge -- everything positive lives in one group.
    """
    v, g = _validate(values, groups)
    uniq = np.unique(g)
    sums = np.array([v[g == u].sum() for u in uniq])

    best_i = int(np.argmax(sums))
    best_label = uniq[best_i]
    best_pnl = float(sums[best_i])
    total = float(v.sum())

    ex_mask = g != best_label
    mean_ex = float(v[ex_mask].mean())
    share = best_pnl / total if total > 0 else float("inf")
    positive_groups = int((sums > 0).sum())

    notes = []
    sign_flip = mean_ex <= 0 < float(v.mean())
    if sign_flip:
        notes.append(
            f"SIGN FLIP: remove group {best_label!r} and the mean goes "
            f"{mean_ex:+.4g} -- the entire edge is one group's story"
        )
    concentrated = (total > 0 and share > SHARE_WARN) or sign_flip
    if total > 0 and share > SHARE_WARN and not sign_flip:
        notes.append(
            f"best group carries {share:.0%} of total PnL -- "
            f"fragile even though the ex-best mean stays positive"
        )
    if positive_groups < max(2, int(0.5 * uniq.size)):
        notes.append(
            f"only {positive_groups}/{uniq.size} groups are positive -- "
            f"the edge is not broadly present across the sample"
        )

    return ConcentrationResult(
        n_obs=int(v.size),
        n_groups=int(uniq.size),
        mean_all=float(v.mean()),
        total_pnl=total,
        best_group=best_label,
        best_group_pnl=best_pnl,
        best_group_share=float(share),
        mean_ex_best=mean_ex,
        positive_groups=positive_groups,
        concentrated=bool(concentrated),
        sign_flip=bool(sign_flip),
        notes=notes,
    )
