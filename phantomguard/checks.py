"""Phantom detectors -- cheap structural checks that catch the classic ways a
backtest lies before you ever look at the Sharpe ratio.

These are heuristics, not proofs. A raised flag means "go look", not "it's
broken". Some phantoms (survivorship, look-ahead in feature construction) can't
be seen from a return series alone, so they live in ``PhantomChecklist`` as
explicit yes/no questions you must answer by hand.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def lookahead_flag(signal, forward_return, contemporaneous_return,
                   margin: float = 1.10):
    """Flag look-ahead: a signal should predict the FUTURE, not the present.

    If ``|corr(signal, contemporaneous)|`` materially exceeds
    ``|corr(signal, forward)|``, the signal is probably peeking at the same bar
    it claims to predict (a classic leak). Returns ``(flagged, detail)``.
    """
    s = np.asarray(signal, dtype=float).ravel()
    fwd = np.asarray(forward_return, dtype=float).ravel()
    con = np.asarray(contemporaneous_return, dtype=float).ravel()
    m = min(s.size, fwd.size, con.size)
    s, fwd, con = s[:m], fwd[:m], con[:m]
    mask = np.isfinite(s) & np.isfinite(fwd) & np.isfinite(con)
    if mask.sum() < 3:
        return False, "too few points to assess look-ahead"
    c_fwd = abs(np.corrcoef(s[mask], fwd[mask])[0, 1])
    c_con = abs(np.corrcoef(s[mask], con[mask])[0, 1])
    flagged = c_con > margin * max(c_fwd, 1e-9)
    return bool(flagged), (f"|corr| contemporaneous={c_con:.3f} vs "
                           f"forward={c_fwd:.3f} (margin x{margin})")


def stale_value_flag(series, max_repeat_frac: float = 0.20):
    """Flag stale/frozen data: long runs of identical values.

    Stale order-book or price marks inflate fills and mean-reversion edges that
    evaporate live. Returns ``(flagged, detail)`` with the largest repeat run as
    a fraction of the sample.
    """
    x = np.asarray(series, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size < 2:
        return False, "too few points"
    same = np.concatenate([[False], x[1:] == x[:-1]])
    # longest consecutive run of repeats
    longest = cur = 0
    for s in same:
        cur = cur + 1 if s else 0
        longest = max(longest, cur)
    frac = (longest + 1) / x.size if longest else 0.0
    return bool(frac > max_repeat_frac), f"longest stale run = {frac:.1%} of sample"


@dataclass
class PhantomChecklist:
    """The phantoms a return series can't reveal -- answer these by hand.

    ``True`` means "I have controlled for this". ``unresolved()`` lists the ones
    you haven't, which is exactly where overfit edges hide.
    """
    universe_frozen_per_date: bool = False    # no survivorship: universe fixed as-of date
    costs_modeled_per_asset: bool = False     # real per-asset slippage, no fixed floor
    no_d_minus_1_mark: bool = False           # entries/exits not marked at look-ahead prices
    resolution_from_source: bool = False      # outcomes from settlement, not a proxy
    no_nested_duplicates: bool = False        # deduped overlapping/nested positions
    trials_counted_cumulatively: bool = False # n_trials spans the whole search, not one run
    notes: dict = field(default_factory=dict)

    def unresolved(self):
        return [k for k, v in vars(self).items()
                if k != "notes" and v is False]

    def clean(self) -> bool:
        return len(self.unresolved()) == 0
