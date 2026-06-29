"""Probability of Backtest Overfitting (PBO) via CSCV.

When you search over N strategy configurations and keep the best one, how likely
is it that your "winner" is just the luckiest noise -- i.e. that it will rank
below median out-of-sample? That probability is the PBO, and it is arguably the
single most honest number you can attach to a strategy search.

Combinatorially-Symmetric Cross-Validation (CSCV) estimates it without a single
parametric assumption: split the timeline into S blocks, take every balanced
way to send half the blocks in-sample and half out-of-sample, and watch how
often the in-sample champion collapses out-of-sample.

Reference
---------
Bailey, Borwein, Lopez de Prado & Zhu (2017), "The Probability of Backtest
Overfitting", Journal of Computational Finance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np


@dataclass
class PBOResult:
    pbo: float                      # P(in-sample best ranks below OOS median)
    n_trials: int                   # number of strategies/configs (columns)
    n_blocks: int                   # S
    n_splits: int                   # number of balanced combinations evaluated
    logits: np.ndarray = field(repr=False)          # logit of OOS relative rank
    oos_ranks: np.ndarray = field(repr=False)        # relative rank in (0,1)
    is_perf: np.ndarray = field(repr=False)          # IS Sharpe of chosen champion
    oos_perf: np.ndarray = field(repr=False)         # OOS Sharpe of chosen champion
    prob_oos_loss: float = 0.0      # fraction of splits where champion loses OOS
    perf_degradation_slope: float = 0.0  # OLS slope of OOS-perf on IS-perf (<1 = decay)

    def __str__(self) -> str:
        verdict = ("LIKELY OVERFIT" if self.pbo > 0.5 else
                   "borderline" if self.pbo > 0.25 else "holds up")
        return (f"PBO = {self.pbo:.3f}  [{verdict}]  "
                f"({self.n_trials} trials, {self.n_blocks} blocks, "
                f"{self.n_splits} splits)\n"
                f"  P(champion loses OOS)      = {self.prob_oos_loss:.3f}\n"
                f"  perf degradation slope     = {self.perf_degradation_slope:.3f} "
                f"(1.0 = none, <0 = inverts)")


def _block_sufficient_stats(M: np.ndarray, n_blocks: int):
    """Per-block sum and sum-of-squares per strategy, for exact Sharpe pooling."""
    T = M.shape[0]
    usable = (T // n_blocks) * n_blocks
    M = M[:usable]
    blocks = M.reshape(n_blocks, usable // n_blocks, M.shape[1])
    s1 = blocks.sum(axis=1)               # (S, N) sum of returns
    s2 = (blocks ** 2).sum(axis=1)        # (S, N) sum of squares
    cnt = np.full(n_blocks, usable // n_blocks, dtype=float)
    return s1, s2, cnt


def _pooled_sharpe(s1_sel, s2_sel, n):
    """Sharpe per strategy from pooled sufficient stats over selected blocks."""
    mean = s1_sel / n
    var = (s2_sel - n * mean ** 2) / (n - 1)
    sd = np.sqrt(np.maximum(var, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = np.where(sd > 0, mean / sd, 0.0)
    return sr


def pbo_cscv(returns_matrix, n_blocks: int = 16) -> PBOResult:
    """Probability of Backtest Overfitting via CSCV.

    Parameters
    ----------
    returns_matrix : array-like, shape (T, N)
        Per-observation returns for each of the N strategies/configurations you
        tried, aligned in time across rows. This is the *whole search*, not one
        survivor -- PBO is a statement about the search.
    n_blocks : int, even
        Number of contiguous time blocks S. C(S, S/2) balanced splits are
        evaluated (S=16 -> 12870). Larger S = finer but combinatorially heavier.

    Returns
    -------
    PBOResult
        ``.pbo`` is the headline: the fraction of splits in which the in-sample
        best configuration ranked at or below the median out-of-sample. PBO near
        0.5 means your selection process has no out-of-sample skill at all.
    """
    M = np.asarray(returns_matrix, dtype=float)
    if M.ndim != 2 or M.shape[1] < 2:
        raise ValueError("returns_matrix must be 2-D with >= 2 strategy columns")
    if n_blocks % 2 != 0 or n_blocks < 4:
        raise ValueError("n_blocks must be even and >= 4")
    if M.shape[0] < n_blocks * 2:
        raise ValueError("need at least 2 observations per block")

    N = M.shape[1]
    s1, s2, cnt = _block_sufficient_stats(M, n_blocks)
    all_blocks = set(range(n_blocks))
    half = n_blocks // 2

    logits, oos_ranks, is_perf, oos_perf = [], [], [], []
    seen = set()
    for is_blocks in combinations(range(n_blocks), half):
        # CSCV is symmetric: (IS, OOS) and (OOS, IS) are the same split once.
        key = frozenset(is_blocks)
        comp = frozenset(all_blocks - key)
        if comp in seen:
            continue
        seen.add(key)

        is_idx = list(is_blocks)
        oos_idx = list(all_blocks - key)
        n_is = cnt[is_idx].sum()
        n_oos = cnt[oos_idx].sum()

        is_sr = _pooled_sharpe(s1[is_idx].sum(0), s2[is_idx].sum(0), n_is)
        oos_sr = _pooled_sharpe(s1[oos_idx].sum(0), s2[oos_idx].sum(0), n_oos)

        champ = int(np.argmax(is_sr))                  # in-sample winner
        # relative OOS rank of the champion among all N (1 = best, N = worst)
        rank = int((oos_sr <= oos_sr[champ]).sum())    # how many it beats/ties
        omega = rank / (N + 1)                          # in (0,1)
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        logits.append(np.log(omega / (1 - omega)))
        oos_ranks.append(omega)
        is_perf.append(is_sr[champ])
        oos_perf.append(oos_sr[champ])

    logits = np.asarray(logits)
    oos_ranks = np.asarray(oos_ranks)
    is_perf = np.asarray(is_perf)
    oos_perf = np.asarray(oos_perf)

    pbo = float((logits <= 0).mean())          # champion at/below OOS median
    prob_oos_loss = float((oos_perf < 0).mean())
    # OLS slope of OOS performance on IS performance: how much the edge decays.
    if np.ptp(is_perf) > 0:
        slope = float(np.polyfit(is_perf, oos_perf, 1)[0])
    else:
        slope = 0.0

    return PBOResult(
        pbo=pbo, n_trials=N, n_blocks=n_blocks, n_splits=len(logits),
        logits=logits, oos_ranks=oos_ranks, is_perf=is_perf, oos_perf=oos_perf,
        prob_oos_loss=prob_oos_loss, perf_degradation_slope=slope,
    )
