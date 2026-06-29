"""Out-of-sample splitting that respects time order.

In-sample Sharpe ratios lie. The questions that matter are: does the edge hold
on data the model never saw, and is it consistent across folds (not carried by
one lucky window)? These helpers give you leak-free splits to answer that.

``purged_kfold`` implements Lopez de Prado's purging + embargo so that
overlapping labels in fold k cannot leak into the training set.
"""
from __future__ import annotations

import numpy as np


def walk_forward_splits(n: int, n_splits: int = 5, min_train: int | None = None):
    """Anchored (expanding-window) walk-forward splits over ``n`` ordered points.

    Yields ``(train_idx, test_idx)`` with time order preserved -- training data
    always precedes its test window. No shuffling, ever.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    idx = np.arange(n)
    fold = n // (n_splits + 1)
    if fold < 1:
        raise ValueError("not enough observations for the requested n_splits")
    if min_train is None:
        min_train = fold
    for k in range(1, n_splits + 1):
        train_end = min_train + (k - 1) * fold
        test_end = min(train_end + fold, n)
        if train_end >= n or train_end >= test_end:
            break
        yield idx[:train_end], idx[train_end:test_end]


def purged_kfold(n: int, n_splits: int = 5, embargo: float = 0.0):
    """Purged K-Fold with an embargo around each test fold (Lopez de Prado).

    ``embargo`` is a fraction of ``n`` removed on both sides of the test fold so
    that label overlap / serial correlation cannot leak from test into train.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    idx = np.arange(n)
    fold_sizes = np.full(n_splits, n // n_splits, dtype=int)
    fold_sizes[: n % n_splits] += 1
    bounds = np.cumsum(np.concatenate([[0], fold_sizes]))
    emb = int(np.ceil(embargo * n))
    for k in range(n_splits):
        start, stop = bounds[k], bounds[k + 1]
        test_idx = idx[start:stop]
        keep = np.ones(n, dtype=bool)
        keep[max(0, start - emb): min(n, stop + emb)] = False
        yield idx[keep], test_idx


def positive_fold_fraction(fold_pnls) -> float:
    """Fraction of out-of-sample folds with PnL > 0.

    A real edge shows up in most folds. A gate of >= 0.60 weeds out edges that
    are really one good window in disguise.
    """
    arr = np.asarray(fold_pnls, dtype=float)
    if arr.size == 0:
        return 0.0
    return float((arr > 0).mean())
