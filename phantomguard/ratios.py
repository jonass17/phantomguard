"""Risk-adjusted return ratios beyond Sharpe: Sortino, Calmar, max drawdown.

Sharpe penalizes ALL volatility -- including the upside nobody complains
about. That makes it systematically unfair to positively-skewed strategies
(trend following, long volatility), which look "risky" precisely because
their wins are lumpy. Two complementary lenses:

- Sortino: like Sharpe, but only DOWNSIDE deviation counts as risk.
- Calmar: annual return divided by the worst peak-to-trough drawdown --
  return per nightmare.

Same convention as ``stats``: everything is computed per-observation;
annualize only for display.
"""
from __future__ import annotations

import numpy as np


def _as_returns(returns) -> np.ndarray:
    r = np.asarray(returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    if r.size < 2:
        raise ValueError("need at least 2 finite return observations")
    return r


def max_drawdown(returns, compound: bool = True) -> float:
    """Worst peak-to-trough loss of the equity curve, as a POSITIVE fraction.

    ``compound=True`` builds the equity curve multiplicatively (right for
    percentage returns); ``compound=False`` uses the cumulative sum (right
    for additive PnL series).
    """
    r = _as_returns(returns)
    if compound:
        equity = np.cumprod(1.0 + r)
        peak = np.maximum.accumulate(equity)
        dd = 1.0 - equity / peak
    else:
        equity = np.cumsum(r)
        peak = np.maximum.accumulate(equity)
        dd = peak - equity
    return float(dd.max())


def downside_deviation(returns, target: float = 0.0) -> float:
    """Root-mean-square of returns BELOW ``target`` (full-sample denominator,
    the standard Sortino convention)."""
    r = _as_returns(returns)
    shortfall = np.minimum(r - target, 0.0)
    return float(np.sqrt(np.mean(shortfall ** 2)))


def sortino_ratio(returns, target: float = 0.0) -> float:
    """Per-observation Sortino ratio: mean excess over downside deviation.

    Only volatility below ``target`` counts as risk. Returns ``inf`` when the
    sample contains no downside at all -- treat that as "sample too small to
    measure risk", not as a good score. Annualize for display with
    ``annualize_sharpe`` (same sqrt-time scaling convention).
    """
    r = _as_returns(returns)
    dd = downside_deviation(r, target)
    excess = float(r.mean() - target)
    if dd == 0.0:
        return float("inf") if excess > 0 else 0.0
    return excess / dd


def calmar_ratio(returns, periods_per_year: float, compound: bool = True) -> float:
    """Annualized return divided by max drawdown -- return per nightmare.

    Uses CAGR when ``compound=True``, otherwise the annualized mean of the
    additive series. Returns ``inf`` when there was no drawdown (again:
    a too-short sample, not brilliance).
    """
    r = _as_returns(returns)
    mdd = max_drawdown(r, compound=compound)
    if compound:
        growth = float(np.prod(1.0 + r))
        if growth <= 0:
            return float("-inf")
        annual = growth ** (periods_per_year / r.size) - 1.0
    else:
        annual = float(r.mean()) * periods_per_year
    if mdd == 0.0:
        return float("inf") if annual > 0 else 0.0
    return float(annual / mdd)
