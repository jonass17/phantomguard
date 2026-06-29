"""Honest Sharpe-ratio statistics.

Probabilistic Sharpe Ratio (PSR), Deflated Sharpe Ratio (DSR), Minimum Track
Record Length (MinTRL) and a bootstrap Sharpe confidence interval.

The point of this module: a single observed Sharpe ratio means almost nothing.
What matters is whether it survives (a) the sampling error of a *short* track
record, (b) non-normal returns (fat tails, skew), and (c) the multiple testing
you did to find it. PSR handles (a)+(b); DSR adds (c).

References
----------
Bailey & Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier",
    Journal of Risk.  -> PSR, MinTRL
Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio: Correcting for
    Selection Bias, Backtest Overfitting and Non-Normality",
    Journal of Portfolio Management.  -> DSR, expected max Sharpe
"""
from __future__ import annotations

import numpy as np
from scipy import stats as _ss

# Euler-Mascheroni constant, used in the expected-maximum-Sharpe estimator.
EULER_MASCHERONI = 0.5772156649015329


def _as_returns(returns) -> np.ndarray:
    r = np.asarray(returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    if r.size < 2:
        raise ValueError("need at least 2 finite return observations")
    return r


def sharpe_ratio(returns, benchmark: float = 0.0) -> float:
    """Per-observation (NON-annualized) Sharpe ratio.

    Every other function here expects Sharpe figures in this same per-observation
    frequency. Annualize only for display, via ``annualize_sharpe``.
    """
    r = _as_returns(returns)
    excess = r - benchmark
    sd = excess.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(excess.mean() / sd)


def annualize_sharpe(sr_per_obs: float, periods_per_year: float) -> float:
    """Scale a per-observation Sharpe to annual (e.g. 252 for daily bars)."""
    return sr_per_obs * np.sqrt(periods_per_year)


def _moments(returns):
    r = _as_returns(returns)
    sr = sharpe_ratio(r)
    sd = r.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        # Constant series carries no skew/kurtosis information and Sharpe is 0.
        # Treat as normal so PSR resolves to 0.5 ("no evidence") instead of NaN.
        return 0.0, 0.0, 3.0, r.size
    g3 = float(_ss.skew(r, bias=False))                 # skewness
    g4 = float(_ss.kurtosis(r, fisher=False, bias=False))  # kurtosis (normal == 3)
    return sr, g3, g4, r.size


def _psr_se(sr: float, g3: float, g4: float, n: int) -> float:
    """Standard error of the Sharpe estimator (Mertens / Bailey-LdP)."""
    var = 1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr ** 2
    return float(np.sqrt(max(var, 1e-12) / (n - 1)))


def probabilistic_sharpe_ratio(returns, sr_benchmark: float = 0.0) -> float:
    """P(true Sharpe > ``sr_benchmark``) given the observed track record.

    ``sr_benchmark`` is in the SAME per-observation frequency as ``returns``.
    Returns a probability in [0, 1]. A common acceptance gate is PSR > 0.95.
    """
    sr, g3, g4, n = _moments(returns)
    se = _psr_se(sr, g3, g4, n)
    z = (sr - sr_benchmark) / se
    return float(_ss.norm.cdf(z))


def expected_max_sharpe(sr_variance: float, n_trials: int) -> float:
    """Expected MAXIMUM Sharpe produced by pure luck across ``n_trials``.

    ``sr_variance`` is the variance of the Sharpe ratios across the trials you
    ran (per-observation frequency). This is the bar a real edge must clear:
    if you tried 1000 things, the best one looks good by chance alone.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_trials == 1:
        return 0.0
    g = EULER_MASCHERONI
    z1 = _ss.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = _ss.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(max(sr_variance, 0.0)) * ((1 - g) * z1 + g * z2))


def deflated_sharpe_ratio(returns, n_trials: int, sr_trials=None,
                          sr_variance: float | None = None) -> float:
    """Deflated Sharpe Ratio: PSR measured against the expected-max Sharpe of luck.

    Parameters
    ----------
    returns : array-like
        Per-observation returns of the *selected* (best) strategy.
    n_trials : int
        How many strategies/configs were tried. Use the CUMULATIVE count across
        the whole research effort, not just this run -- this is the single most
        common way DSR gets gamed.
    sr_trials : array-like, optional
        The Sharpe ratios of all trials (per-obs). Its variance is used directly.
    sr_variance : float, optional
        Variance of trial Sharpes, if you already have it instead of ``sr_trials``.

    A common acceptance gate is DSR > 0.90.
    """
    if sr_variance is None:
        if sr_trials is None:
            raise ValueError("provide either sr_trials or sr_variance")
        sr_variance = float(np.var(np.asarray(sr_trials, dtype=float), ddof=1))
    sr0 = expected_max_sharpe(sr_variance, n_trials)
    return probabilistic_sharpe_ratio(returns, sr_benchmark=sr0)


def min_track_record_length(returns, sr_benchmark: float = 0.0,
                            prob: float = 0.95) -> float:
    """Minimum number of observations for PSR(sr_benchmark) to reach ``prob``.

    If your actual sample is shorter than this, you do not yet have the evidence
    to claim the edge -- regardless of how pretty the Sharpe looks.
    """
    sr, g3, g4, _ = _moments(returns)
    if sr <= sr_benchmark:
        return float("inf")
    z = _ss.norm.ppf(prob)
    var_term = 1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr ** 2
    return float(1.0 + var_term * (z / (sr - sr_benchmark)) ** 2)


def bootstrap_sharpe_ci(returns, n_boot: int = 10000, alpha: float = 0.05,
                        seed: int = 0, periods_per_year: float = 1.0):
    """IID bootstrap CI for the Sharpe ratio.

    Returns ``(point, lo, hi)`` annualized by ``periods_per_year``. The honest
    gate is ``lo > 0``: if the lower bound straddles zero, the edge is not
    distinguishable from noise.
    """
    r = _as_returns(returns)
    rng = np.random.default_rng(seed)
    n = r.size
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = r[idx]
    mean = samples.mean(axis=1)
    sd = samples.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        srs = np.where(sd > 0, mean / sd, np.nan) * np.sqrt(periods_per_year)
    lo = float(np.nanpercentile(srs, 100 * alpha / 2))
    hi = float(np.nanpercentile(srs, 100 * (1 - alpha / 2)))
    point = sharpe_ratio(r) * np.sqrt(periods_per_year)
    return point, lo, hi
