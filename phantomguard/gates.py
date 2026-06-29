"""Acceptance gates -- the pass/fail thresholds an edge must clear.

Defaults encode a deliberately strict, anti-overfitting stance. Loosen them
consciously, in code, where a reviewer can see it -- never by eyeballing.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Gates:
    oos_pnl_min: float = 0.0      # out-of-sample PnL must be positive
    psr_min: float = 0.95         # Probabilistic Sharpe Ratio
    dsr_min: float = 0.90         # Deflated Sharpe Ratio (cumulative n_trials!)
    ci_lower_min: float = 0.0     # bootstrap Sharpe CI lower bound
    pos_folds_min: float = 0.60   # fraction of OOS folds in profit
    sharpe_overfit_warn: float = 3.0  # annual Sharpe above this -> overfit suspect
    win_rate_overfit_warn: float = 0.60  # win rate above this -> overfit suspect


DEFAULT_GATES = Gates()
