"""PhantomGuard -- honest backtest statistics and anti-overfitting guards.

Most backtests are phantoms: a Sharpe ratio that survives only because nobody
deflated it for multiple testing, checked it out of sample, or tried to break
it. PhantomGuard packages the discipline that kills phantoms before they cost
real money.

Quickstart
----------
>>> import numpy as np
>>> from phantomguard import evaluate
>>> rng = np.random.default_rng(0)
>>> rets = rng.normal(0.0005, 0.01, 500)          # a candidate strategy
>>> sr_trials = rng.normal(0, 0.05, 800)          # the 800 things you tried
>>> v = evaluate(rets, n_trials=800, sr_trials=sr_trials,
...              fold_pnls=[0.1, -0.02, 0.05, 0.08, -0.01],
...              periods_per_year=252)
>>> print(v)                                       # doctest: +SKIP
"""
from .stats import (
    sharpe_ratio,
    annualize_sharpe,
    probabilistic_sharpe_ratio,
    expected_max_sharpe,
    deflated_sharpe_ratio,
    min_track_record_length,
    bootstrap_sharpe_ci,
)
from .walkforward import (
    walk_forward_splits,
    purged_kfold,
    positive_fold_fraction,
)
from .checks import lookahead_flag, stale_value_flag, PhantomChecklist
from .gates import Gates, DEFAULT_GATES
from .report import evaluate, Verdict
from .overfitting import pbo_cscv, PBOResult
from .verify import adversarial_verify, build_prompt

__version__ = "0.2.0"

__all__ = [
    "sharpe_ratio", "annualize_sharpe", "probabilistic_sharpe_ratio",
    "expected_max_sharpe", "deflated_sharpe_ratio", "min_track_record_length",
    "bootstrap_sharpe_ci", "walk_forward_splits", "purged_kfold",
    "positive_fold_fraction", "lookahead_flag", "stale_value_flag",
    "PhantomChecklist", "Gates", "DEFAULT_GATES", "evaluate", "Verdict",
    "pbo_cscv", "PBOResult", "adversarial_verify", "build_prompt",
]
