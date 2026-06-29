# PhantomGuard

**Honest backtest statistics and anti-overfitting guards — in one small library.**

Most backtests are *phantoms*: a great-looking Sharpe ratio that survives only
because nobody deflated it for the hundreds of things you tried, checked it out
of sample, or seriously tried to break it. PhantomGuard packages the discipline
that kills phantoms **before** they cost real money.

It does four things:

1. **Significance that accounts for reality** — Probabilistic Sharpe Ratio (PSR),
   **Deflated Sharpe Ratio (DSR)** with *cumulative* trial counts, Minimum Track
   Record Length, and a bootstrap Sharpe confidence interval. Fat tails and skew
   are handled; a short track record is penalised.
2. **Leak-free out-of-sample splits** — anchored walk-forward and Lopez de Prado
   purged K-Fold with embargo, plus a positive-fold-fraction check.
3. **Phantom detectors** — cheap structural flags for look-ahead, stale/frozen
   data, and an explicit checklist for the biases a return series can't reveal
   (survivorship, D-1 marking, nested duplicates).
4. **An adversarial LLM verifier** — hand the whole result to a skeptic model
   whose only job is to *refute* the edge. Only what it can't break should count.

> The philosophy in one line: **a high hit-rate is not an edge.** PhantomGuard
> exists to tell the difference.

## Install

```bash
pip install phantomguard            # core (numpy + scipy)
pip install "phantomguard[llm]"     # + adversarial LLM verifier (anthropic)
```

## Quickstart

```python
import numpy as np
from phantomguard import evaluate

rng = np.random.default_rng(0)
returns  = rng.normal(0.0005, 0.01, 500)   # the strategy you want to deploy
sr_trials = rng.normal(0, 0.05, 800)       # the Sharpes of ALL 800 things you tried

verdict = evaluate(
    returns,
    n_trials=800,            # cumulative across the whole search — not this run!
    sr_trials=sr_trials,     # needed for the Deflated Sharpe gate
    fold_pnls=[0.10, -0.02, 0.05, 0.08, -0.01],
    periods_per_year=252,
)
print(verdict)
```

```
PhantomGuard verdict: FAIL ❌

metrics:
  n_obs                  500
  n_trials               800
  sharpe_annual          0.362
  PSR                    0.6942
  boot_CI_sharpe         (-0.97, 1.738)
  oos_pnl                0.115551
  min_track_record_len   5236.8
  DSR                    0.0017
  pos_folds              0.6

gate failures:
  - PSR 0.694 < 0.95
  - bootstrap CI lower -0.970 <= 0.0
  - DSR 0.002 < 0.9 (n_trials=800)
```

A positive single-run PnL, **failed** — because the edge isn't distinguishable
from noise once you account for the 800 trials and the bootstrap spread. (Run it
yourself: `python examples/quickstart.py` — these are the real numbers, and the
same script shows a clean edge passing.) That's the whole point.

## Probability of Backtest Overfitting (PBO)

The single most honest number for a strategy *search*. You tried N configs and
kept the best — how likely is it that your winner is just the luckiest noise and
will rank below median out of sample? PhantomGuard estimates it with
Combinatorially-Symmetric Cross-Validation (no parametric assumptions):

```python
import numpy as np
from phantomguard import pbo_cscv

# rows = time, columns = every strategy/config you tried
trials = np.random.default_rng(0).normal(0, 1, size=(2000, 50))
res = pbo_cscv(trials, n_blocks=16)
print(res)
# PBO = 0.49  [LIKELY OVERFIT]  (50 trials, 16 blocks, 6435 splits)
#   P(champion loses OOS)      = 0.50
#   perf degradation slope     = ...
```

PBO near **0.5** means your selection process has *no* out-of-sample skill —
exactly what pure noise produces. A genuinely persistent edge drives PBO toward
**0**.

## Command line

```bash
phantomguard check trades.csv --trials 800 --sr-trials trial_sharpes.csv -p 252
phantomguard pbo trial_matrix.csv --blocks 16
```

`check` exits non-zero if the gates fail and `pbo` exits non-zero if PBO > 0.5 —
so you can wire PhantomGuard straight into CI and fail the build on a phantom.

## Adversarial verification

```python
from phantomguard import adversarial_verify

# offline=True returns the skeptic prompt so you can run it in any model:
print(adversarial_verify(verdict, offline=True)["prompt"])

# or live (needs ANTHROPIC_API_KEY and `pip install "phantomguard[llm]"`):
result = adversarial_verify(verdict, model="claude-sonnet-4-6")
print(result["verdict"], result["attacks"])
```

## The gates

Defaults are deliberately strict. An edge passes only if **all** hard gates clear:

| Gate | Default | Why |
|------|---------|-----|
| OOS PnL | `> 0` | it has to make money out of sample |
| PSR | `> 0.95` | Sharpe is significant given length, skew, kurtosis |
| Deflated Sharpe | `> 0.90` | significant *after* multiple testing |
| Bootstrap CI lower | `> 0` | lower bound doesn't straddle zero |
| Positive folds | `>= 0.60` | not carried by one lucky window |
| Annual Sharpe | `> 3.0` ⇒ ⚠️ | warns: too good is usually overfit |

Loosen them consciously, in code (`phantomguard.Gates(...)`), where a reviewer
can see it.

## Why these methods

- Bailey & Lopez de Prado (2012), *The Sharpe Ratio Efficient Frontier* — PSR, MinTRL.
- Bailey & Lopez de Prado (2014), *The Deflated Sharpe Ratio* — DSR, expected max Sharpe.
- Harvey & Liu (2015), *Backtesting* — multiple-testing corrections in finance.

## Status

`0.2.0` — core statistics, **PBO/CSCV**, and a **CLI** are in and tested
(20 tests). Next: HTML report, backtester adapters (vectorbt/backtrader), PyPI.
Issues and PRs welcome, especially additional phantom detectors and verifier
back-ends.

## License

MIT
