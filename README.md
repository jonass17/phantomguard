# PhantomGuard

[![tests](https://github.com/jonass17/phantomguard/actions/workflows/ci.yml/badge.svg)](https://github.com/jonass17/phantomguard/actions/workflows/ci.yml)

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

## New in 0.3: dependence, concentration, harness probes, more ratios

Four checks that catch what significance tests alone miss — each one distilled
from a real phantom autopsy:

**Cluster bootstrap** — your backtest says n=600 trades, but if one signal
fires across 5 correlated assets at the same timestamp, there are only 120
independent pieces of information. The IID bootstrap CI is then too narrow and
fakes significance:

```python
from phantomguard import diagnose_clustering
print(diagnose_clustering(pnl, entry_timestamps))
# widening factor 2.27x
# ! IID CI is anti-conservative ... do not base a significance claim on it
# ! SIGNIFICANCE FLIP: IID CI excludes 0 but the cluster CI does not
```

**Concentration check** — remove the single best day (or market, or event)
and see what is left. A strategy whose entire profit is one group's story is
an anecdote, not an edge. (Longshot strategies flag by design — see the
docstring for the honest interpretation.)

```python
from phantomguard import concentration_check
print(concentration_check(pnl, days))
# ! SIGN FLIP: remove group '2026-06-28' and the mean goes -0.118
```

**Look-ahead cheat probe** — wire an intentional crime into your pipeline
(the signal peeks one bar ahead) and demand the result EXPLODES. If it
doesn't, your harness is broken and every number it produced is void. The
smoke-detector test, with real smoke:

```python
from phantomguard import lookahead_cheat_probe
r = lookahead_cheat_probe(run_backtest)   # run_backtest(shift) -> metric
# harness ok: True   (leak +42.4 vs base +2.9 -- wiring proven correct)
```

**Sortino / Calmar / max drawdown** — Sharpe punishes upside volatility too,
which is unfair to positively-skewed strategies. `sortino_ratio` counts only
downside deviation; `calmar_ratio` is return per nightmare (CAGR / MaxDD).

**Decay check (0.5)** — the most common life cycle of a doomed edge: it existed,
it is fading, and you deploy into the corpse. `decay_check` splits the sample
into early/late halves and measures the time trend; the late half is the only
part that forecasts tomorrow.

**Cost ladder (0.5)** — `cost_ladder(pnl, costs=[0, 0.5, 1, 2])` recomputes the
(cluster-honest) CI under increasing per-trade costs and reports your
break-even cost. Compare it with your REAL spread+slippage+fees.

Runnable walkthroughs: `examples/cluster_demo.py`, `examples/concentration_demo.py`.

## New in 0.6: the oracle control — a negative without a power proof is not a finding

Everything above guards against the phantom **positive**. This release guards
against the opposite failure, the phantom **negative**: a pipeline so weak it
could never have detected a real edge — so its "nothing here" is a statement
about the pipeline, not about the market. The two failure modes need two
different controls:

| Control | Injects | Demands | Catches |
|---------|---------|---------|---------|
| `lookahead_cheat_probe` | a crime (future peek) | explosion | broken wiring that would **fake a positive** |
| `oracle_control` / `oracle_probe` | the answer | near-perfect detection | a setup too weak to produce a **real negative** |

**`oracle_control`** — for trade lists. Same inputs as `audit()` plus the mean
PnL/trade a real edge would produce. It injects that edge synthetically (the
sample keeps its real noise, tails and clustering; only the true mean changes)
and runs the exact same honest audit path on the copy. It also reports the
**MDE** — the minimal detectable effect at this n and clustering:

```python
from phantomguard import audit, oracle_control

print(oracle_control(pnl, claimed_effect=0.8, clusters=entry_ts, groups=days))
# OracleControl (synthetic-edge power probe)
#   observations      : 7531  (725 clusters)
#   claimed effect    : +0.8 per trade
#   injected cluster CI : [+0.48, +1.12]  -> DETECTED
#   MDE (this sample) : 0.319 per trade
#   power margin      : 2.51x
#   powered           : True

# or in one call -- audit() flags an UNPOWERED NEGATIVE automatically:
print(audit(pnl, clusters=entry_ts, groups=days, oracle=0.8))
```

Real-world use: a forensic re-audit of a prediction-market scanner. The trade
list audited hard negative — and that verdict was only *meaningful* because an
injected synthetic edge of the claimed size WAS detected on the same data
(powered, with 2.5x margin). Without that control, "no edge" and "this sample
couldn't tell" are indistinguishable.

**`oracle_probe`** — for model pipelines. Hand it your pipeline exactly as
configured (`fit_score(X, y) -> scores`); it appends the **target itself as a
feature** and demands a near-perfect score (default: AUC >= 0.95 for binary
targets, Spearman for continuous):

```python
from phantomguard import oracle_probe

r = oracle_probe(fit_score, X, y)
# powered=False -> the pipeline cannot see its OWN TARGET; every negative void
```

This catches a failure the cheat probe never can: a tree model whose
`min_child_samples` exceeds the positive count of a rare target reaches only
AUC ~0.57 *with the answer as a feature* — it structurally cannot split, and
every negative it ever produced was void. `oracle_probe` flags rare targets
(< 50 positives or < 2% base rate) explicitly.

On the CLI: `phantomguard audit trades.csv --pnl-col pnl --ts-col entry_ts
--oracle 0.8`. Walkthrough: `examples/oracle_demo.py`.

## One call to run them all: `audit()` (0.4)

Hand it per-trade PnL plus whatever labels you have; it runs the full
battery and aggregates every red flag. The verdict vocabulary is honest by
design: an audit can find guilt, never prove innocence -- the best you can
get is "no red flags", never "the edge is real".

```python
from phantomguard import audit  # or audit_csv("trades.csv", "pnl", "entry_ts", "day")

print(audit(pnl, clusters=entry_timestamps, groups=days))
# PhantomGuard audit -- verdict: NOT ESTABLISHED
#   trades            : 436
#   mean PnL          : +2.94          <- looks nice...
#   cluster 95% CI    : [-4.43, +10.40]  -> includes 0
#   clusters          : 181/436 independent timestamps (CI widening 1.63x)
#   red flags:
#     ! SIGN FLIP: remove group '2026-06-28' and the mean goes -2.55
#     ! honest CI includes zero -- not distinguishable from noise
```

Every omitted label silently disables the check that could have caught your
phantom -- so `audit()` flags missing labels too, and never issues a clean
bill on partial evidence.

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

`0.6.0` — core statistics, **PBO/CSCV**, a **CLI** (`check` / `pbo` / `audit`),
**cluster bootstrap**, **concentration check**, **decay check**, **cost ladder**,
**look-ahead cheat probe**, **oracle controls** (`oracle_control` /
`oracle_probe` — power proofs against phantom negatives), **Sortino/Calmar**
and the one-call **`audit()`** are in and tested (66 tests, CI on
Linux+Windows). Next: HTML report, backtester adapters (vectorbt/backtrader),
PyPI.
Issues and PRs welcome, especially additional phantom detectors and verifier
back-ends.

## License

MIT
