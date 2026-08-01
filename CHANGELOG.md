# Changelog

## 0.6.0 — 2026-08-01

**The oracle control: a negative without a power proof is not a finding.**

Everything so far guarded against the phantom POSITIVE (noise, leakage,
concentration faking an edge). 0.6 adds the guard against the phantom
NEGATIVE: a setup too weak to have detected a real edge, whose "nothing
here" is a statement about the pipeline, not the market.

- New module `phantomguard.oracle`:
  - `oracle_control(pnl, claimed_effect, clusters=, groups=)` — injects a
    synthetic edge of the claimed size (sample centered, real noise/tails/
    clustering preserved) and runs the same honest audit path on the copy;
    reports `powered`, the exact **MDE** (minimal detectable effect at this
    n/clustering — exact via bootstrap shift linearity, not a grid
    approximation), the power margin, and an effect sweep.
  - `oracle_probe(fit_score, X, y, threshold=0.95)` — appends the target
    itself as a feature to the user's pipeline and demands a near-perfect
    score (AUC for binary targets, Spearman for continuous). Flags rare
    targets (< 50 positives / < 2% base rate) where min-samples-per-leaf
    constraints make tree models structurally blind. No new dependencies;
    the pipeline is passed as a callable.
- `audit(..., oracle=<claimed effect>)` runs the control alongside the
  audit; an unpowered negative gets an explicit `UNPOWERED NEGATIVE` red
  flag. A `NOT ESTABLISHED` report without an oracle run now points to the
  missing power proof.
- CLI: `phantomguard audit ... --oracle EFFECT`.
- `examples/oracle_demo.py`; 14 new tests (66 total).
- `pyproject.toml` version had drifted (0.2.0 while the package was 0.5.0);
  both now read 0.6.0.

## 0.5.0

- Decay check (`decay_check`) and cost ladder (`cost_ladder`), wired into
  `audit()`.
- Fix: `audit()` degrades gracefully when all trades share one group.

## 0.4.1

- `phantomguard audit` in the CLI.

## 0.4.0

- One-call `audit()` / `audit_csv()` over a trade list with an honest,
  asymmetric verdict vocabulary (`NOT ESTABLISHED` / `SUSPECT` /
  `NO RED FLAGS`).

## 0.3.0

- Cluster bootstrap (`diagnose_clustering`), concentration check,
  look-ahead cheat probe (`lookahead_cheat_probe`), Sortino/Calmar/max
  drawdown.

## 0.2.0

- PBO/CSCV (`pbo_cscv`) and the CLI (`phantomguard check` / `pbo`).

## 0.1.0

- Honest backtest statistics: PSR, Deflated Sharpe, MinTRL, bootstrap
  Sharpe CI, walk-forward and purged K-Fold splits, phantom detectors,
  adversarial LLM verifier.
