"""Command-line interface: ``phantomguard check`` and ``phantomguard pbo``.

    phantomguard check trades.csv --trials 800 --sr-trials trial_sharpes.csv -p 252
    phantomguard pbo trial_matrix.csv --blocks 16

CSV in, honest verdict out. Only numpy is required.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def _load(path, delimiter, skip_header):
    arr = np.genfromtxt(path, delimiter=delimiter, skip_header=skip_header)
    return np.atleast_1d(arr)


def _cmd_check(args) -> int:
    from .report import evaluate

    returns = _load(args.file, args.delimiter, args.skip_header)
    if returns.ndim > 1:
        returns = returns[:, args.column]

    sr_trials = None
    if args.sr_trials:
        sr_trials = _load(args.sr_trials, args.delimiter, args.skip_header).ravel()

    n_trials = args.trials
    if n_trials is None:
        n_trials = sr_trials.size if sr_trials is not None else 1

    fold_pnls = None
    if args.fold_pnls:
        fold_pnls = [float(x) for x in args.fold_pnls.split(",")]

    verdict = evaluate(
        returns, n_trials=n_trials, sr_trials=sr_trials,
        fold_pnls=fold_pnls, periods_per_year=args.periods_per_year,
    )
    print(verdict)
    return 0 if verdict.passed else 1


def _cmd_pbo(args) -> int:
    from .overfitting import pbo_cscv

    M = _load(args.file, args.delimiter, args.skip_header)
    if M.ndim != 2:
        print("error: pbo needs a 2-D matrix (rows = time, cols = strategies)",
              file=sys.stderr)
        return 2
    res = pbo_cscv(M, n_blocks=args.blocks)
    print(res)
    return 0 if res.pbo <= 0.5 else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="phantomguard",
        description="Honest backtest statistics and anti-overfitting guards.")
    p.add_argument("--delimiter", default=",", help="CSV delimiter (default ,)")
    p.add_argument("--skip-header", type=int, default=0,
                   help="header rows to skip (default 0)")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("check", help="run the full gate battery on a return series")
    c.add_argument("file", help="CSV of per-observation returns")
    c.add_argument("--column", type=int, default=0,
                   help="column index if the CSV has several (default 0)")
    c.add_argument("--trials", type=int, default=None,
                   help="cumulative number of configs tried (for Deflated Sharpe)")
    c.add_argument("--sr-trials", default=None,
                   help="CSV of the Sharpe ratios of all trials")
    c.add_argument("-p", "--periods-per-year", type=float, default=1.0,
                   help="e.g. 252 for daily bars (annualization)")
    c.add_argument("--fold-pnls", default=None,
                   help="comma-separated OOS fold PnLs, e.g. 0.1,-0.02,0.05")
    c.set_defaults(func=_cmd_check)

    b = sub.add_parser("pbo", help="Probability of Backtest Overfitting (CSCV)")
    b.add_argument("file", help="CSV matrix: rows = time, cols = strategies")
    b.add_argument("--blocks", type=int, default=16,
                   help="number of CSCV blocks S, even (default 16)")
    b.set_defaults(func=_cmd_pbo)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
