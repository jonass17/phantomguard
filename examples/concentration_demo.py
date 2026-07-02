"""Demo: the one-day wonder -- a healthy-looking backtest that is really
just one lucky day wearing a 10-day costume.

By construction: the strategy LOSES a little every day (true mean -0.05),
except day 3, where one fat market event hands it a windfall. The headline
average comes out positive anyway.

Run me:  python examples/concentration_demo.py
"""
import numpy as np

from phantomguard import concentration_check

rng = np.random.default_rng(7)

days = np.repeat(np.arange(10), 30)          # 10 days x 30 trades
pnl = rng.normal(-0.05, 0.5, 300)            # slightly negative everywhere...
pnl[days == 3] += 2.0                        # ...except one glorious day

print(f"Backtest says: n = {pnl.size} trades, mean PnL = {pnl.mean():+.4f}  <- looks fine!\n")

r = concentration_check(pnl, days)
print(r)

print()
print("Moral: a mean is not a strategy. Ask every backtest: 'and without")
print("your best day?' Try removing the windfall line (pnl[days == 3] += 2.0)")
print("and re-run -- the flags disappear, because now nothing is hidden.")
