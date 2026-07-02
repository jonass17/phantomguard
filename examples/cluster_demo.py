"""Demo: how dependent trades fake significance -- and how the cluster
bootstrap catches it.

Scenario (synthetic, but a very common real-world shape):
A signal fires on 120 timestamps. Each time it fires, you trade 5 correlated
assets at once. Your backtest report says "n=600 trades" -- but there are only
120 independent pieces of information. The strategy has NO real edge
(true mean = 0 by construction).

Run me:  python examples/cluster_demo.py
"""
import numpy as np

from phantomguard import diagnose_clustering

rng = np.random.default_rng(2026)

N_TIMESTAMPS = 120
ASSETS_PER_SIGNAL = 5

# One shared outcome per timestamp (the market move) ...
shared = rng.normal(0.0, 1.0, N_TIMESTAMPS)          # true mean = 0: no edge!
# ... worn by 5 assets with only tiny individual differences.
pnl = np.repeat(shared, ASSETS_PER_SIGNAL) + rng.normal(0, 0.05, N_TIMESTAMPS * ASSETS_PER_SIGNAL)
timestamps = np.repeat(np.arange(N_TIMESTAMPS), ASSETS_PER_SIGNAL)

print(f"Backtest says: n = {pnl.size} trades, mean PnL = {pnl.mean():+.4f}\n")

d = diagnose_clustering(pnl, timestamps, seed=1)
print(d)

print()
if d.iid_ci[0] > 0 or d.iid_ci[1] < 0:
    print(">>> The IID interval claims significance -- on a strategy with ZERO true edge.")
if d.cluster_ci[0] <= 0 <= d.cluster_ci[1]:
    print(">>> The cluster interval correctly says: not distinguishable from noise.")
print("\nMoral: count timestamps, not trades. Try changing ASSETS_PER_SIGNAL")
print("to 1 (independent trades) or 20 (extreme duplication) and re-run.")
