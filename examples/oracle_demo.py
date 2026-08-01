"""Oracle controls: a negative without a power proof is not a finding.

Two demos, one per failure mode:

1. oracle_control -- a trade list whose audit says "NOT ESTABLISHED".
   Is that a real negative, or could this sample never have detected
   anything? Inject a synthetic edge of the claimed size and see whether
   the same audit path catches it.

2. oracle_probe -- a model pipeline that reports "no signal". Feed it the
   answer (the target itself as a feature): if it still cannot score, the
   pipeline is structurally blind and every negative it produced is void.

Run: python examples/oracle_demo.py
"""
import numpy as np

from phantomguard import audit, oracle_control, oracle_probe

rng = np.random.default_rng(0)

# --- 1a. a POWERED negative: big clean sample, no edge --------------------
pnl = rng.normal(0.0, 1.0, 3000)                 # pure noise, no edge
days = np.arange(3000) // 150

print("=" * 70)
print("1a. Large clean sample -- the negative verdict is a real finding")
print("=" * 70)
print(audit(pnl, clusters=np.arange(3000), groups=days, oracle=0.5))
print()

# --- 1b. an UNPOWERED negative: 200 trades, 5 co-firing clusters ----------
shared = rng.normal(0.0, 1.0, 5)
pnl_bad = np.repeat(shared, 40) + rng.normal(0, 0.05, 200)
ts = np.repeat(np.arange(5), 40)

print("=" * 70)
print("1b. Brutal clustering -- the SAME verdict, but now it means nothing")
print("=" * 70)
print(oracle_control(pnl_bad, claimed_effect=0.1, clusters=ts, groups=ts))
print()

# --- 2. model pipeline: can it see its own target? ------------------------
X = rng.normal(0, 1, (500, 4))
y = (rng.random(500) < 0.05).astype(float)       # rare target: 5% positives

print("=" * 70)
print("2a. A working pipeline scores AUC 1.0 with the target as a feature")
print("=" * 70)
print(oracle_probe(lambda X_, y_: X_[:, -1], X, y))
print()

print("=" * 70)
print("2b. A blind pipeline (cannot split -> constant score) FAILS the probe")
print("=" * 70)
# The real-world version of this: a tree model whose min_child_samples
# exceeds the positive count -- it cannot split even on the answer.
print(oracle_probe(lambda X_, y_: np.zeros(len(X_)), X, y))
