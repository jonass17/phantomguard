"""Run me:  python examples/quickstart.py

Shows PhantomGuard rejecting a phantom and accepting a (synthetic) real edge.
"""
import numpy as np

from phantomguard import evaluate, adversarial_verify

rng = np.random.default_rng(0)

# --- 1) A phantom: looks fine on one run, but it's 1 of 800 trials ----------
phantom = rng.normal(0.0005, 0.01, 500)
sr_trials = rng.normal(0, 0.05, 800)
v1 = evaluate(phantom, n_trials=800, sr_trials=sr_trials,
              fold_pnls=[0.10, -0.02, 0.05, 0.08, -0.01], periods_per_year=252)
print("=== candidate A (1 of 800 trials) ===")
print(v1, "\n")

# --- 2) A genuinely strong, consistent, lightly-searched edge ---------------
real = rng.normal(0.0015, 0.008, 1200)
v2 = evaluate(real, n_trials=5, sr_variance=0.01,
              fold_pnls=[0.12, 0.05, 0.09, 0.07, 0.11], periods_per_year=252)
print("=== candidate B (5 trials, consistent folds) ===")
print(v2, "\n")

# --- 3) Adversarial verifier (offline: prints the skeptic prompt) -----------
print("=== adversarial verifier prompt for candidate B ===")
print(adversarial_verify(v2, offline=True)["prompt"][:600], "...")
