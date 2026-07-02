"""Harness probes: prove that your backtest CAN detect an edge before you
trust it saying there is none (or one).

The look-ahead cheat probe wires an intentional crime into the pipeline --
the signal gets to peek one bar into the future -- and demands that the
result EXPLODES. A peeking signal is close to omniscient; if your harness
does not reward it massively, the harness is broken (signal ignored, join
misaligned, outcomes shuffled) and every number it ever produced is void.
This is the smoke-detector test with real smoke.

The same probe run with a one-bar DELAY answers the opposite question: how
much of the measured edge survives realistic latency? An edge that dies one
bar later is an HFT claim, not a strategy you can trade.

Note: only you can implement the shifting, because only you know where the
signal enters your pipeline. You hand this module a callable; it handles the
comparison and the verdict.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class HarnessProbeResult:
    """Verdict of the look-ahead cheat probe."""
    metric_base: float            # run_backtest(0): the honest run
    metric_leak: float            # run_backtest(-1): signal peeks 1 bar ahead
    metric_delay: Optional[float] # run_backtest(+1): signal delayed 1 bar
    harness_ok: bool              # True -> the leak exploded as it must
    fragile_to_delay: Optional[bool]
    notes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "HarnessProbeResult",
            f"  base   (shift  0) : {self.metric_base:+.6g}",
            f"  leak   (shift -1) : {self.metric_leak:+.6g}",
        ]
        if self.metric_delay is not None:
            lines.append(f"  delay  (shift +1) : {self.metric_delay:+.6g}")
        lines.append(f"  harness ok        : {self.harness_ok}")
        lines += [f"  ! {n}" for n in self.notes]
        return "\n".join(lines)


def lookahead_cheat_probe(
    run_backtest: Callable[[int], float],
    explode_factor: float = 2.0,
    min_improvement: float | None = None,
    run_delay: bool = True,
) -> HarnessProbeResult:
    """Run the backtest honestly, cheating, and delayed -- and judge the harness.

    Parameters
    ----------
    run_backtest : callable
        ``run_backtest(shift) -> metric`` where ``shift`` is applied to the
        signal relative to the outcome: ``0`` = your normal backtest,
        ``-1`` = the signal sees one bar of the FUTURE (the crime),
        ``+1`` = the signal arrives one bar LATE. Larger metric = better
        (mean PnL, Sharpe, hit rate ...). You implement the shift; this
        function implements the judgment.
    explode_factor : float
        The leak run must beat ``explode_factor * |base|`` (and the base
        itself) to count as an explosion.
    min_improvement : float, optional
        Absolute floor for (leak - base). REQUIRED in spirit when your base
        metric is near zero, where the relative test is meaningless: set it
        to a value you would call an obvious edge (e.g. 10x your costs).
    run_delay : bool
        Also run ``shift=+1`` and report edge fragility to one bar of latency.

    Judgment
    --------
    ``harness_ok=False`` means: do not trust ANY result from this pipeline,
    positive or negative, until the wiring is fixed.
    """
    base = float(run_backtest(0))
    leak = float(run_backtest(-1))

    notes = []
    improvement = leak - base
    exploded = leak > base and abs(leak) > explode_factor * abs(base)
    if min_improvement is not None:
        exploded = exploded and improvement >= min_improvement
    elif abs(base) < 1e-9:
        # Relative test is meaningless around zero; be conservative.
        exploded = improvement > 0
        notes.append(
            "base metric ~ 0: relative explosion test is weak -- "
            "pass min_improvement (e.g. 10x costs) for a strict verdict"
        )

    if not exploded:
        notes.append(
            "LEAK DID NOT EXPLODE: a signal that sees the future should crush "
            "this metric. The harness is likely broken (signal ignored, "
            "misaligned join, shuffled outcomes) -- all its results are void"
        )

    metric_delay = None
    fragile = None
    if run_delay:
        metric_delay = float(run_backtest(+1))
        if np.isfinite(base) and base > 0:
            fragile = (base - metric_delay) > 0.5 * abs(base)
            if fragile:
                notes.append(
                    f"edge is latency-fragile: one bar of delay destroys "
                    f"{(base - metric_delay) / abs(base):.0%} of it -- "
                    f"an execution claim, not a strategy claim"
                )

    return HarnessProbeResult(
        metric_base=base,
        metric_leak=leak,
        metric_delay=metric_delay,
        harness_ok=bool(exploded),
        fragile_to_delay=fragile,
        notes=notes,
    )
