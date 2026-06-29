"""Adversarial LLM verifier -- the part that makes PhantomGuard 2026, not 2014.

The statistics tell you *whether* a result is significant. They cannot tell you
*why* it might be wrong: a subtle look-ahead in feature engineering, a universe
that quietly survivorship-biases, a cost model that flatters fills. So we hand
the whole setup to an LLM with a skeptic's brief and ask it to REFUTE the edge.

Only findings it cannot break should count. This mirrors the manual
"adversarial verify agent" step every honest researcher already does -- packaged
so it runs every time, not just when you remember to be suspicious.

The ``anthropic`` package is an optional dependency; ``offline=True`` returns the
prompt without an API call so you can pipe it into any model.
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass

SKEPTIC_SYSTEM = (
    "You are an adversarial quantitative reviewer. Your sole job is to REFUTE "
    "the claim that the described strategy has a real, deployable edge. Assume "
    "it is overfit until proven otherwise. Hunt specifically for: look-ahead / "
    "data leakage, survivorship bias, marking entries/exits at unobtainable "
    "prices (D-1 mark), stale-book or stale-price phantoms, in-sample to "
    "out-of-sample collapse, multiple-testing inflation (was n_trials counted "
    "cumulatively?), nested/duplicate trades inflating n, and cost models that "
    "are too kind. When uncertain, default to 'not proven'. Be concrete and "
    "cite the specific number or design choice that worries you."
)

RESPONSE_INSTRUCTION = (
    "Respond as strict JSON with keys: "
    "`verdict` (one of 'refuted', 'survives', 'inconclusive'), "
    "`confidence` (0-1), "
    "`attacks` (array of {vector, severity, argument}), "
    "`what_would_convince_me` (string)."
)


def build_prompt(summary) -> str:
    """Turn a verdict/metrics object (or any dict/dataclass) into a skeptic brief."""
    if is_dataclass(summary):
        summary = asdict(summary)
    elif hasattr(summary, "metrics"):
        summary = {"metrics": summary.metrics,
                   "reasons": getattr(summary, "reasons", []),
                   "warnings": getattr(summary, "warnings", [])}
    body = json.dumps(summary, indent=2, default=str)
    return (f"Here is a backtest result and its design notes:\n\n{body}\n\n"
            f"{RESPONSE_INSTRUCTION}")


def adversarial_verify(summary, model: str = "claude-sonnet-4-6",
                       api_key: str | None = None, offline: bool = False,
                       max_tokens: int = 1500):
    """Ask an LLM to try to refute the edge. Returns the parsed JSON dict.

    With ``offline=True`` (or no ``anthropic`` installed) it returns
    ``{"verdict": "skipped", "prompt": ...}`` so you can run the prompt yourself.
    """
    prompt = build_prompt(summary)
    if offline:
        return {"verdict": "skipped", "system": SKEPTIC_SYSTEM, "prompt": prompt}

    try:
        import anthropic
    except ImportError:
        return {"verdict": "skipped", "system": SKEPTIC_SYSTEM, "prompt": prompt,
                "note": "pip install anthropic to enable live verification"}

    client = anthropic.Anthropic(api_key=api_key)  # uses ANTHROPIC_API_KEY if None
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SKEPTIC_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(getattr(b, "text", "") for b in msg.content)
    try:
        start, end = text.find("{"), text.rfind("}")
        return json.loads(text[start:end + 1])
    except (ValueError, json.JSONDecodeError):
        return {"verdict": "inconclusive", "raw": text}
