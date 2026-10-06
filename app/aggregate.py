"""Combine N independent judge runs into one verdict. Pure logic, no I/O.

Rules (never force an answer when runs disagree):
  confident - every planned run succeeded, all chose the same result, none
              reported low confidence, and at least 2 runs (so both A/B orders
              were seen).
  review    - a strict majority of *planned* runs chose the same result.
              The verdict is shown as a leaning, for a human to check.
  unclear   - anything else (e.g. a 2-2 split). No verdict.
  error     - no run succeeded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

Letter = Literal["A", "B"]
Status = Literal["confident", "review", "unclear", "error"]
CONF_RANK = {"high": 2, "medium": 1, "low": 0}


@dataclass
class RunResult:
    index: int
    swapped: bool  # True when B was shown to the model before A
    verdict: Optional[Letter] = None
    confidence: Optional[str] = None
    judgment: Optional[dict] = None
    error: Optional[str] = None
    cached: bool = False
    usage: Optional[dict] = None
    model: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.verdict in ("A", "B")

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "swapped": self.swapped,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "judgment": self.judgment,
            "error": self.error,
            "cached": self.cached,
            "usage": self.usage,
            "model": self.model,
        }


@dataclass
class Aggregate:
    status: Status
    verdict: Optional[Letter]
    votes: dict = field(default_factory=dict)
    planned: int = 0
    succeeded: int = 0
    representative: Optional[int] = None  # index of the run whose details we display
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "verdict": self.verdict,
            "votes": self.votes,
            "planned": self.planned,
            "succeeded": self.succeeded,
            "representative": self.representative,
            "explanation": self.explanation,
        }


def aggregate(runs: list[RunResult], planned: Optional[int] = None) -> Aggregate:
    planned = planned if planned is not None else len(runs)
    ok = [r for r in runs if r.ok]
    votes = {"A": sum(r.verdict == "A" for r in ok), "B": sum(r.verdict == "B" for r in ok)}
    failed = planned - len(ok)

    if not ok:
        errors = sorted({r.error for r in runs if r.error})
        return Aggregate("error", None, votes, planned, 0, None,
                         "All runs failed: " + ("; ".join(errors) or "no runs"))

    top: Letter = "A" if votes["A"] >= votes["B"] else "B"
    top_count = votes[top]
    unanimous = top_count == len(ok)
    low = [r for r in ok if r.confidence == "low"]

    if unanimous and failed == 0 and not low and len(ok) >= 2:
        status: Status = "confident"
        explanation = f"All {planned} runs chose {top} with medium or high confidence, in both image orders."
    elif top_count * 2 > planned and votes["A"] != votes["B"]:
        status = "review"
        why = []
        if not unanimous:
            why.append(f"{votes['B' if top == 'A' else 'A']} run(s) disagreed")
        if low:
            why.append(f"{len(low)} run(s) reported low confidence")
        if failed:
            why.append(f"{failed} run(s) failed")
        if len(ok) < 2:
            why.append("only one run, so image order was not checked")
        explanation = f"{top_count}/{planned} runs chose {top}, but " + ", ".join(why) + "."
    else:
        status = "unclear"
        explanation = f"Runs did not agree (A: {votes['A']}, B: {votes['B']}, failed: {failed} of {planned})."

    verdict = top if status in ("confident", "review") else None
    rep_pool = [r for r in ok if verdict is None or r.verdict == verdict]
    rep = max(rep_pool, key=lambda r: (CONF_RANK.get(r.confidence or "", -1), -r.index))
    return Aggregate(status, verdict, votes, planned, len(ok), rep.index, explanation)
