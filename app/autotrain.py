"""Gated auto-training: new lessons are only switched on after they prove themselves.

Each correction you give writes a *candidate* lessons version that is NOT used for judging. Once enough
new corrections have piled up, the candidate and the current (active) lessons are both run on tasks the
candidate was not learned from. The candidate becomes active only if it gets clearly more of them right
without making more confident mistakes. Otherwise it is rejected and the current lessons stay.
"""

from __future__ import annotations

import hashlib
import math
import time
from typing import Optional

from .benchmark import run_tasks
from .config import JudgeConfig, settings
from .learn import SetError, set_tasks

MY_TASKS_SET = "My tasks"  # every evaluation you mark is filed here; it is the pool of labelled answers


def _score(items: list[dict]) -> dict:
    scored = [i for i in items if i.get("label") in ("A", "B")]
    return {
        "tasks": len(scored),
        "right": sum(1 for i in scored if i.get("verdict") == i["label"]),
        "confident_wrong": sum(1 for i in scored if i.get("status") == "confident" and i.get("verdict") != i["label"]),
        "unclear": sum(1 for i in scored if i.get("verdict") is None),
    }


def pick_pool(tasks: list, learned_from: set[int], max_tasks: int) -> list:
    """Labelled tasks the candidate was not learned from, in a fixed pseudo-random order, capped for cost."""
    unseen = [t for t in tasks if t.set_task_id not in learned_from]
    unseen.sort(key=lambda t: hashlib.sha1(str(t.set_task_id).encode()).hexdigest())
    return unseen[:max_tasks]


def pending_corrections(candidate: dict, active: Optional[dict]) -> int:
    """How many tasks the candidate has learned from that the active lessons have not."""
    return len(set(candidate["learned_from"]) - set(active["learned_from"] if active else []))


def decide(baseline: dict, candidate: dict, margin: float) -> tuple[str, str]:
    """('accepted' | 'rejected', plain-English reason)."""
    n = baseline["tasks"]
    needed = max(1, math.ceil(margin * n))
    gain = candidate["right"] - baseline["right"]
    if candidate["confident_wrong"] > baseline["confident_wrong"]:
        return "rejected", (f"It made more confident mistakes ({candidate['confident_wrong']} vs "
                            f"{baseline['confident_wrong']}).")
    if gain < needed:
        return "rejected", (f"It got {candidate['right']} right against {baseline['right']} for the current lessons; "
                            f"it needed at least {needed} more.")
    return "accepted", (f"It got {candidate['right']} right against {baseline['right']} for the current lessons, "
                        f"with no extra confident mistakes.")


async def gate_candidate(db, candidate_id: int, config: JudgeConfig, judge, use_cache: bool = True) -> dict:
    """Test one candidate against the active lessons on unseen tasks, then accept or reject it."""
    cand = db.get_knowledge(candidate_id)
    if not cand or cand["status"] != "candidate":
        raise ValueError("That lessons version is not waiting for a test.")
    active = db.active_knowledge()

    set_id = next((s["id"] for s in db.list_sets() if s["name"] == MY_TASKS_SET), None)
    try:
        _, tasks = set_tasks(db, set_id, labeled_only=True) if set_id else (None, [])
    except SetError:
        tasks = []
    pool = pick_pool(tasks, set(cand["learned_from"]), settings.gate_max_tasks)

    if len(pool) < settings.gate_min_pool:
        result = {"decision": "waiting", "tested_at": time.time(), "pool": len(pool),
                  "needed": settings.gate_min_pool,
                  "reason": f"Only {len(pool)} labelled tasks it was not learned from; it needs "
                            f"{settings.gate_min_pool} to test fairly. Keep labelling answers."}
        db.set_gate_result(candidate_id, result)
        return result

    base_cfg = config.with_knowledge(active)
    cand_cfg = config.with_knowledge(cand)
    base = await run_tasks(pool, f"gate: current lessons vs #{candidate_id}", base_cfg, judge, db,
                           use_cache=use_cache, note=f"gate baseline for lessons #{candidate_id}",
                           mode="gate", set_id=set_id)
    new = await run_tasks(pool, f"gate: candidate #{candidate_id}", cand_cfg, judge, db,
                          use_cache=use_cache, note=f"gate candidate #{candidate_id}", mode="gate",
                          set_id=set_id, baseline_of=base.run_id)
    b, c = _score(base.items), _score(new.items)
    decision, why = decide(b, c, settings.gate_margin)
    result = {"decision": decision, "reason": why, "tested_at": time.time(), "pool": len(pool),
              "baseline": b, "candidate": c, "baseline_run": base.run_id, "candidate_run": new.run_id,
              "baseline_lessons_id": active["id"] if active else None}
    db.set_knowledge_status(candidate_id, decision, result)
    if decision == "accepted":
        db.activate_knowledge(candidate_id)
    return result
