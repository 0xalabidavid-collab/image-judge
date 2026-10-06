"""Train / test / judge-only runs over saved task sets.

"Training" does not change the model's weights - nothing here can. It works like a
person keeping notes:

  1. Judge every labelled task in the training set with the current knowledge.
  2. Pick the tasks it got wrong or was unsure about.
  3. For each one, show the model the images, the correct answer and its own
     reasoning, and ask for general rules that would have got it right.
  4. Merge those rules with the existing lessons into one short list.
  5. Save that as a new knowledge version. Every later judgment reads it.

Testing judges a set without showing it the answers and scores it against the
labels (given up front, or added afterwards). Because lessons are only ever
written from training sets, a test score is a fair, held-out measurement.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .benchmark import BenchmarkOutcome, DatasetTask, compute_metrics, load_task_images, run_tasks
from .config import JudgeConfig
from .judge import JudgeError, api_semaphore
from .storage import image_path

MAX_LESSONS = 12

LESSON_SYSTEM = """\
You are improving an automated judge. The judge sees an image-generation or image-editing
PROMPT, the ORIGINAL images, and two candidate outputs, RESULT A and RESULT B, and decides
which result is the correct output. Exactly one is correct. A human expert who labels these
tasks has given the correct answer.

You will see one task the judge got wrong or was unsure about: the images, the CORRECT
answer, and the judge's own reasoning.

1. what_judge_missed - Look at the images yourself. In one or two sentences, name the
   concrete visual fact that makes the correct result correct (or the other one wrong) and
   that the judge missed, misread, or gave too little weight.

2. lessons - At most one or two general rules that would have led the judge to the correct
   answer here AND would help on other tasks of the same kind. Return NO lessons when the
   judge's instructions already cover the case (it simply misread the image), or when the
   answer depends on details too specific to this one task to generalise. Fewer, well-calibrated
   rules beat many rules: every extra rule is read on every future task and can pull other
   verdicts the wrong way. A good rule:
   - says what to check and how much it usually weighs, in terms of how serious and visible the
     flaw is, e.g. "On text tasks, read the rendered letters one by one in both results; a
     misspelling a viewer would notice is a major flaw, usually worse than a slightly different font";
   - never claims one criterion always wins ("always", "never", "regardless of", "only a
     tiebreaker"): the judge weighs everything together, and absolute rules from one task end up
     contradicting other tasks;
   - never mentions this task's specific content (no "the red car", "the woman on the left");
   - is consistent with how the judge is told to weigh things (see its instructions below):
     judge each image as a whole and weigh how serious and how visible each flaw is. If the correct
     answer shows the labeller weighs something differently, state that preference plainly.
   - adds something the judge's instructions don't already say; don't repeat them.

3. label_seems_wrong - true if, after looking carefully, you believe the given correct answer
   is mistaken or the task is genuinely ambiguous. In that case return no lessons: a wrong
   label must not become a rule.
"""

LESSON_SCHEMA = {
    "type": "object",
    "properties": {
        "what_judge_missed": {"type": "string"},
        "label_seems_wrong": {"type": "boolean"},
        "lessons": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["what_judge_missed", "label_seems_wrong", "lessons"],
    "additionalProperties": False,
}

MERGE_SYSTEM = f"""\
You maintain the list of lessons that an image-comparison judge reads before every task.
Each lesson was learned from a task the judge got wrong. Merge the CURRENT lessons and the
NEW candidate lessons into one short list:
- merge duplicates and near-duplicates into one clearer rule;
- rewrite rules tied to one specific image so they apply generally, or drop them;
- the judge's instructions (given below) are the authority: drop or soften any rule that
  contradicts them, repeats them, or claims one criterion always wins ("always", "never",
  "regardless", "only a tiebreaker") - rewrite it in terms of how serious and visible a flaw is;
- if two rules pull in opposite directions, reconcile them into one rule that says when each
  applies, or keep the one with more support;
- at most {MAX_LESSONS} rules, most important first, each one or two sentences. Fewer is better.
In `changes`, say briefly what you added, merged, softened or dropped.
"""

MERGE_SCHEMA = {
    "type": "object",
    "properties": {
        "lessons": {"type": "array", "items": {"type": "string"}},
        "changes": {"type": "string"},
    },
    "required": ["lessons", "changes"],
    "additionalProperties": False,
}


class SetError(ValueError):
    pass


def set_tasks(db, set_id: int, labeled_only: bool = False) -> tuple[dict, list[DatasetTask]]:
    s = db.get_set(set_id)
    if not s:
        raise SetError(f"No task set with id {set_id}")
    tasks = []
    for t in s["tasks"]:
        if labeled_only and t["label"] not in ("A", "B"):
            continue
        img = t["images"]
        tasks.append(DatasetTask(
            id=f"{t['id']:05d}",
            prompt=t["prompt"],
            originals=[image_path(p) for p in img["originals"]],
            a=image_path(img["a"]),
            b=image_path(img["b"]),
            label=t["label"],
            set_task_id=t["id"],
        ))
    if not tasks:
        raise SetError(f"Set '{s['name']}' has no {'labelled ' if labeled_only else ''}tasks")
    return s, tasks


def refresh_labels(db, run: dict) -> dict:
    """Re-score a set run with the set's current labels (they can be added after the run)."""
    if not run.get("set_id") or not run.get("items"):
        return run
    for item in run["items"]:
        task = db.get_set_task(item["set_task_id"]) if item.get("set_task_id") else None
        if task is not None:
            item["label"] = task["label"]
            item["correct"] = (None if item["verdict"] is None or task["label"] not in ("A", "B")
                               else item["verdict"] == task["label"])
    if run["status"] == "done":
        metrics = compute_metrics(run["items"])
        if run.get("metrics") and "training" in run["metrics"]:
            metrics["training"] = run["metrics"]["training"]
        run["metrics"] = metrics
        db.update_benchmark_metrics(run["id"], metrics)
    return run


def pick_learning_cases(items: list[dict], max_cases: int) -> list[dict]:
    """Mistakes first (confident-and-wrong is the most important), then unsure-but-right."""
    def rank(i: dict) -> Optional[int]:
        if i["status"] == "confident" and i["correct"]:
            return None  # nothing to learn
        if i["status"] == "error" or not i.get("runs"):
            return None  # no judgment to learn from (error, or A and B were identical)
        if i["correct"] is False:
            return {"confident": 0, "review": 1}.get(i["status"], 2)
        if i["status"] == "unclear":
            return 2
        return 3  # review and right: right answer, but not sure enough
    ranked = [(r, i["task_id"], i) for i in items if (r := rank(i)) is not None]
    return [i for _, _, i in sorted(ranked, key=lambda x: (x[0], x[1]))][:max_cases]


def _case_text(item: dict) -> str:
    votes = item.get("votes") or {}
    lines = [
        f"CORRECT ANSWER (from the human labeller): Result {item['label']}",
        f"The judge's verdict: {item['verdict'] or 'no verdict'} ({item['status']}; "
        f"votes A {votes.get('A', 0)}, B {votes.get('B', 0)})",
    ]
    if item.get("decisive_difference"):
        lines.append(f"The judge's decisive difference: {item['decisive_difference']}")
    if item.get("reasoning"):
        lines.append(f"The judge's reasoning: {item['reasoning']}")
    rep = next((r for r in item.get("runs", []) if r.get("failed_requirements")), None)
    if rep:
        lines.append("Requirements where the judge marked A and B differently:")
        lines += [f"- {q['description']} ({q.get('severity', 'unspecified')}): A {q['a']}, B {q['b']}" for q in rep["failed_requirements"]]
    if item.get("user_reason"):
        lines.append(
            "THE LABELLER'S OWN EXPLANATION (the human who knows the answer wrote this; it is the most "
            "important input - base the lessons on it, and set label_seems_wrong to false):\n"
            + item["user_reason"].strip())
    return "\n".join(lines)


async def write_lessons(item: dict, task: DatasetTask, config: JudgeConfig, judge) -> dict:
    images = await asyncio.to_thread(load_task_images, task)
    content: list = [f"PROMPT:\n{task.prompt.strip()}"]
    for i, o in enumerate(images.originals, 1):
        content += [f"ORIGINAL {i}:", o]
    content += ["RESULT A:", images.a, "RESULT B:", images.b, _case_text(item),
                "Write the lessons. Respond with the JSON object only."]
    from .judge import judge_system_prompt
    system = (LESSON_SYSTEM + "\nFOR REFERENCE - THE JUDGE'S CURRENT INSTRUCTIONS:\n<<<\n"
              + judge_system_prompt(config) + "\n>>>\n")
    async with api_semaphore():
        out = await judge.complete_json(system, content, LESSON_SCHEMA, config)
    data = out["data"]
    return {
        "task_id": item["task_id"],
        "prompt": task.prompt,
        "label": item["label"],
        "verdict": item["verdict"],
        "status": item["status"],
        "what_judge_missed": data.get("what_judge_missed", ""),
        "label_seems_wrong": bool(data.get("label_seems_wrong")),
        # A labeller who explained their answer has the final say; otherwise a doubted label isn't learned from.
        "lessons": ([] if data.get("label_seems_wrong") and not item.get("user_reason")
                    else [l for l in data.get("lessons", []) if l.strip()]),
    }


async def merge_lessons(current: list[str], new: list[str], config: JudgeConfig, judge) -> tuple[list[str], str]:
    if not new:
        return list(current), "No new lessons."
    content = [
        "CURRENT LESSONS:\n" + ("\n".join(f"- {l}" for l in current) if current else "(none yet)"),
        "NEW CANDIDATE LESSONS:\n" + "\n".join(f"- {l}" for l in new),
        "Return the merged list. Respond with the JSON object only.",
    ]
    from .judge import judge_system_prompt
    system = (MERGE_SYSTEM + "\nTHE JUDGE'S INSTRUCTIONS (without lessons):\n<<<\n"
              + judge_system_prompt(config.with_knowledge(None)) + "\n>>>\n")
    async with api_semaphore():
        out = await judge.complete_json(system, content, MERGE_SCHEMA, config)
    lessons = [l.strip() for l in out["data"].get("lessons", []) if l.strip()][:MAX_LESSONS]
    return lessons, out["data"].get("changes", "")


@dataclass
class TrainOutcome:
    run: BenchmarkOutcome
    knowledge_id: Optional[int]
    lessons: list[str] = field(default_factory=list)
    cases: list[dict] = field(default_factory=list)
    changes: str = ""


async def train(
    db,
    set_id: int,
    config: JudgeConfig,
    judge,
    use_cache: bool = True,
    max_cases: int = 15,
    note: str = "",
    on_progress: Optional[Callable[[int, int, dict], None]] = None,
    run_id: Optional[int] = None,
) -> TrainOutcome:
    """Judge a labelled set with the current knowledge, learn from the mistakes, save a new version."""
    s, tasks = set_tasks(db, set_id, labeled_only=True)
    base = db.get_knowledge(config.knowledge_id) if config.knowledge_id else None
    outcome = await run_tasks(tasks, f"set: {s['name']}", config, judge, db, use_cache=use_cache, note=note,
                              on_progress=on_progress, run_id=run_id, mode="train", set_id=set_id,
                              finish=False)
    by_id = {t.id: t for t in tasks}
    cases = pick_learning_cases(outcome.items, max_cases)

    results = await asyncio.gather(*(write_lessons(c, by_id[c["task_id"]], config, judge) for c in cases),
                                   return_exceptions=True)
    reviewed, errors = [], []
    for r in results:
        if isinstance(r, JudgeError) and r.fatal:
            db.finish_benchmark(outcome.run_id, outcome.metrics, status="failed")
            raise r
        (errors if isinstance(r, Exception) else reviewed).append(r)
    new = [l for r in reviewed for l in r["lessons"]]

    current = list(base["lessons"]) if base else []
    try:
        lessons, changes = await merge_lessons(current, new, config, judge)
    except JudgeError as exc:
        db.finish_benchmark(outcome.run_id, outcome.metrics, status="failed")
        raise exc

    knowledge_id = config.knowledge_id
    if new:
        knowledge_id = db.add_knowledge(
            guidelines=base["guidelines"] if base else "",
            lessons=lessons,
            source=f"Trained on '{s['name']}' (run #{outcome.run_id}): {len(reviewed)} tasks reviewed, "
                   f"{len(new)} new lessons",
            parent_id=config.knowledge_id,
            activate=True,
            learned_from=list(base["learned_from"] if base else []) + [int(r["task_id"]) for r in reviewed],
        )
        db.set_learned_knowledge(outcome.run_id, knowledge_id)

    metrics = dict(outcome.metrics)
    metrics["training"] = {
        "cases_reviewed": len(reviewed),
        "case_errors": [f"{type(e).__name__}: {e}" for e in errors],
        "new_lessons": len(new),
        "lessons_after": len(lessons),
        "changes": changes,
        "knowledge_before": config.knowledge_id,
        "knowledge_after": knowledge_id,
        "flagged_labels": [r["task_id"] for r in reviewed if r["label_seems_wrong"]],
        "cases": reviewed,
    }
    db.finish_benchmark(outcome.run_id, metrics)
    outcome.metrics = metrics
    return TrainOutcome(outcome, knowledge_id, lessons, reviewed, changes)


async def run_set(
    db,
    set_id: int,
    mode: str,
    config: JudgeConfig,
    judge,
    use_cache: bool = True,
    compare_baseline: bool = False,
    note: str = "",
    on_progress: Optional[Callable[[int, int, dict], None]] = None,
    run_id: Optional[int] = None,
    baseline_run_id: Optional[int] = None,
) -> tuple[BenchmarkOutcome, Optional[BenchmarkOutcome]]:
    """Test (score against labels) or judge-only. Optionally also run without knowledge to compare."""
    s, tasks = set_tasks(db, set_id)
    main = await run_tasks(tasks, f"set: {s['name']}", config, judge, db, use_cache=use_cache, note=note,
                           on_progress=on_progress, run_id=run_id, mode=mode, set_id=set_id)
    baseline = None
    if compare_baseline and (config.knowledge_id or config.lessons or config.guidelines):
        baseline = await run_tasks(tasks, f"set: {s['name']}", config.with_knowledge(None), judge, db,
                                   use_cache=use_cache, note=f"untrained comparison for run #{main.run_id}",
                                   run_id=baseline_run_id, mode=mode, set_id=set_id, baseline_of=main.run_id)
    return main, baseline


def luck_chance(better: int, worse: int) -> float:
    """Chance of a split at least this lopsided if lessons made no real difference.

    Exact one-sided sign test: each task where the two runs disagree is treated as
    a fair coin flip under "no real effect".
    """
    from math import comb
    n = better + worse
    if n == 0:
        return 1.0
    k = max(better, worse)
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def compare_with_baseline(items: list[dict], baseline_items: list[dict]) -> dict:
    """Pair a lessons run with its no-lessons run task by task (labelled tasks only)."""
    base = {i["task_id"]: i for i in baseline_items}
    fixed, broke = [], []
    for i in items:
        b = base.get(i["task_id"])
        if b is None or i.get("label") not in ("A", "B"):
            continue
        now, before = i["correct"] is True, b["correct"] is True  # unclear counts as not right
        if now and not before:
            fixed.append(i["task_id"])
        elif before and not now:
            broke.append(i["task_id"])
    chance = luck_chance(len(fixed), len(broke))
    if not fixed and not broke:
        verdict, summary = "no_change", "The lessons did not change any verdict on these tasks."
    elif chance <= 0.05:
        if len(fixed) > len(broke):
            verdict, summary = "helped", "Lessons very likely helped: a split this one-sided is unlikely to be luck."
        else:
            verdict, summary = "hurt", ("Lessons very likely made it worse. Consider making the previous "
                                        "lessons version active again.")
    elif chance <= 0.2:
        verdict, summary = "maybe", "Could be luck. Test on more tasks to be sure."
    else:
        verdict, summary = "noise", "Can't tell: a difference this small is normal run-to-run noise."
    return {
        "fixed": fixed, "broke": broke, "luck_chance": chance,
        "luck_one_in": round(1 / chance) if chance > 0 else None,
        "verdict": verdict, "summary": summary,
    }


async def learn_from_correction(db, evaluation: dict, true_label: str, reason: str, config: JudgeConfig,
                                judge, gated: bool = False, task_id: Optional[int] = None) -> dict:
    """Learn from one task the user just corrected (or confirmed while it was unsure).

    Writes lessons from this task (using the user's explanation) and merges them into the lessons in use.
    Ungated: saves + activates a new version, so the very next judgment uses it. Gated: saves it as a
    candidate that is NOT used until it passes a test on unseen tasks (see autotrain.py); candidates build
    on each other, so several corrections are tested together.
    """
    result = evaluation["result"]
    agg = result["aggregate"]
    rep = next((r for r in result["runs"] if r["index"] == agg.get("representative")), None) or {}
    j = rep.get("judgment") or {}
    img = evaluation["images"]
    task = DatasetTask(id=f"eval{evaluation['id']}", prompt=evaluation["prompt"],
                       originals=[image_path(p) for p in img["originals"]], a=image_path(img["a"]),
                       b=image_path(img["b"]),
                       label=true_label)
    item = {
        "task_id": task.id, "label": true_label, "verdict": agg.get("verdict"), "status": agg["status"],
        "votes": agg.get("votes") or {}, "decisive_difference": j.get("decisive_difference"),
        "reasoning": j.get("reasoning"), "user_reason": reason.strip(),
        "runs": [{"failed_requirements": [
            {"description": q["description"], "severity": q.get("severity", "unspecified"), "a": q["a"]["status"], "b": q["b"]["status"]}
            for q in j.get("requirements", []) if q["a"]["status"] != q["b"]["status"]]}],
    }
    case = await write_lessons(item, task, config, judge)
    base = (db.pending_candidate() if gated else None) or db.active_knowledge()
    current = list(base["lessons"]) if base else []
    if not case["lessons"]:
        return {"case": case, "knowledge_id": base["id"] if base else None, "lessons": current,
                "new_lessons": [], "changes": "No lesson taken from this task.", "status": "none"}
    lessons, changes = await merge_lessons(current, case["lessons"], config, judge)
    learned_from = list(base["learned_from"] if base else []) + ([task_id] if task_id else [])
    kid = db.add_knowledge(
        guidelines=base["guidelines"] if base else "", lessons=lessons,
        source=f"Learned from your correction on evaluation #{evaluation['id']}",
        parent_id=base["id"] if base else None, activate=not gated,
        status="candidate" if gated else "accepted", learned_from=learned_from)
    return {"case": case, "knowledge_id": kid, "lessons": lessons, "new_lessons": case["lessons"],
            "changes": changes, "status": "candidate" if gated else "accepted"}
