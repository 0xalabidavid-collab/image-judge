"""Accuracy harness: load labelled tasks, run the pipeline on all of them, score it.

Dataset formats (paths are relative to the dataset folder / CSV file):

1. CSV  -  a file (or a folder containing `tasks.csv`) with columns
       id, prompt, originals, result_a, result_b, correct_label
   `originals` holds 0-10 image paths separated by ';'. `correct_label` is A or B.

2. Folder per task  -  <dataset>/<task_id>/ containing
       prompt.txt, label.txt (A or B), original*.{png,jpg,jpeg,webp} (0-10),
       a.<ext> (or result_a.<ext>), b.<ext> (or result_b.<ext>)
"""

from __future__ import annotations

import asyncio
import csv
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .aggregate import RunResult, aggregate
from .config import JudgeConfig, settings
from .images import prepare_image
from .judge import Judge, Task, evaluate

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")


class DatasetError(ValueError):
    pass


class BenchmarkAborted(RuntimeError):
    """A configuration error (bad API key, unknown model) made every task fail."""


@dataclass
class DatasetTask:
    id: str
    prompt: str
    originals: list[Path]
    a: Path
    b: Path
    label: Optional[str]
    set_task_id: Optional[int] = None


def load_dataset(path: str | Path) -> list[DatasetTask]:
    path = Path(path).expanduser().resolve()
    if path.is_file() and path.suffix.lower() == ".csv":
        tasks = _load_csv(path)
    elif path.is_dir() and (path / "tasks.csv").exists():
        tasks = _load_csv(path / "tasks.csv")
    elif path.is_dir():
        tasks = _load_folders(path)
    else:
        raise DatasetError(f"{path} is not a CSV file or a dataset folder")
    if not tasks:
        raise DatasetError(f"no tasks found in {path}")
    problems = []
    for t in tasks:
        if t.label not in ("A", "B"):
            problems.append(f"{t.id}: correct_label must be A or B, got {t.label!r}")
        if len(t.originals) > 10:
            problems.append(f"{t.id}: at most 10 originals, got {len(t.originals)}")
        if not t.prompt.strip():
            problems.append(f"{t.id}: empty prompt")
        for p in [*t.originals, t.a, t.b]:
            if not p.is_file():
                problems.append(f"{t.id}: missing file {p}")
    ids = [t.id for t in tasks]
    if len(set(ids)) != len(ids):
        problems.append("duplicate task ids")
    if problems:
        raise DatasetError("dataset has problems:\n  " + "\n  ".join(problems[:50]))
    return tasks


def _load_csv(csv_path: Path) -> list[DatasetTask]:
    base = csv_path.parent
    tasks = []
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"prompt", "result_a", "result_b", "correct_label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise DatasetError(f"{csv_path} is missing columns: {sorted(missing)}")
        for n, row in enumerate(reader, 1):
            originals = [base / p.strip() for p in (row.get("originals") or "").split(";") if p.strip()]
            tasks.append(DatasetTask(
                id=(row.get("id") or "").strip() or f"row{n:04d}",
                prompt=row["prompt"],
                originals=originals,
                a=base / row["result_a"].strip(),
                b=base / row["result_b"].strip(),
                label=row["correct_label"].strip().upper(),
            ))
    return tasks


def _find(folder: Path, stems: tuple[str, ...]) -> Optional[Path]:
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in IMAGE_EXTS and p.stem.lower() in stems:
            return p
    return None


def _load_folders(root: Path) -> list[DatasetTask]:
    tasks = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        if not (folder / "prompt.txt").exists():
            continue
        a = _find(folder, ("a", "result_a"))
        b = _find(folder, ("b", "result_b"))
        label_file = folder / "label.txt"
        tasks.append(DatasetTask(
            id=folder.name,
            prompt=(folder / "prompt.txt").read_text(encoding="utf-8"),
            originals=sorted(p for p in folder.iterdir()
                             if p.suffix.lower() in IMAGE_EXTS and p.stem.lower().startswith("original")),
            a=a or folder / "a.png",
            b=b or folder / "b.png",
            label=label_file.read_text(encoding="utf-8").strip().upper() if label_file.exists() else "",
        ))
    return tasks


def load_task_images(t: DatasetTask) -> Task:
    return Task(
        prompt=t.prompt,
        originals=[prepare_image(p.read_bytes()) for p in t.originals],
        a=prepare_image(t.a.read_bytes()),
        b=prepare_image(t.b.read_bytes()),
    )


# --- scoring ---------------------------------------------------------------

def wilson_lower(correct: int, n: int, z: float = 1.96) -> Optional[float]:
    """Lower bound of the 95% Wilson score interval: what the accuracy is at least, given n."""
    if n == 0:
        return None
    p = correct / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / denom


def _ratio(num: int, den: int) -> Optional[float]:
    return num / den if den else None


def is_labeled(item: dict) -> bool:
    return item.get("label") in ("A", "B")


def compute_metrics(all_items: list[dict]) -> dict:
    """Accuracy is computed over labelled tasks only; coverage and token counts over all tasks."""
    items = [i for i in all_items if is_labeled(i)]
    total = len(items)
    by = {s: [i for i in items if i["status"] == s] for s in ("confident", "review", "unclear", "error")}
    all_confident = sum(1 for i in all_items if i["status"] == "confident")
    conf_ok = sum(1 for i in by["confident"] if i["correct"])
    rev_ok = sum(1 for i in by["review"] if i["correct"])
    answered = len(by["confident"]) + len(by["review"])

    # Single-run accuracy, split by presentation order, exposes position bias.
    runs = [(r, i["label"]) for i in items for r in i["runs"] if r.get("verdict")]
    unswapped = [(r, l) for r, l in runs if not r["swapped"]]
    swapped = [(r, l) for r, l in runs if r["swapped"]]

    tokens_in = sum((r.get("usage") or {}).get("input_tokens", 0)
                    for i in all_items for r in i["runs"] if not r.get("cached"))
    tokens_out = sum((r.get("usage") or {}).get("output_tokens", 0)
                     for i in all_items for r in i["runs"] if not r.get("cached"))

    return {
        "total": total,  # labelled tasks scored
        "tasks": len(all_items),
        "unlabeled": len(all_items) - total,
        "verdict_counts": {s: sum(1 for i in all_items if i["status"] == s)
                           for s in ("confident", "review", "unclear", "error")},
        "counts": {s: len(v) for s, v in by.items()},
        "overall_accuracy": _ratio(conf_ok + rev_ok, total),  # unclear / error count as wrong
        "answered_accuracy": _ratio(conf_ok + rev_ok, answered),
        "confident_accuracy": _ratio(conf_ok, len(by["confident"])),
        "confident_accuracy_lower95": wilson_lower(conf_ok, len(by["confident"])),
        "confident_correct": conf_ok,
        "confident_wrong": len(by["confident"]) - conf_ok,
        "coverage": _ratio(all_confident, len(all_items)),
        "review_accuracy": _ratio(rev_ok, len(by["review"])),
        "single_run_accuracy": _ratio(sum(r["verdict"] == l for r, l in runs), len(runs)),
        "single_run_accuracy_unswapped": _ratio(sum(r["verdict"] == l for r, l in unswapped), len(unswapped)),
        "single_run_accuracy_swapped": _ratio(sum(r["verdict"] == l for r, l in swapped), len(swapped)),
        "run_count_curve": run_count_curve(items),
        "api_tokens": {"input": tokens_in, "output": tokens_out},
    }


def run_count_curve(items: list[dict]) -> list[dict]:
    """Re-aggregate using only the first k runs of each task, for every k.

    Free (no API calls) and answers "how many runs do we need?" directly.
    """
    items = [i for i in items if is_labeled(i)]
    if not items:
        return []
    max_runs = max(len(i["runs"]) for i in items)
    curve = []
    for k in range(1, max_runs + 1):
        sub = []
        for i in items:
            if not i["runs"]:  # identical A/B or a load error: same outcome for every k
                sub.append({"status": i["status"], "correct": i["correct"], "label": i["label"]})
                continue
            rr = [RunResult(r["index"], r["swapped"], r["verdict"], r["confidence"], error=r.get("error"))
                  for r in i["runs"][:k]]
            agg = aggregate(rr, planned=k)
            sub.append({"status": agg.status, "label": i["label"],
                        "correct": None if agg.verdict is None else agg.verdict == i["label"]})
        conf = [s for s in sub if s["status"] == "confident"]
        conf_ok = sum(1 for s in conf if s["correct"])
        curve.append({
            "runs": k,
            "coverage": _ratio(len(conf), len(sub)),
            "confident_accuracy": _ratio(conf_ok, len(conf)),
            "confident_wrong": len(conf) - conf_ok,
            "overall_accuracy": _ratio(sum(1 for s in sub if s["correct"]), len(sub)),
        })
    return curve


def failure_list(items: list[dict]) -> list[dict]:
    """Everything not correctly answered, most dangerous first (confident and wrong)."""
    order = {"confident": 0, "review": 1, "unclear": 2, "error": 3}
    fails = [i for i in items if is_labeled(i) and not i["correct"]]
    return sorted(fails, key=lambda i: (order[i["status"]], i["task_id"]))


def item_record(t: DatasetTask, result: dict) -> dict:
    agg = result["aggregate"]
    rep = next((r for r in result["runs"] if r["index"] == agg["representative"]), None)
    j = (rep or {}).get("judgment") or {}
    return {
        "task_id": t.id,
        "set_task_id": t.set_task_id,
        "prompt": t.prompt,
        "label": t.label,
        "status": agg["status"],
        "verdict": agg["verdict"],
        "correct": None if agg["verdict"] is None or t.label not in ("A", "B") else agg["verdict"] == t.label,
        "explanation": agg["explanation"],
        "votes": agg["votes"],
        "decisive_difference": j.get("decisive_difference"),
        "reasoning": j.get("reasoning"),
        "notes": result.get("notes", []),
        "runs": [
            {
                "index": r["index"], "swapped": r["swapped"], "verdict": r["verdict"],
                "confidence": r["confidence"], "error": r["error"], "cached": r["cached"],
                "usage": r.get("usage"),
                "decisive_difference": (r.get("judgment") or {}).get("decisive_difference"),
                "reasoning": (r.get("judgment") or {}).get("reasoning"),
                "failed_requirements": [
                    {"id": q["id"], "description": q["description"], "severity": q.get("severity", "unspecified"),
                     "a": q["a"]["status"], "b": q["b"]["status"]}
                    for q in (r.get("judgment") or {}).get("requirements", [])
                    if q["a"]["status"] != q["b"]["status"]
                ],
            }
            for r in result["runs"]
        ],
    }


@dataclass
class BenchmarkOutcome:
    run_id: int
    metrics: dict
    items: list[dict] = field(default_factory=list)
    results_file: Optional[Path] = None


async def run_benchmark(
    dataset: str | Path,
    config: JudgeConfig,
    judge: Judge,
    db,
    limit: Optional[int] = None,
    use_cache: bool = True,
    note: str = "",
    on_progress: Optional[Callable[[int, int, dict], None]] = None,
    run_id: Optional[int] = None,
) -> BenchmarkOutcome:
    tasks = load_dataset(dataset)
    if limit:
        tasks = tasks[:limit]
    return await run_tasks(tasks, str(dataset), config, judge, db, use_cache=use_cache, note=note,
                           on_progress=on_progress, run_id=run_id)


async def run_tasks(
    tasks: list[DatasetTask],
    source: str,
    config: JudgeConfig,
    judge: Judge,
    db,
    use_cache: bool = True,
    note: str = "",
    on_progress: Optional[Callable[[int, int, dict], None]] = None,
    run_id: Optional[int] = None,
    mode: str = "benchmark",
    set_id: Optional[int] = None,
    baseline_of: Optional[int] = None,
    finish: bool = True,
) -> BenchmarkOutcome:
    """Judge every task, store one item per task, and score the labelled ones.

    finish=False leaves the run marked 'running' so a caller (training) can do more work first.
    """
    if run_id is None:
        run_id = db.start_benchmark(source, config.as_dict(), len(tasks), note, mode=mode, set_id=set_id,
                                    baseline_of=baseline_of)
    dataset = source

    # Bound how many tasks hold decoded images in memory at once.
    task_slots = asyncio.Semaphore(max(1, settings.max_concurrency))
    items: list[dict] = []
    abort: list[str] = []  # first fatal error (bad key, unknown model) stops the run

    async def one(t: DatasetTask) -> None:
        async with task_slots:
            if abort:
                return
            try:
                task = await asyncio.to_thread(load_task_images, t)
                result = await evaluate(task, config, judge, db=db, use_cache=use_cache)
            except Exception as exc:  # one broken task must not sink the whole run
                result = {"aggregate": {"status": "error", "verdict": None, "votes": {},
                                        "explanation": f"{type(exc).__name__}: {exc}", "representative": None},
                          "runs": [], "notes": []}
            if result.get("fatal"):
                abort.append(result["fatal"])
                return
            item = item_record(t, result)
            items.append(item)
            db.add_benchmark_item(run_id, item)
            if on_progress:
                on_progress(len(items), len(tasks), item)

    try:
        await asyncio.gather(*(one(t) for t in tasks))
    except BaseException:
        db.finish_benchmark(run_id, compute_metrics(items) if items else None, status="failed")
        raise

    if abort:
        db.finish_benchmark(run_id, compute_metrics(items) if items else None, status="failed")
        raise BenchmarkAborted(abort[0])

    items.sort(key=lambda i: i["task_id"])
    metrics = compute_metrics(items)
    if finish:
        db.finish_benchmark(run_id, metrics)

    settings.results_dir.mkdir(parents=True, exist_ok=True)
    out = settings.results_dir / f"run_{run_id:04d}.json"
    out.write_text(json.dumps({
        "run_id": run_id, "dataset": str(dataset), "config": config.as_dict(), "note": note,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"), "metrics": metrics,
        "failures": failure_list(items), "items": items,
    }, indent=2), encoding="utf-8")
    return BenchmarkOutcome(run_id, metrics, items, out)


def format_report(outcome: BenchmarkOutcome, config: JudgeConfig, max_failures: int = 20) -> str:
    m = outcome.metrics

    def pct(x: Optional[float]) -> str:
        return "n/a" if x is None else f"{x * 100:.1f}%"

    c = m["counts"]
    lines = [
        f"Benchmark run #{outcome.run_id}  model={config.model} rubric={config.rubric_version} "
        f"runs={config.runs} effort={config.effort}",
        f"Tasks: {m['total']}  (confident {c['confident']}, review {c['review']}, "
        f"unclear {c['unclear']}, error {c['error']})",
        "",
        f"  Accuracy on confident verdicts : {pct(m['confident_accuracy'])}  "
        f"({m['confident_correct']}/{c['confident']}; 95% lower bound {pct(m['confident_accuracy_lower95'])})",
        f"  Coverage (% confident)         : {pct(m['coverage'])}",
        f"  Overall accuracy               : {pct(m['overall_accuracy'])}  (unclear/error count as wrong)",
        f"  Accuracy when answered         : {pct(m['answered_accuracy'])}  (confident + review)",
        f"  Review-bucket accuracy         : {pct(m['review_accuracy'])}",
        f"  Single-run accuracy            : {pct(m['single_run_accuracy'])}  "
        f"(A first {pct(m['single_run_accuracy_unswapped'])}, B first {pct(m['single_run_accuracy_swapped'])})",
        f"  API tokens (uncached calls)    : {m['api_tokens']['input']:,} in / {m['api_tokens']['output']:,} out",
        "",
        "  Runs  coverage  conf.acc  conf.wrong  overall",
    ]
    for row in m["run_count_curve"]:
        lines.append(f"  {row['runs']:>4}  {pct(row['coverage']):>8}  {pct(row['confident_accuracy']):>8}"
                     f"  {row['confident_wrong']:>10}  {pct(row['overall_accuracy']):>7}")
    fails = failure_list(outcome.items)
    lines += ["", f"Failures ({len(fails)}; confident-and-wrong first):"]
    for f in fails[:max_failures]:
        lines.append(f"  [{f['status']}] {f['task_id']}: truth={f['label']} verdict={f['verdict'] or '-'} "
                     f"votes A{f['votes'].get('A', 0)}/B{f['votes'].get('B', 0)}")
        lines.append(f"      prompt: {f['prompt'].strip()[:140]}")
        if not f["runs"] and f.get("explanation"):
            lines.append(f"      error: {f['explanation'][:300]}")
        if f.get("decisive_difference"):
            lines.append(f"      model's decisive difference: {f['decisive_difference'][:300]}")
        for r in f["runs"]:
            if r.get("error"):
                lines.append(f"      run {r['index']}: ERROR {r['error'][:200]}")
    if len(fails) > max_failures:
        lines.append(f"  ... {len(fails) - max_failures} more in {outcome.results_file}")
    if outcome.results_file:
        lines += ["", f"Full results: {outcome.results_file}"]
    return "\n".join(lines)
