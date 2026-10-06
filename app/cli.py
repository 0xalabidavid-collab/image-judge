"""Command line: run benchmarks, compare configurations, export feedback as a dataset.

  python -m app.cli benchmark --dataset datasets/mine
  python -m app.cli sweep --dataset datasets/mine --config rubric_version=v1 --config rubric_version=v2
  python -m app.cli export-feedback --out datasets/from_feedback
  python -m app.cli train --set "Train 1"
  python -m app.cli test --set "Test 1" --compare
  python -m app.cli judge --set "Batch 3"
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import shutil
import sys
import time
from pathlib import Path

from .benchmark import BenchmarkAborted, DatasetError, format_report, run_benchmark
from .config import ROOT, JudgeConfig, settings
from .db import DB, open_db
from .storage import get_store
from .learn import SetError, compare_with_baseline, run_set, train
from .providers import RoutingJudge

EXPERIMENTS = ROOT / "EXPERIMENTS.md"


def _pct(x):
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _config_from_args(args) -> JudgeConfig:
    return settings.judge.with_overrides(
        runs=args.runs, model=args.model, rubric_version=args.rubric, effort=args.effort)


def _parse_config(spec: str, base: JudgeConfig) -> JudgeConfig:
    """'rubric_version=v2,runs=6,model=claude-opus-5-5' -> JudgeConfig."""
    overrides = {}
    for part in filter(None, (p.strip() for p in spec.split(","))):
        key, _, value = part.partition("=")
        key = {"rubric": "rubric_version"}.get(key.strip(), key.strip())
        if key not in ("runs", "model", "rubric_version", "effort", "max_tokens"):
            raise SystemExit(f"unknown config key {key!r} in {spec!r}")
        overrides[key] = int(value) if key in ("runs", "max_tokens") else value.strip()
    return base.with_overrides(**overrides)


def _log_experiment(outcome, cfg: JudgeConfig, dataset: str, note: str) -> None:
    """Append one measured row to EXPERIMENTS.md - the log only ever contains real runs."""
    m = outcome.metrics
    row = (f"| {time.strftime('%Y-%m-%d')} | #{outcome.run_id} | {Path(dataset).name} | {m['total']} | "
           f"{cfg.model} | {cfg.rubric_version} | {cfg.runs} | {cfg.effort} | "
           f"{_pct(m['confident_accuracy'])} ({m['confident_correct']}/{m['counts']['confident']}) | "
           f"{_pct(m['confident_accuracy_lower95'])} | {_pct(m['coverage'])} | "
           f"{_pct(m['overall_accuracy'])} | {note} |\n")
    with EXPERIMENTS.open("a", encoding="utf-8") as f:
        f.write(row)


def _progress(done: int, total: int, item: dict) -> None:
    mark = {True: "ok", False: "WRONG", None: "-"}[item["correct"]]
    item = {**item, "label": item["label"] or "?"}
    print(f"  [{done}/{total}] {item['task_id']}: {item['status']:<9} verdict={item['verdict'] or '-'} "
          f"truth={item['label']} {mark}", file=sys.stderr, flush=True)


async def _bench(dataset: str, cfg: JudgeConfig, args, db: DB, note: str):
    return await run_benchmark(dataset, cfg, RoutingJudge(), db, limit=args.limit,
                               use_cache=not args.no_cache, note=note, on_progress=_progress)


def cmd_benchmark(args) -> int:
    db = open_db(settings)
    cfg = _config_from_args(args)
    try:
        outcome = asyncio.run(_bench(args.dataset, cfg, args, db, args.note))
    except (DatasetError, BenchmarkAborted) as exc:
        print(f"Benchmark stopped: {exc}", file=sys.stderr)
        return 2
    print(format_report(outcome, cfg))
    if not args.no_log:
        _log_experiment(outcome, cfg, args.dataset, args.note)
    return 0


def cmd_sweep(args) -> int:
    db = open_db(settings)
    base = _config_from_args(args)
    rows = []
    for spec in args.config:
        cfg = _parse_config(spec, base)
        print(f"\n=== {spec} ===", file=sys.stderr)
        try:
            outcome = asyncio.run(_bench(args.dataset, cfg, args, db, f"{args.note} {spec}".strip()))
        except (DatasetError, BenchmarkAborted) as exc:
            print(f"Benchmark stopped: {exc}", file=sys.stderr)
            return 2
        if not args.no_log:
            _log_experiment(outcome, cfg, args.dataset, f"{args.note} {spec}".strip())
        rows.append((spec, outcome.metrics))
    print("\n| config | conf. accuracy | 95% lower | coverage | overall | confident wrong |")
    print("|---|---|---|---|---|---|")
    for spec, m in rows:
        print(f"| {spec} | {_pct(m['confident_accuracy'])} | {_pct(m['confident_accuracy_lower95'])} | "
              f"{_pct(m['coverage'])} | {_pct(m['overall_accuracy'])} | {m['confident_wrong']} |")
    return 0


def cmd_export_feedback(args) -> int:
    """Turn evaluations the user marked Right/Wrong into a labelled benchmark dataset."""
    db = open_db(settings)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for ev in db.labeled_evaluations():
        def copy(src: str) -> str:
            dst = out / "images" / Path(src).name  # files are content-addressed; dedupes naturally
            if not dst.exists():
                shutil.copyfile(get_store().ensure(src), dst)
            return f"images/{dst.name}"
        try:
            rows.append({
                "id": f"eval{ev['id']:05d}",
                "prompt": ev["prompt"],
                "originals": ";".join(copy(p) for p in ev["images"]["originals"]),
                "result_a": copy(ev["images"]["a"]),
                "result_b": copy(ev["images"]["b"]),
                "correct_label": ev["feedback"]["true_label"],
            })
        except FileNotFoundError as exc:
            print(f"skipping evaluation {ev['id']}: {exc}", file=sys.stderr)
    with (out / "tasks.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "prompt", "originals", "result_a", "result_b", "correct_label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} labelled tasks to {out / 'tasks.csv'}")
    return 0


def _set_config(args, db: DB):
    sets = {x["name"]: x["id"] for x in db.list_sets()}
    if args.set not in sets:
        raise SystemExit(f"No task set called {args.set!r}. Sets: {', '.join(sets) or 'none yet'}")
    if args.lessons == "active":
        know = db.active_knowledge()
    elif args.lessons == "none":
        know = None
    else:
        know = db.get_knowledge(int(args.lessons))
        if not know:
            raise SystemExit(f"No lessons version {args.lessons}")
    cfg = settings.judge.with_overrides(runs=args.runs, model=args.model, effort=args.effort).with_knowledge(know)
    return sets[args.set], cfg


def _score(m) -> str:
    if not m or not m.get("total"):
        return "no answers marked, so no score"
    correct = round((m["overall_accuracy"] or 0) * m["total"])
    return (f"{correct}/{m['total']} correct ({_pct(m['overall_accuracy'])}); confident on {_pct(m['coverage'])}, "
            f"accuracy when confident {_pct(m['confident_accuracy'])}")


def cmd_train(args) -> int:
    db = open_db(settings)
    set_id, cfg = _set_config(args, db)
    try:
        out = asyncio.run(train(db, set_id, cfg, RoutingJudge(), use_cache=not args.no_cache,
                                max_cases=args.max_cases, on_progress=_progress))
    except (SetError, BenchmarkAborted) as exc:
        print(f"Training stopped: {exc}", file=sys.stderr)
        return 2
    t = out.run.metrics["training"]
    print(f"Training run #{out.run.run_id} on {args.set!r}")
    print(f"  Score before learning: {_score(out.run.metrics)}")
    print(f"  Studied {t['cases_reviewed']} mistakes/unsure tasks, {t['new_lessons']} new lessons")
    if t["flagged_labels"]:
        print(f"  Marked answer looks wrong for task(s): {', '.join(t['flagged_labels'])} (not learned from)")
    if out.knowledge_id and out.knowledge_id != cfg.knowledge_id:
        print(f"  Saved lessons version {out.knowledge_id} ({len(out.lessons)} lessons), now active:")
        for i, l in enumerate(out.lessons, 1):
            print(f"    {i}. {l}")
    else:
        print("  Lessons unchanged.")
    return 0


def cmd_run_set(args) -> int:
    db = open_db(settings)
    set_id, cfg = _set_config(args, db)
    try:
        main_run, base = asyncio.run(run_set(db, set_id, args.cmd, cfg, RoutingJudge(), use_cache=not args.no_cache,
                                             compare_baseline=getattr(args, "compare", False), on_progress=_progress))
    except (SetError, BenchmarkAborted) as exc:
        print(f"Run stopped: {exc}", file=sys.stderr)
        return 2
    lessons = f"lessons version {cfg.knowledge_id}" if cfg.knowledge_id else "no lessons"
    print(f"{args.cmd.title()} run #{main_run.run_id} on {args.set!r} with {lessons}: {_score(main_run.metrics)}")
    if base:
        print(f"Without lessons (run #{base.run_id}): {_score(base.metrics)}")
        cmp = compare_with_baseline(main_run.items, base.items)
        if cmp["fixed"] or cmp["broke"]:
            odds = "high" if cmp["luck_chance"] >= 0.5 else f"about 1 in {cmp['luck_one_in']}"
            print(f"Lessons fixed {len(cmp['fixed'])} and broke {len(cmp['broke'])}; "
                  f"chance of this by luck: {odds} ({cmp['luck_chance']:.1%})")
        print(cmp["summary"])
    for i in main_run.items:
        mark = {True: "right", False: "WRONG", None: ""}[i["correct"]]
        print(f"  task {i['task_id']}: {i['status']:<9} verdict={i['verdict'] or '-'} answer={i['label'] or '?'} {mark}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.cli", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def judge_opts(p):
        p.add_argument("--dataset", required=True, help="dataset folder or CSV")
        p.add_argument("--runs", type=int, help=f"independent runs per task (default {settings.judge.runs})")
        p.add_argument("--model", help=f"default {settings.judge.model}")
        p.add_argument("--rubric", help=f"rubric version (default {settings.judge.rubric_version})")
        p.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
        p.add_argument("--limit", type=int, help="only the first N tasks")
        p.add_argument("--no-cache", action="store_true", help="ignore cached judge calls")
        p.add_argument("--no-log", action="store_true", help="don't append a row to EXPERIMENTS.md")
        p.add_argument("--note", default="", help="free text recorded with the run")

    p = sub.add_parser("benchmark", help="run the pipeline on a labelled dataset and report accuracy")
    judge_opts(p)
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("sweep", help="benchmark several configurations on the same dataset")
    judge_opts(p)
    p.add_argument("--config", action="append", required=True,
                   help="comma-separated overrides, e.g. rubric_version=v2,runs=6 (repeatable)")
    p.set_defaults(func=cmd_sweep)

    def set_opts(p):
        p.add_argument("--set", required=True, help="task set name (create sets on the Train & Test page)")
        p.add_argument("--lessons", default="active", help="'active' (default), 'none', or a version number")
        p.add_argument("--runs", type=int)
        p.add_argument("--model")
        p.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
        p.add_argument("--no-cache", action="store_true")

    p = sub.add_parser("train", help="judge a labelled set, learn lessons from the mistakes")
    set_opts(p)
    p.add_argument("--max-cases", type=int, default=15, help="most mistakes to learn from")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("test", help="judge a set without seeing the answers, then score it")
    set_opts(p)
    p.add_argument("--compare", action="store_true", help="also run without lessons")
    p.set_defaults(func=cmd_run_set)

    p = sub.add_parser("judge", help="judge every task in a set (answers optional)")
    set_opts(p)
    p.set_defaults(func=cmd_run_set)

    p = sub.add_parser("export-feedback", help="export Right/Wrong-labelled evaluations as a dataset")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_export_feedback)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
