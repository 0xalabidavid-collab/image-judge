"""Does learning from the labeller's answers help on tasks the judge has not seen?

Splits one labelled task set into a train half and a held-out test half (balanced by label),
learns lessons from the train half only (starting from empty lessons), then judges the test half
with and without those lessons.

Talks to a running Image Judge server (default http://127.0.0.1:8000). Start it first, with the
model/provider you want to evaluate, e.g.

    python scripts/split_experiment.py --set-id 2 --seed 7
"""
import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"


def call(method: str, path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(
        BASE + path, method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)
    except urllib.error.HTTPError as exc:
        sys.exit(f"{method} {path} -> {exc.code}: {exc.read()[:300]!r}")


def wait(run_id: int, every: float = 20.0) -> dict:
    last = -1
    while True:
        run = call("GET", f"/api/benchmarks/{run_id}")
        if run["completed"] != last:
            last = run["completed"]
            print(f"  run #{run_id}: {run['status']} {run['completed']}/{run['total']}", flush=True)
        if run["status"] != "running":
            return run
        time.sleep(every)


def make_split(set_id: int, seed: int) -> tuple[int, int]:
    """Create (or reuse) '<name> - train (seed N)' and '<name> - test (seed N)'. Tasks are copied by
    reference to the same image files, so nothing is re-uploaded."""
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app.db import open_db  # noqa: PLC0415
    from app.config import settings  # noqa: PLC0415

    src = call("GET", f"/api/sets/{set_id}")
    names = (f"{src['name']} - train (seed {seed})", f"{src['name']} - test (seed {seed})")
    existing = {s["name"]: s["id"] for s in call("GET", "/api/sets")["items"]}
    if all(n in existing for n in names):
        print("Reusing existing split sets.")
        return existing[names[0]], existing[names[1]]
    if any(n in existing for n in names):
        sys.exit("Only one of the split sets exists; delete it in the app and rerun.")

    rng = random.Random(seed)
    train, test = [], []
    for label in ("A", "B"):
        group = [t for t in src["tasks"] if t["label"] == label]
        group.sort(key=lambda t: t["id"])
        rng.shuffle(group)
        half = len(group) // 2
        train += group[:half]
        test += group[half:]
    db = open_db(settings)
    ids = []
    for name, tasks in zip(names, (train, test)):
        sid = db.create_set(name)
        for t in sorted(tasks, key=lambda t: t["id"]):
            db.add_set_task(sid, t["prompt"], t["images"], t["label"])
        ids.append(sid)
    print(f"Created '{names[0]}' ({len(train)} tasks, set {ids[0]}) and '{names[1]}' ({len(test)} tasks, set {ids[1]}).")
    return ids[0], ids[1]


def acc(items: list[dict]) -> str:
    n = len(items)
    right = sum(i["verdict"] == i["label"] for i in items)
    conf = [i for i in items if i["status"] == "confident"]
    cr = sum(i["verdict"] == i["label"] for i in conf)
    return (f"agrees with you {right}/{n} ({100 * right / max(n, 1):.0f}%) | confident {len(conf)}, "
            f"right {cr} ({100 * cr / max(len(conf), 1):.0f}%)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set-id", type=int, required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model", default=None)
    ap.add_argument("--runs", type=int, default=None)
    ap.add_argument("--max-cases", type=int, default=20, help="most mistakes to write lessons from")
    args = ap.parse_args()

    train_id, test_id = make_split(args.set_id, args.seed)
    over = {k: v for k, v in (("model", args.model), ("runs", args.runs)) if v}

    # Training switches its lessons on. This is an experiment, so put things back how they were afterwards.
    before = call("GET", "/api/config")["active_knowledge"]
    before_id = before["id"] if before else None
    try:
        experiment(args, train_id, test_id, over)
    finally:
        call("POST", "/api/knowledge/activate", {"id": before_id})
        print(f"\nLessons in use restored to: {'version ' + str(before_id) if before_id else 'none'}. "
              "The lessons it learned are still saved in the Lessons page if you want them.")


def experiment(args, train_id: int, test_id: int, over: dict) -> None:

    print("\n1) Training on the train half only (no earlier lessons)...")
    t = call("POST", "/api/runs", {"mode": "train", "set_id": train_id, "knowledge": "none",
                                    "max_cases": args.max_cases, "note": f"split experiment, seed {args.seed}", **over})
    tr = wait(t["id"])
    if tr["status"] != "done" or not tr.get("learned_knowledge_id"):
        sys.exit(f"Training ended as {tr['status']} with no new lessons: {(tr.get('metrics') or {}).get('error')}")
    kid = tr["learned_knowledge_id"]
    know = next(k for k in call("GET", "/api/knowledge")["items"] if k["id"] == kid)
    print(f"\nLearned {len(know['lessons'])} lessons (knowledge #{kid}):")
    for i, l in enumerate(know["lessons"], 1):
        print(f"  {i}. {l}")

    print("\n2) Testing on the held-out half, with and without the lessons...")
    r = call("POST", "/api/runs", {"mode": "test", "set_id": test_id, "knowledge": kid, "compare_baseline": True,
                                    "note": f"split experiment, seed {args.seed}", **over})
    run = wait(r["id"])
    if run["status"] != "done":
        sys.exit(f"Test ended as {run['status']}: {(run.get('metrics') or {}).get('error')}")
    base = wait(r["baseline_id"]) if r.get("baseline_id") else None

    print(f"\nRESULT on {len(run['items'])} held-out tasks:")
    if base:
        print("  without lessons:", acc(base["items"]))
    print("  with lessons:   ", acc(run["items"]))
    if base:
        b = {i["task_id"]: i for i in base["items"]}
        gained = [i["task_id"] for i in run["items"] if i["verdict"] == i["label"] and b[i["task_id"]]["verdict"] != i["label"]]
        lost = [i["task_id"] for i in run["items"] if i["verdict"] != i["label"] and b[i["task_id"]]["verdict"] == i["label"]]
        print(f"  lessons fixed {len(gained)} tasks, broke {len(lost)}")
        if run.get("comparison"):
            print("  comparison:", json.dumps(run["comparison"])[:400])
    print(f"\nRuns: train #{t['id']}, test #{r['id']}, baseline #{r.get('baseline_id')}")


if __name__ == "__main__":
    main()
