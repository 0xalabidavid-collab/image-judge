"""SQLite storage: evaluations, feedback, task sets, knowledge versions, runs, and a judge-call cache."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from .identity import who

SCHEMA = """
CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    prompt TEXT NOT NULL,
    images TEXT NOT NULL,          -- JSON: {"originals": [paths], "a": path, "b": path}
    config TEXT NOT NULL,          -- JSON JudgeConfig
    status TEXT NOT NULL,          -- confident | review | unclear | error
    verdict TEXT,                  -- A | B | NULL
    result TEXT NOT NULL           -- JSON: aggregate + runs
);
CREATE TABLE IF NOT EXISTS feedback (
    evaluation_id INTEGER PRIMARY KEY REFERENCES evaluations(id),
    created_at REAL NOT NULL,
    verdict_correct INTEGER,       -- 1 right, 0 wrong, NULL when there was no verdict
    true_label TEXT NOT NULL       -- A | B, derived or chosen by the user
);
CREATE TABLE IF NOT EXISTS judge_cache (
    key TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    judgment TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS benchmark_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    finished_at REAL,
    dataset TEXT NOT NULL,
    config TEXT NOT NULL,
    status TEXT NOT NULL,          -- running | done | failed | interrupted
    total INTEGER NOT NULL DEFAULT 0,
    completed INTEGER NOT NULL DEFAULT 0,
    metrics TEXT,
    note TEXT
);
CREATE TABLE IF NOT EXISTS benchmark_items (
    run_id INTEGER NOT NULL REFERENCES benchmark_runs(id),
    task_id TEXT NOT NULL,
    label TEXT NOT NULL,
    status TEXT NOT NULL,
    verdict TEXT,
    correct INTEGER,
    result TEXT NOT NULL,
    PRIMARY KEY (run_id, task_id)
);
CREATE TABLE IF NOT EXISTS task_sets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    name TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS set_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    set_id INTEGER NOT NULL REFERENCES task_sets(id),
    created_at REAL NOT NULL,
    prompt TEXT NOT NULL,
    images TEXT NOT NULL,          -- JSON: {"originals": [paths], "a": path, "b": path}
    label TEXT                     -- A | B | NULL (not known yet)
);
CREATE TABLE IF NOT EXISTS knowledge (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    parent_id INTEGER,
    guidelines TEXT NOT NULL DEFAULT '',
    lessons TEXT NOT NULL DEFAULT '[]',   -- JSON list of rule strings
    source TEXT NOT NULL DEFAULT '',      -- e.g. "trained on 'Train 1' (run #7)" or "edited by hand"
    active INTEGER NOT NULL DEFAULT 0
);
"""

# Columns added after the first release; created on startup if an older database lacks them.
MIGRATIONS = {
    "benchmark_runs": {
        "mode": "TEXT NOT NULL DEFAULT 'benchmark'",  # benchmark | train | test | judge
        "set_id": "INTEGER",
        "started_by": "TEXT NOT NULL DEFAULT ''",
        "knowledge_id": "INTEGER",          # knowledge the judge used
        "learned_knowledge_id": "INTEGER",  # train runs: the version they produced
        "baseline_of": "INTEGER",           # untrained comparison run for this run id
    },
    "evaluations": {"knowledge_id": "INTEGER", "set_task_id": "INTEGER", "created_by": "TEXT NOT NULL DEFAULT ''"},
    "feedback": {"reason": "TEXT NOT NULL DEFAULT ''",  # why the correct result is correct
                 "labelled_by": "TEXT NOT NULL DEFAULT ''"},  # who gave the answer (hosted app)
    "knowledge": {
        "status": "TEXT NOT NULL DEFAULT 'accepted'",  # accepted | candidate | superseded | rejected
        "learned_from": "TEXT NOT NULL DEFAULT '[]'",  # JSON list of set-task ids the lessons were learned from
        "gate_result": "TEXT",                         # JSON: how the held-out test went
        "author": "TEXT NOT NULL DEFAULT ''",          # who taught it (hosted app)
    },
    "set_tasks": {"created_by": "TEXT NOT NULL DEFAULT ''", "labelled_by": "TEXT NOT NULL DEFAULT ''"},
}


class DB:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        for table, columns in MIGRATIONS.items():
            existing = {r["name"] for r in self._conn.execute(f"PRAGMA table_info({table})")}
            for name, decl in columns.items():
                if name not in existing:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
        self._conn.commit()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def _one(self, sql: str, args: tuple = ()) -> Optional[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, args).fetchone()

    def _all(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, args).fetchall()

    # --- judge cache -----------------------------------------------------
    def cache_get(self, key: str) -> Optional[dict]:
        row = self._one("SELECT judgment FROM judge_cache WHERE key = ?", (key,))
        return json.loads(row["judgment"]) if row else None

    def cache_put(self, key: str, judgment: dict) -> None:
        with self.tx() as c:
            c.execute("INSERT OR REPLACE INTO judge_cache VALUES (?, ?, ?)",
                      (key, time.time(), json.dumps(judgment)))

    # --- evaluations -----------------------------------------------------
    def add_evaluation(self, prompt: str, images: dict, config: dict, result: dict) -> int:
        agg = result["aggregate"]
        with self.tx() as c:
            cur = c.execute(
                "INSERT INTO evaluations (created_at, prompt, images, config, status, verdict, result, knowledge_id,"
                " created_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (time.time(), prompt, json.dumps(images), json.dumps(config),
                 agg["status"], agg["verdict"], json.dumps(result), config.get("knowledge_id"), who()),
            )
            return int(cur.lastrowid)

    def get_evaluation(self, eval_id: int) -> Optional[dict]:
        row = self._one(
            "SELECT e.*, f.verdict_correct, f.true_label FROM evaluations e"
            " LEFT JOIN feedback f ON f.evaluation_id = e.id WHERE e.id = ?", (eval_id,))
        return _eval_row(row) if row else None

    def list_evaluations(self, limit: int = 50) -> list[dict]:
        rows = self._all(
            "SELECT e.*, f.verdict_correct, f.true_label FROM evaluations e"
            " LEFT JOIN feedback f ON f.evaluation_id = e.id ORDER BY e.id DESC LIMIT ?", (limit,))
        return [_eval_row(r, include_result=False) for r in rows]

    def labeled_evaluations(self) -> list[dict]:
        rows = self._all(
            "SELECT e.*, f.verdict_correct, f.true_label FROM evaluations e"
            " JOIN feedback f ON f.evaluation_id = e.id ORDER BY e.id")
        return [_eval_row(r, include_result=False) for r in rows]

    def set_feedback(self, eval_id: int, verdict_correct: Optional[bool], true_label: str,
                     reason: str = "") -> None:
        with self.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO feedback (evaluation_id, created_at, verdict_correct, true_label, reason,"
                " labelled_by) VALUES (?, ?, ?, ?, ?, ?)",
                (eval_id, time.time(), None if verdict_correct is None else int(verdict_correct), true_label,
                 reason.strip(), who()),
            )

    def set_feedback_reason(self, eval_id: int, reason: str) -> bool:
        """Add or change the reason on an answer already given. False if there is no answer for it."""
        with self.tx() as c:
            cur = c.execute("UPDATE feedback SET reason = ?, labelled_by = CASE WHEN labelled_by = '' THEN ?"
                            " ELSE labelled_by END WHERE evaluation_id = ?", (reason.strip(), who(), eval_id))
            return bool(getattr(cur, "rowcount", 1))

    def answers_missing_reason(self, limit: int = 100) -> list[dict]:
        """Answers worth explaining (the judge was wrong or unsure) that have no written reason yet."""
        rows = self._all(
            "SELECT e.id, e.prompt, e.images, e.status, e.verdict, f.true_label, f.verdict_correct, f.labelled_by"
            " FROM feedback f JOIN evaluations e ON e.id = f.evaluation_id"
            " WHERE f.reason = '' AND (f.verdict_correct IS NULL OR f.verdict_correct = 0 OR e.status <> 'confident')"
            " ORDER BY (CASE WHEN e.status = 'confident' AND f.verdict_correct = 0 THEN 0 ELSE 1 END), e.id DESC"
            " LIMIT ?", (limit,))
        return [{"id": r["id"], "prompt": r["prompt"], "images": json.loads(r["images"]), "status": r["status"],
                 "verdict": r["verdict"], "true_label": r["true_label"],
                 "verdict_correct": None if r["verdict_correct"] is None else bool(r["verdict_correct"]),
                 "labelled_by": r["labelled_by"]} for r in rows]

    def reason_coverage(self) -> dict:
        row = self._one("SELECT COUNT(*) AS n, SUM(CASE WHEN reason <> '' THEN 1 ELSE 0 END) AS with_reason"
                        " FROM feedback")
        return {"answers": int(row["n"] or 0), "with_reason": int(row["with_reason"] or 0)}

    def link_evaluation_task(self, eval_id: int, task_id: int) -> None:
        with self.tx() as c:
            c.execute("UPDATE evaluations SET set_task_id = ? WHERE id = ?", (task_id, eval_id))

    def evaluation_task_id(self, eval_id: int) -> Optional[int]:
        row = self._one("SELECT set_task_id FROM evaluations WHERE id = ?", (eval_id,))
        return row["set_task_id"] if row else None

    def feedback_stats(self) -> dict:
        rows = self._all(
            "SELECT e.status, f.verdict_correct FROM feedback f JOIN evaluations e ON e.id = f.evaluation_id")
        stats: dict[str, Any] = {}
        for status in ("confident", "review", "unclear"):
            subset = [r["verdict_correct"] for r in rows if r["status"] == status]
            judged = [v for v in subset if v is not None]
            stats[status] = {"labeled": len(subset), "correct": sum(judged), "with_verdict": len(judged)}
        return stats

    # --- benchmark / train / test / judge runs ---------------------------
    def start_benchmark(self, dataset: str, config: dict, total: int, note: str = "", mode: str = "benchmark",
                        set_id: Optional[int] = None, baseline_of: Optional[int] = None) -> int:
        with self.tx() as c:
            cur = c.execute(
                "INSERT INTO benchmark_runs (created_at, dataset, config, status, total, note, mode, set_id,"
                " knowledge_id, baseline_of, started_by) VALUES (?, ?, ?, 'running', ?, ?, ?, ?, ?, ?, ?)",
                (time.time(), dataset, json.dumps(config), total, note, mode, set_id,
                 config.get("knowledge_id"), baseline_of, who()),
            )
            return int(cur.lastrowid)

    def add_benchmark_item(self, run_id: int, item: dict) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO benchmark_items VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run_id, item["task_id"], item["label"] or "", item["status"], item["verdict"],
                 None if item["correct"] is None else int(item["correct"]), json.dumps(item)),
            )
            c.execute("UPDATE benchmark_runs SET completed = completed + 1 WHERE id = ?", (run_id,))

    def finish_benchmark(self, run_id: int, metrics: Optional[dict], status: str = "done") -> None:
        with self.tx() as c:
            c.execute("UPDATE benchmark_runs SET status = ?, finished_at = ?, metrics = ? WHERE id = ?",
                      (status, time.time(), json.dumps(metrics) if metrics else None, run_id))

    def update_benchmark_metrics(self, run_id: int, metrics: Optional[dict]) -> None:
        with self.tx() as c:
            c.execute("UPDATE benchmark_runs SET metrics = ? WHERE id = ?",
                      (json.dumps(metrics) if metrics else None, run_id))

    def set_learned_knowledge(self, run_id: int, knowledge_id: int) -> None:
        with self.tx() as c:
            c.execute("UPDATE benchmark_runs SET learned_knowledge_id = ? WHERE id = ?", (knowledge_id, run_id))

    def mark_interrupted_runs(self) -> None:
        """Runs left 'running' by a previous process (server restart, Ctrl+C) can never finish."""
        with self.tx() as c:
            c.execute("UPDATE benchmark_runs SET status = 'interrupted', finished_at = ? WHERE status = 'running'",
                      (time.time(),))

    def get_benchmark(self, run_id: int, include_items: bool = True) -> Optional[dict]:
        row = self._one("SELECT * FROM benchmark_runs WHERE id = ?", (run_id,))
        if not row:
            return None
        out = _bench_row(row)
        if include_items:
            out["items"] = [json.loads(r["result"]) for r in
                            self._all("SELECT result FROM benchmark_items WHERE run_id = ? ORDER BY task_id", (run_id,))]
        return out

    def list_benchmarks(self, limit: int = 50, mode: Optional[str] = None) -> list[dict]:
        if mode:
            rows = self._all("SELECT * FROM benchmark_runs WHERE mode = ? ORDER BY id DESC LIMIT ?", (mode, limit))
        else:
            rows = self._all("SELECT * FROM benchmark_runs ORDER BY id DESC LIMIT ?", (limit,))
        return [_bench_row(r) for r in rows]

    def runs_for_set(self, set_id: int) -> list[dict]:
        return [_bench_row(r) for r in
                self._all("SELECT * FROM benchmark_runs WHERE set_id = ? ORDER BY id", (set_id,))]

    # --- task sets -------------------------------------------------------
    def create_set(self, name: str) -> int:
        with self.tx() as c:
            cur = c.execute("INSERT INTO task_sets (created_at, name) VALUES (?, ?)", (time.time(), name.strip()))
            return int(cur.lastrowid)

    def get_or_create_set(self, name: str) -> int:
        row = self._one("SELECT id FROM task_sets WHERE name = ?", (name.strip(),))
        return int(row["id"]) if row else self.create_set(name)

    def list_sets(self) -> list[dict]:
        rows = self._all(
            "SELECT s.id, s.name, s.created_at, COUNT(t.id) AS tasks,"
            " CAST(SUM(CASE WHEN t.label IN ('A','B') THEN 1 ELSE 0 END) AS INTEGER) AS labeled"
            " FROM task_sets s LEFT JOIN set_tasks t ON t.set_id = s.id GROUP BY s.id ORDER BY s.id DESC")
        return [{"id": r["id"], "name": r["name"], "created_at": r["created_at"],
                 "tasks": r["tasks"], "labeled": r["labeled"] or 0} for r in rows]

    def get_set(self, set_id: int) -> Optional[dict]:
        row = self._one("SELECT * FROM task_sets WHERE id = ?", (set_id,))
        if not row:
            return None
        tasks = [_task_row(r) for r in self._all("SELECT * FROM set_tasks WHERE set_id = ? ORDER BY id", (set_id,))]
        return {"id": row["id"], "name": row["name"], "created_at": row["created_at"], "tasks": tasks}

    def rename_set(self, set_id: int, name: str) -> None:
        with self.tx() as c:
            c.execute("UPDATE task_sets SET name = ? WHERE id = ?", (name.strip(), set_id))

    def delete_set(self, set_id: int) -> None:
        with self.tx() as c:
            c.execute("DELETE FROM set_tasks WHERE set_id = ?", (set_id,))
            c.execute("DELETE FROM task_sets WHERE id = ?", (set_id,))

    def add_set_task(self, set_id: int, prompt: str, images: dict, label: Optional[str]) -> int:
        with self.tx() as c:
            cur = c.execute("INSERT INTO set_tasks (set_id, created_at, prompt, images, label, created_by, labelled_by)"
                            " VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (set_id, time.time(), prompt, json.dumps(images), label, who(),
                             who() if label else ""))
            return int(cur.lastrowid)

    def get_set_task(self, task_id: int) -> Optional[dict]:
        row = self._one("SELECT * FROM set_tasks WHERE id = ?", (task_id,))
        return _task_row(row) if row else None

    def set_task_label(self, task_id: int, label: Optional[str]) -> None:
        with self.tx() as c:
            c.execute("UPDATE set_tasks SET label = ?, labelled_by = ? WHERE id = ?", (label, who(), task_id))

    def delete_set_task(self, task_id: int) -> None:
        with self.tx() as c:
            c.execute("DELETE FROM set_tasks WHERE id = ?", (task_id,))

    # --- knowledge (guidelines + learned lessons) -------------------------
    def add_knowledge(self, guidelines: str, lessons: list[str], source: str,
                      parent_id: Optional[int] = None, activate: bool = True, status: str = "accepted",
                      learned_from: Optional[list[int]] = None) -> int:
        """A new lessons version. status 'candidate' (with activate=False) is held back until it passes a test."""
        with self.tx() as c:
            if activate:
                c.execute("UPDATE knowledge SET active = 0")
            if status == "candidate":  # only one pending candidate at a time; it carries all earlier ones' lessons
                c.execute("UPDATE knowledge SET status = 'superseded' WHERE status = 'candidate'")
            cur = c.execute(
                "INSERT INTO knowledge (created_at, parent_id, guidelines, lessons, source, active, status,"
                " learned_from, author) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (time.time(), parent_id, guidelines, json.dumps(lessons), source, int(activate), status,
                 json.dumps(sorted(set(learned_from or []))), who()))
            return int(cur.lastrowid)

    def pending_candidate(self) -> Optional[dict]:
        row = self._one("SELECT * FROM knowledge WHERE status = 'candidate' ORDER BY id DESC LIMIT 1")
        return _knowledge_row(row) if row else None

    def set_knowledge_status(self, knowledge_id: int, status: str, gate_result: Optional[dict] = None) -> None:
        with self.tx() as c:
            c.execute("UPDATE knowledge SET status = ?, gate_result = ? WHERE id = ?",
                      (status, json.dumps(gate_result) if gate_result is not None else None, knowledge_id))

    def set_gate_result(self, knowledge_id: int, gate_result: dict) -> None:
        with self.tx() as c:
            c.execute("UPDATE knowledge SET gate_result = ? WHERE id = ?", (json.dumps(gate_result), knowledge_id))

    def get_knowledge(self, knowledge_id: int) -> Optional[dict]:
        row = self._one("SELECT * FROM knowledge WHERE id = ?", (knowledge_id,))
        return _knowledge_row(row) if row else None

    def active_knowledge(self) -> Optional[dict]:
        row = self._one("SELECT * FROM knowledge WHERE active = 1 ORDER BY id DESC LIMIT 1")
        return _knowledge_row(row) if row else None

    def list_knowledge(self) -> list[dict]:
        return [_knowledge_row(r) for r in self._all("SELECT * FROM knowledge ORDER BY id DESC")]

    def activate_knowledge(self, knowledge_id: Optional[int]) -> None:
        """Make a version active; None means judge with no guidelines or lessons."""
        with self.tx() as c:
            c.execute("UPDATE knowledge SET active = 0")
            if knowledge_id is not None:
                c.execute("UPDATE knowledge SET active = 1, status = 'accepted' WHERE id = ?", (knowledge_id,))


def _eval_row(row: sqlite3.Row, include_result: bool = True) -> dict:
    out = {
        "id": row["id"],
        "created_at": row["created_at"],
        "prompt": row["prompt"],
        "images": json.loads(row["images"]),
        "config": json.loads(row["config"]),
        "status": row["status"],
        "verdict": row["verdict"],
        "feedback": None if row["true_label"] is None else {
            "verdict_correct": None if row["verdict_correct"] is None else bool(row["verdict_correct"]),
            "true_label": row["true_label"],
        },
    }
    if include_result:
        out["result"] = json.loads(row["result"])
    return out


def _task_row(row: sqlite3.Row) -> dict:
    return {"id": row["id"], "set_id": row["set_id"], "created_at": row["created_at"], "prompt": row["prompt"],
            "images": json.loads(row["images"]), "label": row["label"],
            "created_by": row["created_by"], "labelled_by": row["labelled_by"]}


def _knowledge_row(row: sqlite3.Row) -> dict:
    return {"id": row["id"], "created_at": row["created_at"], "parent_id": row["parent_id"],
            "guidelines": row["guidelines"], "lessons": json.loads(row["lessons"]),
            "source": row["source"], "active": bool(row["active"]), "status": row["status"],
            "learned_from": json.loads(row["learned_from"]),
            "gate_result": json.loads(row["gate_result"]) if row["gate_result"] else None,
            "author": row["author"]}


def _bench_row(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "finished_at": row["finished_at"],
        "dataset": row["dataset"],
        "config": json.loads(row["config"]),
        "status": row["status"],
        "total": row["total"],
        "completed": row["completed"],
        "metrics": json.loads(row["metrics"]) if row["metrics"] else None,
        "note": row["note"],
        "mode": row["mode"],
        "set_id": row["set_id"],
        "knowledge_id": row["knowledge_id"],
        "learned_knowledge_id": row["learned_knowledge_id"],
        "baseline_of": row["baseline_of"],
    }


def open_db(settings=None) -> DB:
    """The configured backend: Postgres (Supabase) when IMAGE_JUDGE_DATABASE_URL is set, else local SQLite."""
    if settings is None:
        from .config import settings as default_settings
        settings = default_settings
    if settings.database_url:
        from .pgdb import PostgresDB
        return PostgresDB(settings.database_url)
    return DB(settings.db_path)
