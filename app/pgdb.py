"""Postgres (Supabase) backend. Subclasses the SQLite DB so every query method is shared; only the
connection and the few SQLite-only statements (INSERT OR REPLACE, lastrowid, '?' placeholders) differ."""

from __future__ import annotations

import re
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from .db import DB

SCHEMA_FILE = Path(__file__).resolve().parent.parent / "supabase" / "schema.sql"

# Tables whose primary key is a generated id: INSERTs get RETURNING id so cursor.lastrowid works.
ID_TABLES = {"evaluations", "benchmark_runs", "task_sets", "set_tasks", "knowledge"}

# INSERT OR REPLACE INTO <table> VALUES (...)  ->  an explicit upsert. (columns, conflict target)
UPSERTS = {
    "judge_cache": (["key", "created_at", "judgment"], ["key"]),
    "benchmark_items": (["run_id", "task_id", "label", "status", "verdict", "correct", "result"],
                        ["run_id", "task_id"]),
    "feedback": (None, ["evaluation_id"]),  # written with an explicit column list
}

_OR_REPLACE = re.compile(r"INSERT\s+OR\s+REPLACE\s+INTO\s+(\w+)\s*(\([^)]*\))?\s*(VALUES\s*\(.*\))\s*$",
                         re.IGNORECASE | re.DOTALL)
_INSERT = re.compile(r"^\s*INSERT\s+INTO\s+(\w+)", re.IGNORECASE)


def translate(sql: str) -> tuple[str, bool]:
    """SQLite-flavoured SQL -> Postgres. Returns (sql, wants_returning_id)."""
    m = _OR_REPLACE.search(sql)
    if m:
        table, cols_text, values = m.group(1), m.group(2), m.group(3)
        default_cols, conflict = UPSERTS[table]
        cols = [c.strip() for c in cols_text.strip("() ").split(",")] if cols_text else list(default_cols)
        updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c not in conflict)
        sql = (f"INSERT INTO {table} ({', '.join(cols)}) {values} "
               f"ON CONFLICT ({', '.join(conflict)}) DO UPDATE SET {updates}")
    sql = sql.replace("?", "%s")
    m = _INSERT.match(sql)
    returning = bool(m and m.group(1).lower() in ID_TABLES and "returning" not in sql.lower())
    if returning:
        sql += " RETURNING id"
    return sql, returning


class _Cursor:
    def __init__(self, cur, lastrowid: Optional[int] = None):
        self._cur = cur
        self.lastrowid = lastrowid

    @property
    def rowcount(self) -> int:
        return self._cur.rowcount

    def fetchone(self):
        return self._cur.fetchone()

    def fetchall(self):
        return self._cur.fetchall()


class _Conn:
    """Just enough of the sqlite3 connection interface for DB: execute() returning a cursor."""

    def __init__(self, url: str):
        self.url = url
        self._c = None
        self._connect()

    def _connect(self):
        import psycopg
        from psycopg.rows import dict_row
        from psycopg.types.string import TextLoader

        # autocommit: each statement outside tx() commits at once (no idle open transactions);
        # prepare_threshold=None: required behind Supabase's transaction-mode pooler.
        self._c = psycopg.connect(self.url, autocommit=True, row_factory=dict_row, prepare_threshold=None)
        # jsonb comes back as text, exactly like the TEXT columns in SQLite (the app json.loads it).
        self._c.adapters.register_loader("jsonb", TextLoader)
        self._c.adapters.register_loader("json", TextLoader)

    @property
    def conn(self):
        if self._c is None or self._c.closed:
            self._connect()
        return self._c

    def execute(self, sql: str, args: tuple = ()) -> _Cursor:
        import psycopg
        sql, returning = translate(sql)
        for attempt in (0, 1):
            try:
                cur = self.conn.execute(sql, args)
                break
            except psycopg.OperationalError:
                # A dropped connection (idle timeout, pooler restart): reconnect once, outside a transaction.
                if attempt or self._c is None or self._c.info.transaction_status != psycopg.pq.TransactionStatus.IDLE:
                    raise
                self._c = None
        if returning:
            return _Cursor(cur, int(cur.fetchone()["id"]))
        return _Cursor(cur)

    def executescript(self, script: str) -> None:
        self.conn.execute(script)

    def close(self) -> None:
        if self._c is not None and not self._c.closed:
            self._c.close()


class PostgresDB(DB):
    def __init__(self, url: str, apply_schema: bool = True):
        self.url = url
        self._lock = threading.RLock()
        self._conn = _Conn(url)
        if apply_schema:
            self._conn.executescript(SCHEMA_FILE.read_text(encoding="utf-8"))

    @contextmanager
    def tx(self) -> Iterator[_Conn]:
        """Statements inside commit together or not at all."""
        with self._lock:
            with self._conn.conn.transaction():
                yield self._conn
