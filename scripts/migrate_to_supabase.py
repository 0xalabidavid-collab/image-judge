"""Copy the local SQLite data and uploaded images to Supabase (Postgres + Storage).

    python scripts/migrate_to_supabase.py             # dry run: shows exactly what would be copied
    python scripts/migrate_to_supabase.py --apply     # do it

Needs IMAGE_JUDGE_DATABASE_URL in .env plus an image bucket: either the S3_* settings (Cloudflare R2 or
Backblaze B2) or SUPABASE_URL + SUPABASE_SERVICE_KEY (see supabase/README.md).
Safe to run again: rows keep their ids and are skipped if already there, images are skipped if the bucket
already holds a file of the same size. The local SQLite file and images are never modified or deleted.
"""
import argparse
import json
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.storage import ImageStore, image_name, make_backend  # noqa: E402

# Parents before children (foreign keys).
TABLES = ["task_sets", "knowledge", "evaluations", "feedback", "set_tasks", "benchmark_runs",
          "benchmark_items", "judge_cache"]
HAS_ID = {"task_sets", "knowledge", "evaluations", "set_tasks", "benchmark_runs"}
IMAGE_JSON = {"evaluations", "set_tasks"}  # tables whose `images` column lists image files


def normalise_images(raw: str) -> str:
    img = json.loads(raw)
    return json.dumps({"originals": [image_name(p) for p in img["originals"]],
                       "a": image_name(img["a"]), "b": image_name(img["b"])})


def image_names(rows) -> set[str]:
    names: set[str] = set()
    for r in rows:
        img = json.loads(r["images"])
        names.update(image_name(p) for p in [*img["originals"], img["a"], img["b"]])
    return names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually copy (default is a dry run)")
    ap.add_argument("--sqlite", default=str(settings.db_path), help="source SQLite file")
    ap.add_argument("--uploads", default=str(settings.upload_dir), help="source images folder")
    ap.add_argument("--workers", type=int, default=4, help="parallel image uploads")
    args = ap.parse_args()

    src = sqlite3.connect(f"file:{args.sqlite}?mode=ro", uri=True)
    src.row_factory = sqlite3.Row
    data = {t: src.execute(f"SELECT * FROM {t}").fetchall() for t in TABLES}
    eval_ids = {r["id"] for r in data["evaluations"]}
    orphans = [r for r in data["feedback"] if r["evaluation_id"] not in eval_ids]
    data["feedback"] = [r for r in data["feedback"] if r["evaluation_id"] in eval_ids]

    uploads = Path(args.uploads)
    names = sorted(image_names(data["evaluations"]) | image_names(data["set_tasks"]))
    ok, empty, absent = [], [], []
    for n in names:
        p = uploads / n
        (absent if not p.exists() else empty if p.stat().st_size == 0 else ok).append(n)
    total_bytes = sum((uploads / n).stat().st_size for n in ok)

    print(f"Source: {args.sqlite}")
    for t in TABLES:
        print(f"  {t:16} {len(data[t]):6} rows")
    if orphans:
        print(f"  (skipping {len(orphans)} feedback rows whose evaluation no longer exists)")
    print(f"  images: {len(ok)} files, {total_bytes / 2**20:.0f} MB"
          + (f"; {len(empty)} empty and {len(absent)} missing on disk (cannot be copied)" if empty or absent else ""))
    if total_bytes > 1 * 2**30:
        print("  NOTE: that is over the 1 GB free Storage allowance of Supabase. Cloudflare R2 and Backblaze B2 "
              "include 10 GB free.")
    if not args.apply:
        print("\nDry run only. Run again with --apply to copy.")
        return 0

    if not settings.database_url:
        sys.exit("Set IMAGE_JUDGE_DATABASE_URL in .env first.")
    backend = make_backend(settings)
    if backend is None:
        sys.exit("No image bucket is configured. Set S3_ENDPOINT_URL, S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY "
                 "(Cloudflare R2 or Backblaze B2), or SUPABASE_URL and SUPABASE_SERVICE_KEY, in .env first.")

    from app.pgdb import PostgresDB  # noqa: E402
    db = PostgresDB(settings.database_url)  # also creates the tables
    conn = db._conn.conn

    print("\nCopying rows...")
    for t in TABLES:
        rows = data[t]
        if not rows:
            continue
        cols = list(rows[0].keys())
        values = []
        for r in rows:
            d = dict(r)
            if t in IMAGE_JSON:
                d["images"] = normalise_images(d["images"])
            values.append(tuple(d[c] for c in cols))
        override = " OVERRIDING SYSTEM VALUE" if t in HAS_ID else ""
        sql = (f"INSERT INTO {t} ({', '.join(cols)}){override} VALUES ({', '.join(['%s'] * len(cols))})"
               " ON CONFLICT DO NOTHING")
        with conn.transaction():
            with conn.cursor() as cur:
                cur.executemany(sql, values)
        print(f"  {t}: {len(rows)} rows sent")
    with conn.transaction():
        for t in sorted(HAS_ID):  # new rows must not collide with the ids copied above
            conn.execute(f"SELECT setval(pg_get_serial_sequence('{t}', 'id'),"
                         f" COALESCE((SELECT MAX(id) FROM {t}), 1), (SELECT MAX(id) FROM {t}) IS NOT NULL)")

    print(f"\nUploading {len(ok)} images with {args.workers} workers...")
    store = ImageStore(uploads, backend)
    print(f"  destination: {backend.label}")
    store.ensure_bucket()
    done = {"up": 0, "skip": 0}

    def push(name: str) -> None:
        raw = (uploads / name).read_bytes()
        if store.remote_size(name) == len(raw):
            done["skip"] += 1
        else:
            store.upload(name, raw)
            done["up"] += 1

    failed: list[tuple[str, str]] = []

    def guarded(name: str) -> None:
        try:
            push(name)
        except Exception as exc:  # report every failure, don't stop at the first
            failed.append((name, str(exc)[:120]))

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(guarded, ok))
    print(f"  uploaded {done['up']}, already there {done['skip']}, failed {len(failed)}")
    for name, why in failed[:10]:
        print(f"    {name}: {why}")

    print("\nVerifying...")
    bad = 0
    for t in TABLES:
        n_dest = conn.execute(f"SELECT COUNT(*) AS n FROM {t}").fetchone()["n"]
        expected = len(data[t])
        flag = "ok" if n_dest >= expected else "MISSING ROWS"
        bad += flag != "ok"
        print(f"  {t:16} local {expected:6}  supabase {n_dest:6}  {flag}")
    mismatched = [n for n in ok if store.remote_size(n) != (uploads / n).stat().st_size]
    print(f"  images: {len(ok) - len(mismatched)}/{len(ok)} present with matching size")
    if bad or mismatched or failed:
        print("\nNot everything copied. Fix the errors above and run again; it will only copy what is missing.")
        return 1
    print("\nAll copied and verified. To use Supabase, keep the three settings in .env and restart the app.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
