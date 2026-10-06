"""Offline checks of the live backends: SQL translation, schema syntax, and the image stores.

There is no Postgres server in the test run, so the SQL is checked with Postgres's own grammar
(pglast) and the bucket calls with fakes. The live path is checked by running
scripts/migrate_to_supabase.py against real accounts.
"""
import io
import re
from pathlib import Path

import httpx
import pytest
from botocore.exceptions import ClientError

from app.config import Settings
from app.pgdb import SCHEMA_FILE, translate
from app.storage import (ImageStore, S3Backend, StorageError, SupabaseBackend, image_name,
                         make_backend)

pglast = pytest.importorskip("pglast")


def sql_statements_in_db_module() -> list[str]:
    """Every SQL string the shared DB class sends, found in its source."""
    src = Path(__file__).resolve().parent.parent / "app" / "db.py"
    text = src.read_text(encoding="utf-8")
    body = text[text.index("class DB:"):]
    found = re.findall(r'"((?:SELECT|INSERT|UPDATE|DELETE)[^"]*)"((?:\s*\n\s*"[^"]*")*)', body)
    return ["".join([first, *re.findall(r'"([^"]*)"', rest)]) for first, rest in found]


def test_every_db_statement_is_valid_postgres_after_translation():
    statements = sql_statements_in_db_module()
    assert len(statements) > 25  # the scan found the queries
    for sql in statements:
        out, _ = translate(sql)
        pglast.parse_sql(out.replace("%s", "NULL"))


def test_or_replace_becomes_an_upsert():
    sql, returning = translate("INSERT OR REPLACE INTO judge_cache VALUES (?, ?, ?)")
    assert "ON CONFLICT (key) DO UPDATE SET created_at = EXCLUDED.created_at, judgment = EXCLUDED.judgment" in sql
    assert not returning
    sql, _ = translate("INSERT OR REPLACE INTO feedback (evaluation_id, created_at, verdict_correct, true_label,"
                       " reason) VALUES (?, ?, ?, ?, ?)")
    assert "ON CONFLICT (evaluation_id) DO UPDATE SET" in sql and "reason = EXCLUDED.reason" in sql
    sql, _ = translate("INSERT OR REPLACE INTO benchmark_items VALUES (?, ?, ?, ?, ?, ?, ?)")
    assert "(run_id, task_id, label, status, verdict, correct, result)" in sql
    assert "ON CONFLICT (run_id, task_id)" in sql


def test_inserts_into_id_tables_return_the_id():
    sql, returning = translate("INSERT INTO task_sets (created_at, name) VALUES (?, ?)")
    assert returning and sql.endswith("RETURNING id") and "%s" in sql and "?" not in sql
    assert translate("INSERT INTO feedback (evaluation_id) VALUES (?)")[1] is False
    assert translate("UPDATE knowledge SET active = 0")[1] is False


def test_schema_parses_and_covers_every_table():
    schema = SCHEMA_FILE.read_text(encoding="utf-8")
    stmts = pglast.parse_sql(schema)
    assert len(stmts) >= 16
    tables = set(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", schema))
    assert tables == {"evaluations", "feedback", "judge_cache", "task_sets", "set_tasks", "knowledge",
                      "benchmark_runs", "benchmark_items"}
    # No table may be left open to Supabase's public API.
    for t in tables:
        assert f"ALTER TABLE {t}" in re.sub(r"\s+", " ", schema)
    assert "CREATE POLICY" not in schema.upper()


def test_schema_columns_match_the_sqlite_schema():
    from app.db import MIGRATIONS, SCHEMA
    lite: dict[str, set[str]] = {}
    for name, body in re.findall(r"CREATE TABLE IF NOT EXISTS (\w+) \((.*?)\n\);", SCHEMA, re.S):
        lite[name] = {m.group(1) for line in body.splitlines()
                      if (m := re.match(r"\s+(\w+)\s+(?:INTEGER|TEXT|REAL)", line))}
    for table, cols in MIGRATIONS.items():
        lite[table] |= set(cols)
    schema = SCHEMA_FILE.read_text(encoding="utf-8")
    for name, body in re.findall(r"CREATE TABLE IF NOT EXISTS (\w+) \((.*?)\n\);", schema, re.S):
        pg = {m.group(1) for line in body.splitlines()
              if (m := re.match(r"\s+(\w+)\s+(?:BIGINT|INTEGER|TEXT|DOUBLE|JSONB)", line))}
        assert pg == lite[name], f"{name}: {pg ^ lite[name]}"


# --- image store: Supabase Storage -------------------------------------------
def fake_storage(objects: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer service-key"
        path = request.url.path
        name = path.rsplit("/", 1)[-1]
        if request.method == "POST" and path.endswith("/storage/v1/bucket"):
            return httpx.Response(409, text="Bucket already exists")
        if request.method == "POST":
            assert request.headers["x-upsert"] == "true"
            objects[name] = request.content
            return httpx.Response(200, json={})
        if name not in objects:
            return httpx.Response(404, json={"error": "not found"})
        if request.method == "HEAD":
            return httpx.Response(200, headers={"content-length": str(len(objects[name]))})
        return httpx.Response(200, content=objects[name])
    return httpx.Client(transport=httpx.MockTransport(handler))


def make_store(tmp_path, objects):
    backend = SupabaseBackend("https://x.supabase.co", "service-key", "image-judge", client=fake_storage(objects))
    return ImageStore(tmp_path / "up", backend)


def test_image_name_accepts_paths_and_names():
    assert image_name("C:\\Users\\me\\data\\uploads\\ab12.png") == "ab12.png"
    assert image_name("/srv/uploads/ab12.png") == "ab12.png"
    assert image_name("ab12.png") == "ab12.png"


def test_save_writes_locally_and_to_the_bucket(tmp_path):
    objects: dict = {}
    store = make_store(tmp_path, objects)
    path = store.save("ab12.png", b"pixels")
    assert path.read_bytes() == b"pixels" and objects["ab12.png"] == b"pixels"
    assert store.remote_size("ab12.png") == 6 and store.remote_size("nope.png") is None


def test_ensure_downloads_a_missing_image(tmp_path):
    store = make_store(tmp_path, {"cd34.png": b"remote pixels"})
    path = store.ensure("C:\\old\\machine\\cd34.png")  # an old row holding a path from another computer
    assert path.read_bytes() == b"remote pixels" and path.parent == tmp_path / "up"


def test_ensure_repairs_an_empty_local_file(tmp_path):
    store = make_store(tmp_path, {"ef56.png": b"good"})
    (tmp_path / "up").mkdir()
    (tmp_path / "up" / "ef56.png").write_bytes(b"")
    assert store.ensure("ef56.png").read_bytes() == b"good"


def test_ensure_without_a_bucket_reports_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        ImageStore(tmp_path / "up").ensure("zz.png")


def test_ensure_bucket_tolerates_an_existing_bucket(tmp_path):
    make_store(tmp_path, {}).ensure_bucket()


def test_supabase_errors_are_reported():
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(500, text="boom")))
    with pytest.raises(StorageError):
        SupabaseBackend("https://x.supabase.co", "service-key", client=client).put("a.png", b"x")


# --- image store: Cloudflare R2 / Backblaze B2 (S3 interface) -----------------
class FakeS3:
    """Just the boto3 calls the app makes, with S3's error shapes."""

    def __init__(self, objects=None, buckets=("img",), can_create=True):
        self.objects = objects if objects is not None else {}
        self.buckets = set(buckets)
        self.can_create = can_create
        self.calls = []

    def _err(self, code, op):
        return ClientError({"Error": {"Code": code, "Message": code}}, op)

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise self._err("404", "HeadObject")
        return {"ContentLength": len(self.objects[Key])}

    def put_object(self, Bucket, Key, Body, ContentType):
        self.calls.append(("put", Key, ContentType))
        self.objects[Key] = Body

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise self._err("NoSuchKey", "GetObject")
        return {"Body": io.BytesIO(self.objects[Key])}

    def list_objects_v2(self, Bucket, MaxKeys):
        if Bucket not in self.buckets:
            raise self._err("NoSuchBucket", "ListObjectsV2")
        return {}

    def create_bucket(self, Bucket):
        if not self.can_create:
            raise self._err("AccessDenied", "CreateBucket")
        self.buckets.add(Bucket)


def s3_store(local_dir, fake):
    return ImageStore(local_dir, S3Backend(fake, "img", "Cloudflare R2"))


def test_s3_save_and_download_round_trip(tmp_path):
    fake = FakeS3()
    store = s3_store(tmp_path / "up", fake)
    store.save("ab12.png", b"pixels")
    assert fake.objects["ab12.png"] == b"pixels" and fake.calls == [("put", "ab12.png", "image/png")]
    assert store.remote_size("ab12.png") == 6 and store.remote_size("nope.png") is None
    fresh = s3_store(tmp_path / "fresh-machine", fake)  # nothing on disk: fetched from the bucket
    assert fresh.ensure("C:\\somewhere\\ab12.png").read_bytes() == b"pixels"


def test_s3_missing_object_is_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        s3_store(tmp_path / "up", FakeS3()).ensure("gone.png")


def test_s3_other_errors_are_reported_with_the_service_name(tmp_path):
    class Denied(FakeS3):
        def put_object(self, **kw):
            raise self._err("AccessDenied", "PutObject")
    with pytest.raises(StorageError, match=r"Cloudflare R2.*AccessDenied"):
        s3_store(tmp_path / "up", Denied()).save("a.png", b"x")


def test_s3_ensure_bucket_accepts_an_existing_bucket(tmp_path):
    fake = FakeS3()
    s3_store(tmp_path / "up", fake).ensure_bucket()
    assert fake.buckets == {"img"}


def test_s3_ensure_bucket_creates_a_missing_one(tmp_path):
    fake = FakeS3(buckets=())
    s3_store(tmp_path / "up", fake).ensure_bucket()
    assert "img" in fake.buckets


def test_s3_ensure_bucket_explains_when_it_cannot_create(tmp_path):
    with pytest.raises(StorageError, match="Create it in the dashboard"):
        s3_store(tmp_path / "up", FakeS3(buckets=(), can_create=False)).ensure_bucket()


def test_backend_choice_follows_the_settings():
    assert make_backend(Settings(s3_endpoint_url="", supabase_url="", supabase_service_key="")) is None
    sb = make_backend(Settings(s3_endpoint_url="", supabase_url="https://x.supabase.co", supabase_service_key="k"))
    assert isinstance(sb, SupabaseBackend)
    r2 = make_backend(Settings(s3_endpoint_url="https://acct.r2.cloudflarestorage.com", s3_access_key_id="id",
                               s3_secret_access_key="secret", s3_bucket="img", s3_region="auto",
                               supabase_url="https://x.supabase.co", supabase_service_key="k"))
    assert isinstance(r2, S3Backend) and r2.label == "Cloudflare R2" and r2.bucket == "img"  # S3 wins over Supabase
    assert r2.client.meta.endpoint_url == "https://acct.r2.cloudflarestorage.com"
    b2 = make_backend(Settings(s3_endpoint_url="https://s3.us-west-004.backblazeb2.com", s3_access_key_id="id",
                               s3_secret_access_key="secret", s3_region="us-west-004"))
    assert b2.label == "Backblaze B2"


def test_image_path_falls_back_so_one_bad_image_does_not_stop_a_run(tmp_path, monkeypatch):
    from app import storage
    monkeypatch.setattr(storage.settings, "upload_dir", tmp_path / "up", raising=False)
    monkeypatch.setattr(storage.settings, "s3_endpoint_url", "", raising=False)
    monkeypatch.setattr(storage.settings, "supabase_url", "", raising=False)
    (tmp_path / "up").mkdir()
    (tmp_path / "up" / "empty.png").write_bytes(b"")
    assert storage.image_path("empty.png") == tmp_path / "up" / "empty.png"   # empty file: no crash
    assert storage.image_path("C:/gone/missing.png") == tmp_path / "up" / "missing.png"  # absent: no crash
    (tmp_path / "up" / "ok.png").write_bytes(b"data")
    assert storage.image_path("ok.png").read_bytes() == b"data"
