"""Image files: a local folder, optionally mirrored to a private cloud bucket.

Images are content-addressed (<sha256>.<ext>), so a name identifies exactly one file. The database
stores only that name (older rows hold a full local path; the name is its last part). With a bucket
configured, every saved image is also uploaded, and any image missing locally is downloaded on first
use, so the app works on a fresh machine with nothing but the database and the bucket.

The bucket can be Cloudflare R2 or Backblaze B2 (anything S3-compatible), or Supabase Storage.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Optional, Protocol, Union
from urllib.parse import quote

import httpx

from .config import Settings, settings


class StorageError(RuntimeError):
    pass


def image_name(path_or_name: Union[str, Path]) -> str:
    """'C:\\...\\uploads\\ab12.png', '/x/ab12.png' or 'ab12.png' -> 'ab12.png'."""
    return str(path_or_name).replace("\\", "/").rsplit("/", 1)[-1]


class RemoteBackend(Protocol):
    label: str

    def size(self, name: str) -> Optional[int]:
        """Size of the stored object, or None if it is not there."""

    def put(self, name: str, raw: bytes) -> None: ...

    def get(self, name: str) -> bytes:
        """The object's bytes; FileNotFoundError if it is not there."""

    def ensure_bucket(self) -> None: ...


# --- S3-compatible: Cloudflare R2, Backblaze B2, AWS S3, MinIO ---------------------------------
class S3Backend:
    NOT_FOUND = {"404", "NoSuchKey", "NotFound", "NoSuchBucket"}

    def __init__(self, client, bucket: str, label: str = "S3 storage"):
        self.client = client
        self.bucket = bucket
        self.label = label

    @classmethod
    def from_settings(cls, cfg: Settings) -> "S3Backend":
        import boto3
        from botocore.config import Config

        client = boto3.client(
            "s3",
            endpoint_url=cfg.s3_endpoint_url,
            aws_access_key_id=cfg.s3_access_key_id,
            aws_secret_access_key=cfg.s3_secret_access_key,
            region_name=cfg.s3_region or "auto",  # R2 uses "auto"; B2 wants its own region, e.g. us-west-004
            config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"},
                          s3={"addressing_style": "path"}),
        )
        host = cfg.s3_endpoint_url.split("//")[-1]
        label = "Cloudflare R2" if "r2.cloudflarestorage.com" in host else \
            "Backblaze B2" if "backblazeb2.com" in host else "S3 storage"
        return cls(client, cfg.s3_bucket, label)

    @staticmethod
    def _code(exc) -> str:
        return str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))

    def _fail(self, what: str, exc: Exception) -> StorageError:
        return StorageError(f"{self.label}: {what} failed ({self._code(exc) or type(exc).__name__}: {exc})")

    def size(self, name: str) -> Optional[int]:
        from botocore.exceptions import BotoCoreError, ClientError
        try:
            return int(self.client.head_object(Bucket=self.bucket, Key=name)["ContentLength"])
        except ClientError as exc:
            if self._code(exc) in self.NOT_FOUND:
                return None
            raise self._fail(f"check of {name}", exc) from exc
        except BotoCoreError as exc:
            raise self._fail(f"check of {name}", exc) from exc

    def put(self, name: str, raw: bytes) -> None:
        from botocore.exceptions import BotoCoreError, ClientError
        try:
            self.client.put_object(Bucket=self.bucket, Key=name, Body=raw,
                                   ContentType=mimetypes.guess_type(name)[0] or "application/octet-stream")
        except (ClientError, BotoCoreError) as exc:
            raise self._fail(f"upload of {name}", exc) from exc

    def get(self, name: str) -> bytes:
        from botocore.exceptions import BotoCoreError, ClientError
        try:
            return self.client.get_object(Bucket=self.bucket, Key=name)["Body"].read()
        except ClientError as exc:
            if self._code(exc) in self.NOT_FOUND:
                raise FileNotFoundError(name) from exc
            raise self._fail(f"download of {name}", exc) from exc
        except BotoCoreError as exc:
            raise self._fail(f"download of {name}", exc) from exc

    def ensure_bucket(self) -> None:
        """Check the bucket is reachable; create it if it is missing and the key is allowed to."""
        from botocore.exceptions import BotoCoreError, ClientError
        try:
            self.client.list_objects_v2(Bucket=self.bucket, MaxKeys=1)  # works with object-level keys too
            return
        except ClientError as exc:
            code = self._code(exc)
            if code not in self.NOT_FOUND:
                raise self._fail(f"access to bucket {self.bucket!r}", exc) from exc
        except BotoCoreError as exc:
            raise self._fail(f"access to bucket {self.bucket!r}", exc) from exc
        try:
            self.client.create_bucket(Bucket=self.bucket)
        except (ClientError, BotoCoreError) as exc:
            raise StorageError(f"{self.label}: bucket {self.bucket!r} does not exist and could not be created "
                               f"with this key. Create it in the dashboard (private), then run again.") from exc


# --- Supabase Storage ---------------------------------------------------------------------------
class SupabaseBackend:
    label = "Supabase Storage"

    def __init__(self, url: str, service_key: str, bucket: str = "image-judge",
                 client: Optional[httpx.Client] = None):
        self.url = url.rstrip("/")
        self.key = service_key
        self.bucket = bucket
        self._client = client

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=120)
        return self._client

    def _headers(self, **extra: str) -> dict:
        return {"Authorization": f"Bearer {self.key}", "apikey": self.key, **extra}

    def _object_url(self, name: str, authenticated: bool = False) -> str:
        kind = "authenticated/" if authenticated else ""
        return f"{self.url}/storage/v1/object/{kind}{self.bucket}/{quote(name)}"

    def size(self, name: str) -> Optional[int]:
        r = self._http().head(self._object_url(name, True), headers=self._headers())
        if r.status_code in (400, 404):
            return None
        if r.status_code != 200:
            raise StorageError(f"Storage check for {name} failed: HTTP {r.status_code}")
        return int(r.headers.get("content-length", -1))

    def put(self, name: str, raw: bytes) -> None:
        ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
        r = self._http().post(self._object_url(name), content=raw,
                              headers=self._headers(**{"Content-Type": ctype, "x-upsert": "true"}))
        if r.status_code not in (200, 201):
            raise StorageError(f"Upload of {name} failed: HTTP {r.status_code} {r.text[:200]}")

    def get(self, name: str) -> bytes:
        r = self._http().get(self._object_url(name, True), headers=self._headers())
        if r.status_code in (400, 404):
            raise FileNotFoundError(name)
        if r.status_code != 200:
            raise StorageError(f"Download of {name} failed: HTTP {r.status_code}")
        return r.content

    def ensure_bucket(self) -> None:
        r = self._http().post(f"{self.url}/storage/v1/bucket", headers=self._headers(**{"Content-Type": "application/json"}),
                              json={"id": self.bucket, "name": self.bucket, "public": False})
        if r.status_code in (200, 201):
            return
        if r.status_code in (400, 409) and "already exists" in r.text.lower():
            return
        raise StorageError(f"Could not create bucket {self.bucket!r}: HTTP {r.status_code} {r.text[:200]}")


def make_backend(cfg: Settings) -> Optional[RemoteBackend]:
    """The cloud bucket the settings describe, or None for local-only. S3-compatible wins if both are set."""
    if cfg.s3_endpoint_url and cfg.s3_access_key_id and cfg.s3_secret_access_key:
        return S3Backend.from_settings(cfg)
    if cfg.supabase_url and cfg.supabase_service_key:
        return SupabaseBackend(cfg.supabase_url, cfg.supabase_service_key, cfg.supabase_bucket)
    return None


# --- the store the app uses ---------------------------------------------------------------------
class ImageStore:
    def __init__(self, local_dir: Path, backend: Optional[RemoteBackend] = None):
        self.local_dir = Path(local_dir)
        self.backend = backend

    @property
    def remote_enabled(self) -> bool:
        return self.backend is not None

    # remote operations (only valid when a backend is configured)
    def remote_size(self, name: str) -> Optional[int]:
        return self.backend.size(image_name(name))

    def upload(self, name: str, raw: bytes) -> None:
        self.backend.put(image_name(name), raw)

    def ensure_bucket(self) -> None:
        self.backend.ensure_bucket()

    # local + remote
    def local_path(self, name: str) -> Path:
        return self.local_dir / image_name(name)

    def save(self, name: str, raw: bytes) -> Path:
        """Write the file locally (atomically) and, if configured, to the bucket."""
        name = image_name(name)
        path = self.local_path(name)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.stat().st_size != len(raw):
            tmp = path.with_suffix(path.suffix + ".part")
            try:
                tmp.write_bytes(raw)
                tmp.replace(path)
            except OSError:
                tmp.unlink(missing_ok=True)
                raise
        if self.remote_enabled:
            self.upload(name, raw)
        return path

    def ensure(self, path_or_name: Union[str, Path]) -> Path:
        """Return a readable local path, downloading from the bucket if the file is not here."""
        name = image_name(path_or_name)
        path = self.local_path(name)
        if path.is_file() and path.stat().st_size > 0:
            return path
        # An old row may point at a file elsewhere on this machine.
        original = Path(str(path_or_name))
        if original.is_file() and original.stat().st_size > 0:
            return original
        if not self.remote_enabled:
            raise FileNotFoundError(f"Image {name} is not in {self.local_dir}")
        raw = self.backend.get(name)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(raw)
        tmp.replace(path)
        return path


_store: Optional[ImageStore] = None
_store_key: tuple = ()


def image_path(path_or_name: Union[str, Path]) -> Path:
    """A local path for an image, fetched from the bucket if needed. If it cannot be found or fetched, the
    expected local path is returned anyway, so that one task fails when it is judged ("could not read image")
    instead of a whole run failing before it starts."""
    try:
        return get_store().ensure(path_or_name)
    except (FileNotFoundError, StorageError):
        return get_store().local_path(image_name(path_or_name))


def _settings_key(cfg: Settings) -> tuple:
    return (cfg.upload_dir, cfg.supabase_url, cfg.supabase_service_key, cfg.supabase_bucket,
            cfg.s3_endpoint_url, cfg.s3_access_key_id, cfg.s3_secret_access_key, cfg.s3_bucket, cfg.s3_region)


def get_store() -> ImageStore:
    """The store for the current settings (rebuilt if they change, e.g. a different upload folder)."""
    global _store, _store_key
    key = _settings_key(settings)
    if _store is None or key != _store_key:
        _store = ImageStore(settings.upload_dir, make_backend(settings))
        _store_key = key
    return _store


def reset_store() -> None:
    global _store, _store_key
    _store, _store_key = None, ()
