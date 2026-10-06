"""Runtime settings, read from environment variables (optionally a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Optional
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def provider_of(model: str) -> str:
    if model == "codex" or model.startswith("codex:"):
        return "codex"
    if model.startswith("claude-code"):
        return "claude-code"
    return "gemini" if model.startswith("gemini") else "anthropic"


def has_key(provider: str) -> bool:
    if provider == "codex":  # uses the local Codex CLI login instead of an API key
        import shutil
        return bool(os.environ.get("IMAGE_JUDGE_CODEX_BIN") or shutil.which("codex"))
    if provider == "claude-code":  # uses the local CLI's login instead of a key
        import shutil
        return bool(os.environ.get("IMAGE_JUDGE_CLAUDE_BIN") or shutil.which("claude"))
    if provider == "gemini":
        return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def _default_model() -> str:
    if os.environ.get("IMAGE_JUDGE_MODEL"):
        return os.environ["IMAGE_JUDGE_MODEL"]
    # Use whichever provider has a key; Gemini if only a Gemini key is present.
    if has_key("gemini") and not has_key("anthropic"):
        return "gemini-3.1-pro-preview"
    return "claude-fable-5-1"


@dataclass(frozen=True)
class JudgeConfig:
    """Everything that changes what the judge outputs. Part of the cache key."""

    model: str = field(default_factory=_default_model)
    runs: int = int(os.environ.get("IMAGE_JUDGE_RUNS", "4"))
    rubric_version: str = os.environ.get("IMAGE_JUDGE_RUBRIC", "v1")
    effort: str = os.environ.get("IMAGE_JUDGE_EFFORT", "high")
    max_tokens: int = int(os.environ.get("IMAGE_JUDGE_MAX_TOKENS", "32000"))
    # Anthropic only: server-side refusal fallback ("default" lets the API pick by refusal category).
    use_fallbacks: bool = os.environ.get("IMAGE_JUDGE_FALLBACKS", "1") != "0"
    # Knowledge: project guidelines + lessons learned in training (see app/learn.py).
    knowledge_id: Optional[int] = None
    guidelines: str = ""
    lessons: tuple[str, ...] = ()

    def with_knowledge(self, knowledge: Optional[dict]) -> "JudgeConfig":
        if not knowledge:
            return replace(self, knowledge_id=None, guidelines="", lessons=())
        return replace(self, knowledge_id=knowledge["id"], guidelines=knowledge.get("guidelines") or "",
                       lessons=tuple(knowledge.get("lessons") or ()))

    def with_overrides(self, **kwargs) -> "JudgeConfig":
        return replace(self, **{k: v for k, v in kwargs.items() if v is not None})

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "runs": self.runs,
            "rubric_version": self.rubric_version,
            "effort": self.effort,
            "max_tokens": self.max_tokens,
            "use_fallbacks": self.use_fallbacks,
            "knowledge_id": self.knowledge_id,
            "lessons": len(self.lessons),
            "has_guidelines": bool(self.guidelines.strip()),
        }


@dataclass
class Settings:
    db_path: Path = Path(os.environ.get("IMAGE_JUDGE_DB", str(ROOT / "data" / "image_judge.sqlite3")))
    upload_dir: Path = Path(os.environ.get("IMAGE_JUDGE_UPLOADS", str(ROOT / "data" / "uploads")))
    results_dir: Path = ROOT / "benchmarks" / "results"
    # Live storage (all optional). With database_url set, data lives in Postgres (Supabase) instead
    # of the local SQLite file; with the Supabase URL + service key set, images are mirrored to a
    # private Storage bucket. See supabase/README.md.
    database_url: str = os.environ.get("IMAGE_JUDGE_DATABASE_URL", "")
    supabase_url: str = os.environ.get("SUPABASE_URL", "")
    supabase_service_key: str = os.environ.get("SUPABASE_SERVICE_KEY", "")
    supabase_bucket: str = os.environ.get("SUPABASE_BUCKET", "image-judge")
    # Auto-training gate: lessons learned from corrections stay switched off until they beat the current
    # lessons on tasks they were not learned from (see app/autotrain.py).
    auto_gate: bool = os.environ.get("IMAGE_JUDGE_AUTO_GATE", "1").lower() in ("1", "true", "yes", "on")
    gate_every: int = int(os.environ.get("IMAGE_JUDGE_GATE_EVERY", "5"))        # new corrections before a test
    gate_min_pool: int = int(os.environ.get("IMAGE_JUDGE_GATE_MIN_POOL", "12"))  # fewest unseen tasks to test on
    gate_max_tasks: int = int(os.environ.get("IMAGE_JUDGE_GATE_MAX_TASKS", "20"))  # most tasks per test (cost cap)
    gate_margin: float = float(os.environ.get("IMAGE_JUDGE_GATE_MARGIN", "0.10"))  # net gain needed, share of tasks
    # Image bucket on any S3-compatible service: Cloudflare R2 (recommended: 10 GB free, free downloads)
    # or Backblaze B2. If set, it is used for images instead of Supabase Storage.
    s3_endpoint_url: str = os.environ.get("S3_ENDPOINT_URL", "")   # e.g. https://<account>.r2.cloudflarestorage.com
    s3_access_key_id: str = os.environ.get("S3_ACCESS_KEY_ID", "")
    s3_secret_access_key: str = os.environ.get("S3_SECRET_ACCESS_KEY", "")
    s3_bucket: str = os.environ.get("S3_BUCKET", "image-judge")
    s3_region: str = os.environ.get("S3_REGION", "auto")           # R2: auto. B2: its region, e.g. us-west-004
    # Hosted mode: every page and API call needs a Supabase login (see app/auth.py). The anon key is
    # public by design (it can only sign people in); never put the service key in a page.
    supabase_anon_key: str = os.environ.get("SUPABASE_ANON_KEY", "")
    auth_required: bool = os.environ.get("IMAGE_JUDGE_AUTH", "").lower() in ("1", "true", "yes", "on")
    # Owners (comma-separated emails). Only they can run training/benchmarks and change which lessons are in
    # use. Empty = everyone who can sign in is an owner, so set this before inviting labellers.
    admin_emails: frozenset = frozenset(
        e.strip().lower() for e in os.environ.get("IMAGE_JUDGE_ADMIN_EMAILS", "").split(",") if e.strip())
    # Optional extra allowlist (comma-separated emails). Empty = any user of the Supabase project.
    allowed_emails: frozenset = frozenset(
        e.strip().lower() for e in os.environ.get("IMAGE_JUDGE_ALLOWED_EMAILS", "").split(",") if e.strip())
    # Max simultaneous API calls across the whole process (rate-limit friendliness).
    max_concurrency: int = int(os.environ.get("IMAGE_JUDGE_MAX_CONCURRENCY", "4"))
    # SDK-level retries for 429 / 5xx / connection errors (exponential backoff).
    api_max_retries: int = int(os.environ.get("IMAGE_JUDGE_API_RETRIES", "5"))
    api_timeout_s: float = float(os.environ.get("IMAGE_JUDGE_API_TIMEOUT", "600"))
    max_upload_bytes: int = 25 * 1024 * 1024
    judge: JudgeConfig = field(default_factory=JudgeConfig)


settings = Settings()
