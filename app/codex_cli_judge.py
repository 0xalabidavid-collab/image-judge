"""Judge calls through the local Codex CLI, using the user's existing Codex login.

Model names are "codex" (the CLI default) or "codex:<model-id>". Each call is
ephemeral, ignores repository/user instructions, runs in an empty read-only
workspace, and is constrained by the rubric's JSON schema.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from .config import JudgeConfig, settings
from .judge import JsonJudge, JudgeError


def cli_model(model: str) -> Optional[str]:
    """Convert codex:gpt-6-astra to gpt-6-astra; codex uses the CLI default."""
    _, separator, name = model.partition(":")
    return name if separator and name else None


def find_codex_binary() -> Optional[str]:
    return os.environ.get("IMAGE_JUDGE_CODEX_BIN") or shutil.which("codex")


def build_command(binary: str, config: JudgeConfig, schema_path: Path, output_path: Path,
                  image_paths: list[Path], workdir: Path) -> list[str]:
    cmd = [
        binary, "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--sandbox", "read-only",
        "--cd", str(workdir),
        "--output-schema", str(schema_path),
        "--output-last-message", str(output_path),
        "--color", "never",
        "--json",
        "-c", f'model_reasoning_effort="{config.effort}"',
    ]
    if model := cli_model(config.model):
        cmd += ["--model", model]
    for path in image_paths:
        cmd += ["--image", str(path)]
    return cmd


def _prompt(system: str, content: list) -> tuple[str, list]:
    """Replace interleaved image objects with stable numbered attachment markers."""
    parts = [system.strip(), "TASK INPUT"]
    images = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        else:
            images.append(item)
            parts.append(f"[ATTACHED IMAGE {len(images)}]")
    parts.append(
        "The images were attached in the exact numbered order shown above. Do not use tools or inspect "
        "unrelated files. Evaluate only the supplied prompt and images, then return the required JSON."
    )
    return "\n\n".join(parts), images


def _usage_from_events(stdout: str) -> dict:
    """Extract the last token-usage object emitted by Codex JSON events."""
    found = {"input_tokens": 0, "output_tokens": 0}

    def visit(value):
        nonlocal found
        if isinstance(value, dict):
            if "input_tokens" in value and "output_tokens" in value:
                found = {
                    "input_tokens": int(value.get("input_tokens") or 0),
                    "output_tokens": int(value.get("output_tokens") or 0),
                }
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for line in stdout.splitlines():
        try:
            visit(json.loads(line))
        except json.JSONDecodeError:
            continue
    return found


class CodexCliJudge(JsonJudge):
    def __init__(self, binary: Optional[str] = None, runner=None):
        self.binary = binary
        self._run = runner or subprocess.run

    async def complete_json(self, system: str, content: list, schema: dict, config: JudgeConfig) -> dict:
        binary = self.binary or find_codex_binary()
        if not binary:
            raise JudgeError(
                "Codex CLI not found. Install/open Codex and log in, or set IMAGE_JUDGE_CODEX_BIN.",
                fatal=True,
            )

        prompt, images = _prompt(system, content)
        try:
            with tempfile.TemporaryDirectory(prefix="image-judge-codex-") as temp:
                root = Path(temp)
                schema_path = root / "output.schema.json"
                output_path = root / "result.json"
                schema_path.write_text(json.dumps(schema), encoding="utf-8")
                image_paths = []
                for index, image in enumerate(images, 1):
                    extension = ".jpg" if image.media_type == "image/jpeg" else ".png"
                    path = root / f"image-{index:02d}{extension}"
                    path.write_bytes(base64.standard_b64decode(image.data_b64))
                    image_paths.append(path)

                cmd = build_command(binary, config, schema_path, output_path, image_paths, root)
                proc = await asyncio.to_thread(
                    self._run,
                    cmd,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=settings.api_timeout_s,
                    cwd=str(root),
                )
                raw = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
        except subprocess.TimeoutExpired as exc:
            raise JudgeError(f"Codex did not answer within {settings.api_timeout_s:.0f}s.") from exc
        except OSError as exc:
            raise JudgeError(f"Could not start Codex ({binary}): {exc}", fatal=True) from exc

        if proc.returncode != 0 or not raw.strip():
            detail = (proc.stderr or proc.stdout or "no output").strip()[:500]
            lower = detail.lower()
            fatal = any(token in lower for token in (
                "not logged in", "login", "authentication", "unauthorized", "invalid model",
                "model not found", "usage limit", "rate limit",
            ))
            raise JudgeError(f"Codex failed (exit {proc.returncode}): {detail}", fatal=fatal)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise JudgeError(f"Codex returned invalid JSON: {exc}") from exc
        return {
            "data": data,
            "model": cli_model(config.model) or "codex-default",
            "usage": _usage_from_events(proc.stdout or ""),
        }
