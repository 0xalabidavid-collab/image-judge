"""Judge calls through the local Claude Code CLI (`claude -p`), using your Claude subscription.

Model names: "claude-code" (the CLI's default model) or "claude-code:<alias or id>",
e.g. "claude-code:fable", "claude-code:opus", "claude-code:claude-opus-5-5".

The images go to the CLI on stdin as the same base64 content blocks the API judge
sends, all tools are disabled, and --json-schema makes the CLI return output that
matches the rubric schema. Works only on a machine where `claude` is installed
and logged in.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from .config import JudgeConfig, settings
from .judge import JsonJudge, JudgeError, anthropic_blocks

PREFIX = "claude-code"


def cli_model(model: str) -> Optional[str]:
    """'claude-code:fable' -> 'fable'; 'claude-code' -> None (CLI default)."""
    _, _, alias = model.partition(":")
    return alias or None


def find_claude_binary() -> Optional[str]:
    if os.environ.get("IMAGE_JUDGE_CLAUDE_BIN"):
        return os.environ["IMAGE_JUDGE_CLAUDE_BIN"]
    found = shutil.which("claude")
    if not found:
        return None
    # On Windows `claude` is a .cmd wrapper; cmd.exe would mangle the quotes in our
    # JSON arguments, so call the real executable it points to.
    if found.lower().endswith((".cmd", ".bat")):
        exe = Path(found).parent / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
        if exe.exists():
            return str(exe)
    return found


def build_command(binary: str, config: JudgeConfig, system: str, schema: dict) -> list[str]:
    cmd = [
        binary, "-p",
        "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
        "--tools", "",                        # judge only: no file, shell or web access
        "--system-prompt", system,
        "--json-schema", json.dumps(schema),
        "--effort", config.effort,
        "--no-session-persistence",           # don't fill the session history with judge calls
        "--safe-mode",                        # ignore CLAUDE.md, hooks, plugins, MCP servers
    ]
    if model := cli_model(config.model):
        cmd += ["--model", model]
    return cmd


def parse_events(stdout: str) -> tuple[Optional[dict], Optional[str]]:
    """Return (result event, model reported at init) from the CLI's stream-json output."""
    result, model = None, None
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "system" and event.get("subtype") == "init":
            model = event.get("model")
        elif event.get("type") == "result":
            result = event
    return result, model


class ClaudeCodeJudge(JsonJudge):
    def __init__(self, binary: Optional[str] = None, runner=None):
        self.binary = binary
        self._run = runner or subprocess.run  # injectable for tests

    async def complete_json(self, system: str, content: list, schema: dict, config: JudgeConfig) -> dict:
        binary = self.binary or find_claude_binary()
        if not binary:
            raise JudgeError("Claude Code CLI not found. Install it and log in, or set IMAGE_JUDGE_CLAUDE_BIN.",
                             fatal=True)
        message = {"type": "user", "message": {"role": "user", "content": anthropic_blocks(content)}}
        stdin = json.dumps(message) + "\n"
        cmd = build_command(binary, config, system, schema)

        try:
            # A worker thread rather than asyncio subprocesses: those fail under some
            # Windows event loops (e.g. uvicorn --reload).
            proc = await asyncio.to_thread(
                self._run, cmd, input=stdin, capture_output=True, text=True, encoding="utf-8",
                timeout=settings.api_timeout_s, cwd=tempfile.gettempdir(),
            )
        except subprocess.TimeoutExpired as exc:
            raise JudgeError(f"Claude Code did not answer within {settings.api_timeout_s:.0f}s.") from exc
        except OSError as exc:
            raise JudgeError(f"Could not start Claude Code ({binary}): {exc}", fatal=True) from exc

        result, model = parse_events(proc.stdout or "")
        if result is None:
            err = (proc.stderr or proc.stdout or "").strip()[:400]
            fatal = any(s in err.lower() for s in ("not logged in", "login", "unknown option", "invalid model"))
            raise JudgeError(f"Claude Code failed (exit {proc.returncode}): {err or 'no output'}", fatal=fatal)
        if result.get("is_error") or result.get("subtype") != "success":
            status = result.get("api_error_status")
            detail = str(result.get("result") or result.get("subtype") or "")[:300]
            if status == 429 or "limit" in detail.lower():
                raise JudgeError(f"Claude usage limit reached: {detail}", fatal=True)
            if status in (401, 403) or "log in" in detail.lower() or "login" in detail.lower():
                raise JudgeError(f"Claude Code is not logged in: {detail}", fatal=True)
            if status in (400, 404):
                raise JudgeError(f"Claude Code rejected the request: {detail}", fatal=True)
            raise JudgeError(f"Claude Code error ({status or result.get('subtype')}): {detail}")

        data = result.get("structured_output")
        if data is None:
            try:
                data = json.loads(result.get("result") or "")
            except json.JSONDecodeError as exc:
                raise JudgeError(f"Judge returned invalid JSON: {exc}") from exc
        usage = result.get("usage") or {}
        return {
            "data": data,
            "model": model or config.model,
            "usage": {
                "input_tokens": (usage.get("input_tokens") or 0) + (usage.get("cache_read_input_tokens") or 0)
                + (usage.get("cache_creation_input_tokens") or 0),
                "output_tokens": usage.get("output_tokens") or 0,
            },
        }
