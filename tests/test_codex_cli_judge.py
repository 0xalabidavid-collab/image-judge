"""Codex CLI judge command building, image attachment, parsing, and failures."""

import asyncio
import base64
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.codex_cli_judge import CodexCliJudge, build_command, cli_model, find_codex_binary
from app.config import JudgeConfig, provider_of
from app.images import prepare_image
from app.judge import JudgeError, Task, evaluate
from conftest import make_judgment, png_bytes

CFG = JudgeConfig(model="codex:gpt-6-astra", runs=2, rubric_version="v1", effort="high")


class FakeRun:
    def __init__(self, result=None, stdout="", stderr="", returncode=0, raise_exc=None):
        self.result = result
        self.stdout, self.stderr, self.returncode, self.raise_exc = stdout, stderr, returncode, raise_exc
        self.calls = []

    def __call__(self, cmd, **kwargs):
        output = cmd[cmd.index("--output-last-message") + 1]
        images = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "--image"]
        self.calls.append({**kwargs, "cmd": list(cmd), "images": [(p, Path(p).read_bytes()) for p in images]})
        if self.raise_exc:
            raise self.raise_exc
        if self.result is not None:
            result = self.result(len(self.calls) - 1) if callable(self.result) else self.result
            Path(output).write_text(json.dumps(result), encoding="utf-8")
        return SimpleNamespace(stdout=self.stdout, stderr=self.stderr, returncode=self.returncode)


def task():
    return Task("Make the circle red", [prepare_image(png_bytes((40, 90, 200)))],
                prepare_image(png_bytes((220, 30, 30))), prepare_image(png_bytes((30, 160, 60))))


def judge(runner):
    return CodexCliJudge(binary="codex.exe", runner=runner)


def test_routing_and_model_alias():
    assert provider_of("codex") == provider_of("codex:gpt-6-astra") == "codex"
    assert cli_model("codex:gpt-6-astra") == "gpt-6-astra"
    assert cli_model("codex") is None


def test_command_is_ephemeral_isolated_and_schema_constrained(tmp_path):
    cmd = build_command("codex.exe", CFG, tmp_path / "s.json", tmp_path / "o.json",
                        [tmp_path / "a.png"], tmp_path)
    assert cmd[:2] == ["codex.exe", "exec"]
    assert "--ephemeral" in cmd and "--ignore-user-config" in cmd and "--ignore-rules" in cmd
    assert cmd[cmd.index("--sandbox") + 1] == "read-only"
    assert cmd[cmd.index("--model") + 1] == "gpt-6-astra"
    assert cmd[cmd.index("--image") + 1].endswith("a.png")
    assert "--model" not in build_command("codex.exe", CFG.with_overrides(model="codex"),
                                           tmp_path / "s", tmp_path / "o", [], tmp_path)


def test_sends_numbered_images_and_parses_structured_result():
    result = make_judgment("B", "high")
    events = json.dumps({"type": "turn.completed", "usage": {"input_tokens": 120, "output_tokens": 40}})
    run = FakeRun(result=result, stdout=events)
    t = task()
    out = asyncio.run(judge(run).judge_once(t, CFG, swapped=True, notes=["measured fact"]))
    call = run.calls[0]
    assert len(call["images"]) == 3 and all(data.startswith(b"\x89PNG") for _, data in call["images"])
    assert call["input"].index("RESULT A:") < call["input"].index("RESULT B:")
    assert call["images"][1][1] == base64.standard_b64decode(t.b.data_b64)
    assert "[ATTACHED IMAGE 3]" in call["input"]
    assert out["judgment"]["verdict"] == "A"
    assert out["usage"] == {"input_tokens": 120, "output_tokens": 40}


@pytest.mark.parametrize("run,needle,fatal", [
    (FakeRun(stderr="Please login to continue", returncode=1), "login", True),
    (FakeRun(stderr="temporary failure", returncode=1), "exit 1", False),
    (FakeRun(result=None, returncode=0), "no output", False),
    (FakeRun(result=None, raise_exc=subprocess.TimeoutExpired("codex", 600)), "did not answer", False),
    (FakeRun(result=None, raise_exc=FileNotFoundError("missing")), "Could not start", True),
])
def test_failures_become_judge_errors(run, needle, fatal):
    with pytest.raises(JudgeError, match=needle) as exc:
        asyncio.run(judge(run).judge_once(task(), CFG, False, []))
    assert exc.value.fatal is fatal


def test_missing_binary_is_fatal(monkeypatch):
    monkeypatch.delenv("IMAGE_JUDGE_CODEX_BIN", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert find_codex_binary() is None
    with pytest.raises(JudgeError, match="not found") as exc:
        asyncio.run(CodexCliJudge(runner=FakeRun()).judge_once(task(), CFG, False, []))
    assert exc.value.fatal


def test_end_to_end_pipeline():
    run = FakeRun(result=lambda i: make_judgment("B" if i == 0 else "A", "high"))
    result = asyncio.run(evaluate(task(), CFG, judge(run)))
    assert result["aggregate"]["status"] == "confident" and result["aggregate"]["verdict"] == "B"
    assert len(run.calls) == 2
