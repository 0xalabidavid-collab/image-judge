"""ClaudeCodeJudge command building and output parsing, against a fake subprocess runner."""

import asyncio
import json
import subprocess
from types import SimpleNamespace

import pytest

from app.claude_code_judge import ClaudeCodeJudge, build_command, cli_model, find_claude_binary
from app.config import JudgeConfig, provider_of
from app.images import prepare_image
from app.judge import JudgeError, Task, evaluate
from app.rubric import JUDGMENT_SCHEMA
from conftest import make_judgment, png_bytes

CFG = JudgeConfig(model="claude-code:fable", runs=2, rubric_version="v1", effort="high")


def stream(*events):
    return "\n".join(json.dumps(e) for e in events) + "\n"


def ok_output(verdict="B"):
    j = make_judgment(verdict, "high")
    return stream(
        {"type": "system", "subtype": "init", "model": "claude-fable-5-1"},
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "StructuredOutput", "input": j}]}},
        {"type": "result", "subtype": "success", "is_error": False, "api_error_status": None,
         "result": json.dumps(j), "structured_output": j,
         "usage": {"input_tokens": 50, "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 0,
                   "output_tokens": 400}},
    )


class FakeRun:
    def __init__(self, stdout="", stderr="", returncode=0, raise_exc=None):
        self.calls = []
        self.stdout, self.stderr, self.returncode, self.raise_exc = stdout, stderr, returncode, raise_exc

    def __call__(self, cmd, **kwargs):
        self.calls.append({"cmd": cmd, **kwargs})
        if self.raise_exc:
            raise self.raise_exc
        stdout = self.stdout(len(self.calls) - 1) if callable(self.stdout) else self.stdout
        return SimpleNamespace(stdout=stdout, stderr=self.stderr, returncode=self.returncode)


def task():
    return Task("Make the circle red", [prepare_image(png_bytes((40, 90, 200)))],
                prepare_image(png_bytes((220, 30, 30))), prepare_image(png_bytes((30, 160, 60))))


def judge(runner):
    return ClaudeCodeJudge(binary="claude.exe", runner=runner)


def test_routing_and_model_alias():
    assert provider_of("claude-code") == provider_of("claude-code:opus") == "claude-code"
    assert provider_of("claude-opus-5-5") == "anthropic"
    assert cli_model("claude-code:fable") == "fable"
    assert cli_model("claude-code") is None


def test_command_shape():
    from app.judge import judge_system_prompt
    cmd = build_command("claude.exe", CFG, judge_system_prompt(CFG), JUDGMENT_SCHEMA)
    arg = lambda flag: cmd[cmd.index(flag) + 1]
    assert cmd[:2] == ["claude.exe", "-p"]
    assert arg("--input-format") == "stream-json" and arg("--output-format") == "stream-json"
    assert arg("--tools") == ""
    assert json.loads(arg("--json-schema")) == JUDGMENT_SCHEMA
    assert "RESULT A" in arg("--system-prompt")
    assert arg("--model") == "fable" and arg("--effort") == "high"
    assert "--no-session-persistence" in cmd and "--safe-mode" in cmd
    assert "--model" not in build_command("claude.exe", CFG.with_overrides(model="claude-code"), "s", {})


def test_sends_images_on_stdin_and_parses_result():
    run = FakeRun(stdout=ok_output("B"))
    out = asyncio.run(judge(run).judge_once(task(), CFG, swapped=True, notes=["n"]))
    msg = json.loads(run.calls[0]["input"])
    content = msg["message"]["content"]
    texts = [c.get("text") for c in content]
    assert msg["type"] == "user" and texts.index("RESULT A:") < texts.index("RESULT B:")
    assert sum(c["type"] == "image" for c in content) == 3
    assert out["judgment"]["verdict"] == "A"
    assert out["model"] == "claude-fable-5-1"
    assert out["usage"] == {"input_tokens": 1050, "output_tokens": 400}


def test_falls_back_to_result_text_without_structured_output():
    j = make_judgment("A")
    run = FakeRun(stdout=stream({"type": "result", "subtype": "success", "is_error": False, "result": json.dumps(j)}))
    assert asyncio.run(judge(run).judge_once(task(), CFG, False, []))["judgment"]["verdict"] == "A"


@pytest.mark.parametrize("run,needle,fatal", [
    (FakeRun(stdout="", stderr="Invalid API key · Please run /login", returncode=1), "login", True),
    (FakeRun(stdout="", stderr="boom", returncode=1), "exit 1", False),
    (FakeRun(stdout=stream({"type": "result", "subtype": "success", "is_error": True, "api_error_status": 429,
                            "result": "Claude usage limit reached"})), "usage limit", True),
    (FakeRun(stdout=stream({"type": "result", "subtype": "error_during_execution", "is_error": True,
                            "api_error_status": 529, "result": "Overloaded"})), "error", False),
    (FakeRun(stdout=stream({"type": "result", "subtype": "success", "is_error": False, "result": "{nope"})),
     "invalid JSON", False),
    (FakeRun(raise_exc=subprocess.TimeoutExpired("claude", 600)), "did not answer", False),
    (FakeRun(raise_exc=FileNotFoundError("missing")), "Could not start", True),
])
def test_failures_become_judge_errors(run, needle, fatal):
    with pytest.raises(JudgeError, match=needle) as exc:
        asyncio.run(judge(run).judge_once(task(), CFG, False, []))
    assert exc.value.fatal is fatal


def test_missing_binary_is_fatal(monkeypatch):
    monkeypatch.delenv("IMAGE_JUDGE_CLAUDE_BIN", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert find_claude_binary() is None
    with pytest.raises(JudgeError, match="not found") as exc:
        asyncio.run(ClaudeCodeJudge(runner=FakeRun()).judge_once(task(), CFG, False, []))
    assert exc.value.fatal


def test_end_to_end_pipeline():
    run = FakeRun(stdout=lambda i: ok_output("B" if i == 0 else "A"))
    result = asyncio.run(evaluate(task(), CFG, judge(run)))
    assert result["aggregate"]["status"] == "confident" and result["aggregate"]["verdict"] == "B"
    assert len(run.calls) == 2
