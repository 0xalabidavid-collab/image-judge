"""AnthropicJudge request building and response handling, against a fake SDK client."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from app import benchmark as bm
from app.config import JudgeConfig
from app.images import prepare_image
from app.judge import AnthropicJudge, JudgeError, Task, evaluate
from app.rubric import JUDGMENT_SCHEMA
from conftest import make_judgment, png_bytes
from test_benchmark import row, write_csv_dataset

CFG = JudgeConfig(model="claude-fable-5-1", runs=2, rubric_version="v1", effort="high", use_fallbacks=True)


class FakeStream:
    def __init__(self, message):
        self.message = message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get_final_message(self):
        return self.message


class FakeClient:
    def __init__(self, text=None, stop_reason="end_turn", tool_input=None):
        self.requests = []
        self.text = text if text is not None else json.dumps(make_judgment("B", "high"))
        self.stop_reason = stop_reason
        self.tool_input = tool_input
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))
        self.messages = SimpleNamespace(stream=self._stream)

    def _stream(self, **kwargs):
        self.requests.append(kwargs)
        content = (
            [SimpleNamespace(type="tool_use", name="submit_json", input=self.tool_input)]
            if self.tool_input is not None else
            [SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=(self.text(len(self.requests) - 1) if callable(self.text) else self.text))]
        )
        msg = SimpleNamespace(
            stop_reason=self.stop_reason,
            content=content,
            model=kwargs["model"],
            usage=SimpleNamespace(input_tokens=1234, output_tokens=567),
        )
        return FakeStream(msg)


def task():
    return Task("Make the circle red", [prepare_image(png_bytes((40, 90, 200)))],
                prepare_image(png_bytes((220, 30, 30))), prepare_image(png_bytes((30, 160, 60))))


def labels_in_order(request):
    content = request["messages"][0]["content"]
    return [c["text"] for c in content if c["type"] == "text" and c["text"].startswith(("RESULT", "ORIGINAL"))]


def test_request_shape():
    client = FakeClient()
    out = asyncio.run(AnthropicJudge(client).judge_once(task(), CFG, swapped=False, notes=["n1"]))
    req = client.requests[0]
    assert req["model"] == "claude-fable-5-1"
    assert req["output_config"]["effort"] == "high"
    assert req["output_config"]["format"] == {"type": "json_schema", "schema": JUDGMENT_SCHEMA}
    assert req["fallbacks"] == "default" and req["betas"] == ["server-side-fallback-2026-07-01"]
    assert "thinking" not in req  # always on for this model; explicit config can 400
    assert "temperature" not in req
    assert labels_in_order(req) == ["ORIGINAL 1:", "RESULT A:", "RESULT B:"]
    assert out["judgment"]["verdict"] == "B"
    assert out["usage"] == {"input_tokens": 1234, "output_tokens": 567}


def test_swapped_run_presents_original_b_as_display_a_and_normalizes():
    client = FakeClient()
    t = task()
    out = asyncio.run(AnthropicJudge(client).judge_once(t, CFG, swapped=True, notes=[]))
    content = client.requests[0]["messages"][0]["content"]
    texts = [c.get("text") for c in content]
    i_a = texts.index("RESULT A:")
    assert i_a < texts.index("RESULT B:")
    assert content[i_a + 1]["source"]["data"] == t.b.data_b64
    assert out["judgment"]["verdict"] == "A"
    req = out["judgment"]["requirements"][0]
    assert req["a"]["status"] == "pass" and req["a"]["evidence"].startswith("A:")
    assert req["b"]["status"] == "fail" and req["b"]["evidence"].startswith("B:")


def test_custom_gateway_forces_schema_tool_and_reads_tool_input():
    judgment = make_judgment("A", "high")
    client = FakeClient(stop_reason="tool_use", tool_input=judgment)
    out = asyncio.run(AnthropicJudge(client, custom_gateway=True).judge_once(task(), CFG, False, []))
    req = client.requests[0]
    assert "output_config" not in req and "betas" not in req and "fallbacks" not in req
    assert req["tool_choice"] == {"type": "tool", "name": "submit_json"}
    assert req["tools"][0]["input_schema"] == JUDGMENT_SCHEMA
    assert out["judgment"]["verdict"] == "A"


def test_fallbacks_can_be_disabled():
    client = FakeClient()
    asyncio.run(AnthropicJudge(client).judge_once(task(), CFG.with_overrides(use_fallbacks=False), False, []))
    assert "fallbacks" not in client.requests[0] and "betas" not in client.requests[0]


@pytest.mark.parametrize("stop_reason,needle", [("refusal", "declined"), ("max_tokens", "truncated")])
def test_bad_stop_reasons_raise(stop_reason, needle):
    with pytest.raises(JudgeError, match=needle):
        asyncio.run(AnthropicJudge(FakeClient(stop_reason=stop_reason)).judge_once(task(), CFG, False, []))


def test_invalid_json_raises():
    with pytest.raises(JudgeError, match="invalid JSON"):
        asyncio.run(AnthropicJudge(FakeClient(text="{nope")).judge_once(task(), CFG, False, []))


def test_end_to_end_with_fake_client():
    client = FakeClient(text=lambda i: json.dumps(make_judgment("B" if i == 0 else "A", "high")))
    result = asyncio.run(evaluate(task(), CFG, AnthropicJudge(client)))
    assert result["aggregate"]["status"] == "confident" and result["aggregate"]["verdict"] == "B"
    assert len(client.requests) == 2


def test_benchmark_aborts_on_fatal_error(tmp_path, db, monkeypatch):
    from conftest import FakeJudge
    monkeypatch.setattr(bm.settings, "results_dir", tmp_path / "results", raising=False)
    write_csv_dataset(tmp_path / "ds", [row(i, "A") for i in range(6)])
    judge = FakeJudge(error=JudgeError("No Anthropic credentials", fatal=True))
    with pytest.raises(bm.BenchmarkAborted, match="credentials"):
        asyncio.run(bm.run_benchmark(tmp_path / "ds", CFG, judge, db))
    assert db.list_benchmarks()[0]["status"] == "failed"


# --- rate limits: wait and try again instead of failing the judgment --------------------------
import anthropic
import httpx


def rate_limit_error(retry_after=None):
    headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
    response = httpx.Response(429, request=httpx.Request("POST", "https://api.example/v1/messages"), headers=headers)
    return anthropic.RateLimitError("slow down", response=response, body=None)


class RateLimitedClient(FakeClient):
    """Says 'too many requests' for the first `limited` calls, then answers normally."""

    def __init__(self, limited, retry_after=None, **kw):
        super().__init__(**kw)
        self.limited, self.retry_after, self.calls = limited, retry_after, 0

    def _stream(self, **kwargs):
        self.calls += 1
        if self.calls <= self.limited:
            err = rate_limit_error(self.retry_after)

            class Refused(FakeStream):
                async def __aenter__(self):
                    raise err
            return Refused(None)
        return super()._stream(**kwargs)


@pytest.fixture
def sleeps(monkeypatch):
    waited = []

    async def fake_sleep(seconds):
        waited.append(seconds)
    monkeypatch.setattr("app.judge.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("app.judge.settings.rate_limit_waits_s", (10.0, 20.0, 40.0), raising=False)
    return waited


def test_a_busy_spell_is_waited_out(sleeps):
    client = RateLimitedClient(limited=2)
    out = asyncio.run(AnthropicJudge(client).judge_once(task(), CFG, swapped=False, notes=[]))
    assert out["judgment"]["verdict"] == "B" and client.calls == 3
    assert sleeps == [10.0, 20.0]  # the configured waits, in order


def test_the_apis_own_retry_after_hint_wins(sleeps):
    asyncio.run(AnthropicJudge(RateLimitedClient(limited=1, retry_after=7)).judge_once(task(), CFG, False, []))
    assert sleeps == [7.0]


def test_it_gives_up_after_the_last_wait(sleeps):
    client = RateLimitedClient(limited=99)
    with pytest.raises(JudgeError, match="Rate limited"):
        asyncio.run(AnthropicJudge(client).judge_once(task(), CFG, False, []))
    assert client.calls == 4 and sleeps == [10.0, 20.0, 40.0]  # 1 try + 3 waits
