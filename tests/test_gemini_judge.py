"""GeminiJudge request building and response handling, against a fake google-genai client."""

import asyncio
import base64
import json
from types import SimpleNamespace

import pytest
from google.genai import errors, types

from app import config as config_mod
from app.config import JudgeConfig, provider_of
from app.gemini_judge import GeminiJudge
from app.images import prepare_image
from app.judge import JudgeError, Task, evaluate
from app.providers import RoutingJudge
from app.rubric import JUDGMENT_SCHEMA
from conftest import make_judgment, png_bytes

CFG = JudgeConfig(model="gemini-3.1-pro-preview", runs=2, rubric_version="v1", effort="high")


class FakeGemini:
    def __init__(self, text=None, finish="STOP", block=None, raise_exc=None, candidates=True):
        self.requests = []
        self.text = text if text is not None else json.dumps(make_judgment("A", "high"))
        self.finish, self.block, self.raise_exc, self.candidates = finish, block, raise_exc, candidates
        self.aio = SimpleNamespace(models=SimpleNamespace(generate_content=self._generate))

    async def _generate(self, model, contents, config):
        self.requests.append({"model": model, "contents": contents, "config": config})
        if self.raise_exc:
            raise self.raise_exc
        return SimpleNamespace(
            text=(self.text(len(self.requests) - 1) if callable(self.text) else self.text),
            prompt_feedback=SimpleNamespace(block_reason=self.block) if self.block else None,
            candidates=[SimpleNamespace(finish_reason=types.FinishReason(self.finish))] if self.candidates else [],
            usage_metadata=SimpleNamespace(prompt_token_count=900, candidates_token_count=300, thoughts_token_count=700),
        )


def task():
    return Task("Make the circle red", [prepare_image(png_bytes((40, 90, 200)))],
                prepare_image(png_bytes((220, 30, 30))), prepare_image(png_bytes((30, 160, 60))))


def run(client, cfg=CFG, swapped=False):
    return asyncio.run(GeminiJudge(client).judge_once(task(), cfg, swapped, []))


def test_request_shape():
    client = FakeGemini()
    out = run(client)
    req = client.requests[0]
    cfg = req["config"]
    assert req["model"] == "gemini-3.1-pro-preview"
    assert cfg.response_mime_type == "application/json"
    assert cfg.response_json_schema == JUDGMENT_SCHEMA
    assert cfg.thinking_config.thinking_level == types.ThinkingLevel.HIGH
    assert cfg.media_resolution == types.MediaResolution.MEDIA_RESOLUTION_HIGH
    assert "RESULT A" in cfg.system_instruction
    texts = [p for p in req["contents"] if isinstance(p, str)]
    assert texts[0].startswith("PROMPT:")
    assert [t for t in texts if t.startswith(("ORIGINAL", "RESULT"))] == ["ORIGINAL 1:", "RESULT A:", "RESULT B:"]
    images = [p for p in req["contents"] if isinstance(p, types.Part)]
    assert len(images) == 3 and images[0].inline_data.mime_type == "image/png"
    assert out["judgment"]["verdict"] == "A"
    assert out["usage"] == {"input_tokens": 900, "output_tokens": 1000}  # thinking counted as output


def test_swapped_run_presents_original_b_as_display_a_and_normalizes():
    client = FakeGemini()
    t = task()
    out = asyncio.run(GeminiJudge(client).judge_once(t, CFG, True, []))
    contents = client.requests[0]["contents"]
    i_a, i_b = contents.index("RESULT A:"), contents.index("RESULT B:")
    assert i_a < i_b
    assert contents[i_a + 1].inline_data.data == base64.standard_b64decode(t.b.data_b64)
    assert out["judgment"]["verdict"] == "B"


def test_effort_maps_to_thinking_level_and_25_uses_budget():
    client = FakeGemini()
    run(client, CFG.with_overrides(effort="low"))
    assert client.requests[0]["config"].thinking_config.thinking_level == types.ThinkingLevel.LOW
    run(client, CFG.with_overrides(model="gemini-2.5-pro"))
    tc = client.requests[1]["config"].thinking_config
    assert tc.thinking_budget == -1 and tc.thinking_level is None


@pytest.mark.parametrize("kwargs,needle,fatal", [
    ({"finish": "MAX_TOKENS"}, "truncated", False),
    ({"finish": "SAFETY"}, "declined", False),
    ({"block": "PROHIBITED_CONTENT"}, "blocked", False),
    ({"candidates": False}, "no answer", False),
    ({"text": "{nope"}, "invalid JSON", False),
    ({"raise_exc": errors.ClientError(403, {"error": {"message": "bad key", "status": "PERMISSION_DENIED"}})}, "key", True),
    ({"raise_exc": errors.ClientError(404, {"error": {"message": "nope", "status": "NOT_FOUND"}})}, "not found", True),
    ({"raise_exc": errors.ClientError(429, {"error": {"message": "quota", "status": "RESOURCE_EXHAUSTED"}})}, "rate limit", False),
    ({"raise_exc": errors.ClientError(429, {"error": {"message": "Quota exceeded ... limit: 0, model: gemini-3.1-pro", "status": "RESOURCE_EXHAUSTED"}})}, "no quota", True),
    ({"raise_exc": errors.ClientError(429, {"error": {"message": "quota", "status": "RESOURCE_EXHAUSTED", "details": [
        {"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}})}, "Daily", True),
    ({"raise_exc": errors.ServerError(503, {"error": {"message": "busy", "status": "UNAVAILABLE"}})}, "server error", False),
])
def test_failures_become_judge_errors(kwargs, needle, fatal):
    with pytest.raises(JudgeError, match=needle) as exc:
        run(FakeGemini(**kwargs))
    assert exc.value.fatal is fatal


def test_missing_key_is_fatal(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(JudgeError, match="GEMINI_API_KEY") as exc:
        asyncio.run(GeminiJudge().judge_once(task(), CFG, False, []))
    assert exc.value.fatal


def test_end_to_end_pipeline_with_fake_gemini():
    client = FakeGemini(text=lambda i: json.dumps(make_judgment("A" if i == 0 else "B", "high")))
    result = asyncio.run(evaluate(task(), CFG, GeminiJudge(client)))
    assert result["aggregate"]["status"] == "confident" and result["aggregate"]["verdict"] == "A"
    assert len(client.requests) == 2


def test_routing_by_model_name():
    assert provider_of("gemini-3.8-flash") == "gemini"
    assert provider_of("claude-opus-5-5") == "anthropic"
    router = RoutingJudge()
    fake = FakeGemini()
    router._judges["gemini"] = GeminiJudge(fake)
    asyncio.run(router.judge_once(task(), CFG, False, []))
    assert len(fake.requests) == 1


def test_default_model_follows_available_key(monkeypatch):
    monkeypatch.delenv("IMAGE_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    assert config_mod._default_model() == "gemini-3.1-pro-preview"
    monkeypatch.setenv("IMAGE_JUDGE_MODEL", "gemini-3.8-flash")
    assert config_mod._default_model() == "gemini-3.8-flash"
