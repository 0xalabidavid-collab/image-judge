"""Judge calls against the Google Gemini API (google-genai SDK)."""

from __future__ import annotations

import base64
import json
from typing import Optional

import httpx
from google import genai
from google.genai import errors, types

from .config import JudgeConfig, has_key, settings
from .judge import JsonJudge, JudgeError

# The rubric's effort levels map onto Gemini 3's thinking levels.
THINKING_LEVEL = {
    "low": types.ThinkingLevel.LOW,
    "medium": types.ThinkingLevel.MEDIUM,
    "high": types.ThinkingLevel.HIGH,
    "xhigh": types.ThinkingLevel.HIGH,
    "max": types.ThinkingLevel.HIGH,
}
BLOCKED_FINISH = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "IMAGE_SAFETY", "RECITATION"}


def _quota_ids(exc: errors.APIError) -> list[str]:
    """Names of the quotas a 429 violated, e.g. GenerateRequestsPerDayPerProjectPerModel-FreeTier."""
    details = exc.details.get("error", {}).get("details", []) if isinstance(exc.details, dict) else []
    return [v.get("quotaId", "") for d in details for v in d.get("violations", [])]


def _image_part(img) -> types.Part:
    return types.Part.from_bytes(data=base64.standard_b64decode(img.data_b64), mime_type=img.media_type)


def _thinking_config(config: JudgeConfig) -> Optional[types.ThinkingConfig]:
    if config.model.startswith("gemini-2.5"):
        return types.ThinkingConfig(thinking_budget=-1)  # 2.5 models: dynamic thinking budget
    return types.ThinkingConfig(thinking_level=THINKING_LEVEL.get(config.effort, types.ThinkingLevel.HIGH))


class GeminiJudge(JsonJudge):
    def __init__(self, client: Optional[genai.Client] = None):
        self._client = client

    @property
    def client(self) -> genai.Client:
        # Created on first use so a missing key surfaces as a judge error, not a crash at startup.
        if self._client is None:
            if not has_key("gemini"):  # check first: a half-built SDK client logs a traceback on cleanup
                raise JudgeError("No Gemini credentials: set GEMINI_API_KEY.", fatal=True)
            try:
                # Reads GEMINI_API_KEY (or GOOGLE_API_KEY) from the environment.
                self._client = genai.Client(http_options=types.HttpOptions(
                    timeout=int(settings.api_timeout_s * 1000),  # milliseconds
                    # Exponential backoff on 408 / 429 / 5xx.
                    retry_options=types.HttpRetryOptions(attempts=settings.api_max_retries + 1),
                ))
            except ValueError as exc:
                raise JudgeError("No Gemini credentials: set GEMINI_API_KEY.", fatal=True) from exc
        return self._client

    async def complete_json(self, system: str, content: list, schema: dict, config: JudgeConfig) -> dict:
        parts = [c if isinstance(c, str) else _image_part(c) for c in content]
        gen_config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=schema,
            max_output_tokens=config.max_tokens,
            thinking_config=_thinking_config(config),
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,  # small details decide verdicts
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),  # no tools used
        )
        try:
            response = await self.client.aio.models.generate_content(
                model=config.model, contents=parts, config=gen_config)
        except errors.ClientError as exc:
            if exc.code in (401, 403):
                raise JudgeError("Gemini API key missing or invalid (set GEMINI_API_KEY).", fatal=True) from exc
            if exc.code == 404:
                raise JudgeError(f"Gemini model {config.model!r} not found; set IMAGE_JUDGE_MODEL.", fatal=True) from exc
            if exc.code == 429:
                if "limit: 0" in (exc.message or ""):  # model not included in this plan: retrying can't help
                    raise JudgeError(f"Your Gemini plan has no quota for {config.model} (e.g. Pro on the free "
                                     "tier). Use another model such as gemini-3.8-flash, or enable billing.",
                                     fatal=True) from exc
                if any("PerDay" in q for q in _quota_ids(exc)):  # resets daily: stop, don't burn the run
                    raise JudgeError(f"Daily Gemini quota for {config.model} is used up "
                                     "(free tier: about 20 requests/day per model). Try tomorrow, use "
                                     "another model, or enable billing.", fatal=True) from exc
                raise JudgeError("Gemini rate limit / quota hit after retries; lower "
                                 f"IMAGE_JUDGE_MAX_CONCURRENCY or check your quota. ({(exc.message or '')[:200]})") from exc
            if exc.code == 400:
                raise JudgeError(f"Gemini rejected the request: {exc.message}", fatal=True) from exc
            raise JudgeError(f"Gemini API error {exc.code}: {exc.message}") from exc
        except errors.ServerError as exc:
            raise JudgeError(f"Gemini server error {exc.code}: {exc.message}") from exc
        except httpx.HTTPError as exc:
            raise JudgeError(f"Could not reach the Gemini API: {exc}") from exc

        feedback = response.prompt_feedback
        if feedback is not None and feedback.block_reason:
            raise JudgeError(f"Gemini blocked the request ({feedback.block_reason}).")
        if not response.candidates:
            raise JudgeError("Gemini returned no answer.")
        finish = str(getattr(response.candidates[0].finish_reason, "value", response.candidates[0].finish_reason))
        if finish == "MAX_TOKENS":
            raise JudgeError("Judge output was truncated (raise IMAGE_JUDGE_MAX_TOKENS).")
        if finish in BLOCKED_FINISH:
            raise JudgeError(f"Gemini declined to judge these images ({finish}).")
        try:
            data = json.loads(response.text or "")
        except json.JSONDecodeError as exc:
            raise JudgeError(f"Judge returned invalid JSON: {exc}") from exc

        usage = response.usage_metadata
        return {
            "data": data,
            "model": config.model,
            "usage": {
                "input_tokens": (usage.prompt_token_count or 0) if usage else 0,
                # Thinking tokens are billed as output.
                "output_tokens": ((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)) if usage else 0,
            },
        }
