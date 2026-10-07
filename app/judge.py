"""Shared judge types, the Anthropic judge, and the N-run pipeline around any judge."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Protocol

import anthropic

from .aggregate import RunResult, aggregate
from .config import JudgeConfig, settings
from .images import PreparedImage, near_identical
from .images import difference_map
from .rubric import (JUDGMENT_SCHEMA, USES_DIFFERENCE_MAPS, build_user_content, schema_for,
                     system_prompt)

CACHE_SCHEMA_VERSION = 2


@dataclass
class Task:
    prompt: str
    originals: list[PreparedImage]
    a: PreparedImage
    b: PreparedImage
    # Filled in by evaluate() for rubrics that use them: [("A", map), ("B", map)].
    difference_maps: Optional[list] = None


class JudgeError(Exception):
    def __init__(self, message: str, fatal: bool = False):
        super().__init__(message)
        self.fatal = fatal  # e.g. bad API key: retrying other runs is pointless


class Judge(Protocol):
    async def judge_once(self, task: Task, config: JudgeConfig, swapped: bool, notes: list[str]) -> dict:
        """Return {"judgment": <dict matching JUDGMENT_SCHEMA>, "usage": {...}, "model": str}."""


def measured_notes(task: Task) -> list[str]:
    """Deterministic pixel facts that are cheap to compute and easy for a model to miss."""
    notes = []
    for label, img in (("A", task.a), ("B", task.b)):
        for i, orig in enumerate(task.originals, 1):
            if near_identical(img, orig):
                notes.append(f"Result {label} is visually identical to Original {i} (no visible edit).")
    return notes


def judge_system_prompt(config: JudgeConfig) -> str:
    return system_prompt(config.rubric_version, config.guidelines, config.lessons)


_RESULT_LABEL = re.compile(r"(?<![A-Za-z0-9_])([AB])(?![A-Za-z0-9_])")


def _swap_result_labels(text: str) -> str:
    """Swap standalone A/B references in model evidence without touching words such as RGB."""
    return _RESULT_LABEL.sub(lambda match: "B" if match.group(1) == "A" else "A", text)


def normalize_judgment(judgment: dict, swapped: bool) -> dict:
    """Map display labels back to the task's stable A/B labels after a swapped presentation."""
    if not swapped:
        return judgment
    j = deepcopy(judgment)
    j["verdict"] = "B" if j["verdict"] == "A" else "A"

    impressions = j.get("first_impression")
    if isinstance(impressions, dict) and "a" in impressions and "b" in impressions:
        impressions["a"], impressions["b"] = impressions["b"], impressions["a"]
        impressions["a"] = _swap_result_labels(impressions["a"])
        impressions["b"] = _swap_result_labels(impressions["b"])

    for requirement in j.get("requirements", []):
        if isinstance(requirement, dict) and "a" in requirement and "b" in requirement:
            requirement["a"], requirement["b"] = requirement["b"], requirement["a"]
            for key in ("a", "b"):
                evidence = requirement[key].get("evidence")
                if isinstance(evidence, str):
                    requirement[key]["evidence"] = _swap_result_labels(evidence)

    for field in ("visible_differences", "decisive_difference", "reasoning"):
        value = j.get(field)
        if isinstance(value, str):
            j[field] = _swap_result_labels(value)
        elif isinstance(value, list):
            j[field] = [_swap_result_labels(item) if isinstance(item, str) else item for item in value]
    return j


def judge_content(task: Task, swapped: bool, notes: list[str]) -> list:
    """Provider-neutral judge input using position labels that are normalized after the response.

    A swapped run deliberately presents original B as display A and original A as display B. This
    prevents schema fields named a and b from conflicting with presentation order.
    """
    results = [("A", task.a), ("B", task.b)]
    maps = list(task.difference_maps or [])
    shown_notes = notes
    if swapped:
        results = [("A", task.b), ("B", task.a)]
        maps = [(("B" if label == "A" else "A"), image) for label, image in reversed(maps)]
        shown_notes = [_swap_result_labels(note) for note in notes]
    return build_user_content(task.prompt, task.originals, results, shown_notes, maps)


class JsonJudge:
    """A provider that can answer "here is text + images, reply with JSON matching this schema".

    Judging is built on that one method; so is learning from mistakes (app/learn.py).
    """

    async def complete_json(self, system: str, content: list, schema: dict, config: JudgeConfig) -> dict:
        """Return {"data": <dict>, "model": str, "usage": {"input_tokens", "output_tokens"}}."""
        raise NotImplementedError

    async def judge_once(self, task: Task, config: JudgeConfig, swapped: bool, notes: list[str]) -> dict:
        out = await self.complete_json(judge_system_prompt(config), judge_content(task, swapped, notes),
                                       schema_for(config.rubric_version), config)
        validate_judgment(out["data"])
        judgment = normalize_judgment(out["data"], swapped)
        return {"judgment": judgment, "model": out["model"], "usage": out["usage"]}


def anthropic_blocks(content: list) -> list[dict]:
    return [{"type": "text", "text": c} if isinstance(c, str) else c.content_block() for c in content]


class AnthropicJudge(JsonJudge):
    def __init__(self, client: Optional[anthropic.AsyncAnthropic] = None,
                 custom_gateway: Optional[bool] = None):
        base_url = os.environ.get("ANTHROPIC_BASE_URL") or None
        self.custom_gateway = bool(base_url) if client is None else bool(custom_gateway)
        self.client = client or anthropic.AsyncAnthropic(
            base_url=base_url,
            max_retries=settings.api_max_retries,  # SDK backs off on 429 / 5xx / connection errors
            timeout=settings.api_timeout_s,
        )

    @staticmethod
    def _retry_after(exc) -> Optional[float]:
        """The API's own 'try again in N seconds' hint, if it sent one."""
        try:
            value = float(exc.response.headers.get("retry-after", ""))
        except (AttributeError, TypeError, ValueError):
            return None
        return min(max(value, 1.0), 120.0)

    async def _stream(self, messages_api, kwargs):
        """One request. If rate limited even after the SDK's quick retries, wait and try again, so a short
        busy spell does not fail the whole judgment."""
        waits = settings.rate_limit_waits_s
        for attempt in range(len(waits) + 1):
            try:
                async with messages_api.stream(**kwargs) as stream:
                    return await stream.get_final_message()
            except anthropic.RateLimitError as exc:
                if attempt == len(waits):
                    raise
                await asyncio.sleep(self._retry_after(exc) or waits[attempt])

    async def complete_json(self, system: str, content: list, schema: dict, config: JudgeConfig) -> dict:
        kwargs = dict(
            model=config.model,
            max_tokens=config.max_tokens,
            system=system,
            messages=[{"role": "user", "content": anthropic_blocks(content)}],
        )
        if self.custom_gateway:
            # Anthropic-compatible gateways commonly support tool calling more reliably than
            # Anthropic's newer output_config JSON-schema extension. A forced, schema-constrained
            # tool call produces the same JSON object without relying on a text block.
            kwargs.update(
                tools=[{
                    "name": "submit_json",
                    "description": "Return the requested structured result.",
                    "input_schema": schema,
                }],
                tool_choice={"type": "tool", "name": "submit_json"},
            )
            messages_api = self.client.messages
        else:
            kwargs["output_config"] = {
                "effort": config.effort,
                "format": {"type": "json_schema", "schema": schema},
            }
            if config.use_fallbacks:
                kwargs.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
            messages_api = self.client.beta.messages

        try:
            message = await self._stream(messages_api, kwargs)
        except anthropic.AuthenticationError as exc:
            raise JudgeError("Anthropic API key missing or invalid (set ANTHROPIC_API_KEY).", fatal=True) from exc
        except anthropic.PermissionDeniedError as exc:
            raise JudgeError(f"API key not permitted to use {config.model}: {exc.message}", fatal=True) from exc
        except anthropic.NotFoundError as exc:
            raise JudgeError(f"Model {config.model!r} not found; set IMAGE_JUDGE_MODEL.", fatal=True) from exc
        except anthropic.BadRequestError as exc:
            raise JudgeError(f"Request rejected: {exc.message}", fatal=True) from exc
        except anthropic.RateLimitError as exc:
            raise JudgeError("Rate limited after retries; lower IMAGE_JUDGE_MAX_CONCURRENCY or try later.") from exc
        except anthropic.APIStatusError as exc:
            raise JudgeError(f"API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise JudgeError(f"Could not reach the Anthropic API: {exc}") from exc
        except TypeError as exc:
            if "authentication" in str(exc):  # raised by the SDK when no credentials are configured
                raise JudgeError("No Anthropic credentials: set ANTHROPIC_API_KEY.", fatal=True) from exc
            raise

        if message.stop_reason == "refusal":
            raise JudgeError("Model declined to judge these images.")
        if message.stop_reason == "max_tokens":
            raise JudgeError("Judge output was truncated (raise IMAGE_JUDGE_MAX_TOKENS).")

        data = next(
            (block.input for block in message.content
             if block.type == "tool_use" and getattr(block, "name", None) == "submit_json"),
            None,
        )
        if data is None:
            raw = "".join(block.text for block in message.content if block.type == "text").strip()
            # Some compatible gateways wrap otherwise valid JSON in Markdown fences.
            fence = chr(96) * 3
            if raw.startswith(fence) and raw.endswith(fence):
                raw = raw.split("\n", 1)[-1].rsplit(fence, 1)[0].strip()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                block_types = ", ".join(block.type for block in message.content) or "none"
                raise JudgeError(
                    f"Judge returned invalid JSON or no usable structured output (content blocks: {block_types}; "
                    f"stop reason: {message.stop_reason})."
                ) from exc
        usage = message.usage
        return {
            "data": data,
            "model": message.model,
            "usage": {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens},
        }

def validate_judgment(j: dict) -> None:
    if j.get("verdict") not in ("A", "B"):
        raise JudgeError(f"Judge returned no valid verdict: {j.get('verdict')!r}")
    if j.get("confidence") not in ("high", "medium", "low"):
        raise JudgeError(f"Judge returned no valid confidence: {j.get('confidence')!r}")
    if not isinstance(j.get("requirements"), list):
        raise JudgeError("Judge returned no requirements list")


def cache_key(task: Task, config: JudgeConfig, run_index: int) -> str:
    payload = {
        "v": CACHE_SCHEMA_VERSION,
        "model": config.model,
        "rubric": config.rubric_version,
        "effort": config.effort,
        "prompt": task.prompt.strip(),
        "originals": [o.sha256 for o in task.originals],
        "a": task.a.sha256,
        "b": task.b.sha256,
        "run": run_index,  # run i is the i-th independent sample; i odd => swapped
    }
    if config.guidelines.strip() or config.lessons:
        # Different guidelines/lessons => a different judge. (Only added when present, so
        # judgments cached before knowledge existed stay valid.)
        payload["knowledge"] = hashlib.sha256(
            json.dumps([config.guidelines.strip(), list(config.lessons)]).encode()).hexdigest()
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


_semaphores: dict[int, asyncio.Semaphore] = {}


def api_semaphore() -> asyncio.Semaphore:
    """Per-event-loop cap on concurrent API calls (shared by the UI and benchmarks)."""
    loop_id = id(asyncio.get_running_loop())
    if loop_id not in _semaphores:
        _semaphores[loop_id] = asyncio.Semaphore(settings.max_concurrency)
    return _semaphores[loop_id]


def _run_from(i: int, swapped: bool, out: dict, cached: bool = False) -> RunResult:
    j = out["judgment"]
    return RunResult(i, swapped, j["verdict"], j["confidence"], j, cached=cached,
                     usage=out.get("usage"), model=out.get("model"))


async def evaluate(task: Task, config: JudgeConfig, judge: Judge, db=None, use_cache: bool = True) -> dict:
    """Run `config.runs` independent judgments (half with A/B order swapped) and aggregate."""
    if near_identical(task.a, task.b):
        runs: list[RunResult] = []
        agg = aggregate(runs, planned=config.runs)
        agg.status, agg.explanation = "unclear", "Result A and Result B are visually identical; nothing to judge."
        return {"aggregate": agg.to_dict(), "runs": [], "notes": ["A and B are identical"], "config": config.as_dict()}

    notes = measured_notes(task)
    if config.rubric_version in USES_DIFFERENCE_MAPS and task.originals and task.difference_maps is None:
        maps = []
        for label, img in (("A", task.a), ("B", task.b)):
            heat, changed = await asyncio.to_thread(difference_map, img, task.originals[0])
            if heat is not None:
                maps.append((label, heat))
                notes.append(f"Compared with Original 1, Result {label} differs over about {changed:.0%} of "
                             "the image area (see its difference map).")
        task.difference_maps = maps
    fatal = asyncio.Event()

    async def one(i: int) -> RunResult:
        swapped = i % 2 == 1
        key = cache_key(task, config, i)
        if use_cache and db is not None and (hit := db.cache_get(key)):
            return _run_from(i, swapped, hit, cached=True)
        if fatal.is_set():
            return RunResult(i, swapped, error="skipped after a fatal error in another run")
        try:
            async with api_semaphore():
                if fatal.is_set():
                    return RunResult(i, swapped, error="skipped after a fatal error in another run")
                deadline = asyncio.get_running_loop().time() + settings.api_timeout_s
                for attempt in range(2):
                    try:
                        remaining = deadline - asyncio.get_running_loop().time()
                        if remaining <= 0:
                            raise TimeoutError
                        out = await asyncio.wait_for(
                            judge.judge_once(task, config, swapped, notes),
                            timeout=remaining,
                        )
                        break
                    except JudgeError as exc:
                        # Gateways occasionally return malformed structured output. Retry one
                        # non-fatal response, but keep both attempts inside the same hard deadline.
                        if exc.fatal or attempt == 1:
                            raise
        except TimeoutError:
            return RunResult(
                i, swapped,
                error=f"Judge timed out after {settings.api_timeout_s:.0f}s (evaluation deadline).",
            )
        except JudgeError as exc:
            if exc.fatal:
                fatal.set()
            return RunResult(i, swapped, error=str(exc))
        if db is not None:
            db.cache_put(key, out)
        return _run_from(i, swapped, out)

    runs = list(await asyncio.gather(*(one(i) for i in range(config.runs))))
    agg = aggregate(runs, planned=config.runs)
    result = {
        "aggregate": agg.to_dict(),
        "runs": [r.to_dict() for r in runs],
        "notes": notes,
        "config": config.as_dict(),
    }
    if fatal.is_set():
        # A configuration problem (bad key, unknown model): callers should stop, not retry.
        result["fatal"] = next((r.error for r in runs if r.error and not r.error.startswith("skipped")), "fatal error")
    return result
