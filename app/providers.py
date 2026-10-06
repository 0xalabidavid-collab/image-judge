"""Pick the judge implementation from the model name, so configs and sweeps can mix providers."""

from __future__ import annotations

from .config import JudgeConfig, provider_of
from .judge import Task


class RoutingJudge:
    def __init__(self):
        self._judges: dict[str, object] = {}

    def _get(self, provider: str):
        if provider not in self._judges:
            if provider == "codex":
                from .codex_cli_judge import CodexCliJudge
                self._judges[provider] = CodexCliJudge()
            elif provider == "claude-code":
                from .claude_code_judge import ClaudeCodeJudge
                self._judges[provider] = ClaudeCodeJudge()
            elif provider == "gemini":
                from .gemini_judge import GeminiJudge
                self._judges[provider] = GeminiJudge()
            else:
                from .judge import AnthropicJudge
                self._judges[provider] = AnthropicJudge()
        return self._judges[provider]

    async def judge_once(self, task: Task, config: JudgeConfig, swapped: bool, notes: list[str]) -> dict:
        return await self._get(provider_of(config.model)).judge_once(task, config, swapped, notes)

    async def complete_json(self, system: str, content: list, schema: dict, config: JudgeConfig) -> dict:
        return await self._get(provider_of(config.model)).complete_json(system, content, schema, config)
