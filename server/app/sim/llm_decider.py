from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.llm.client import LLMClient


class MoveTo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float = Field(ge=-20.0, le=20.0)
    z: float = Field(ge=-20.0, le=20.0)


class RelationDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    to_id: str = Field(min_length=1, max_length=64)
    delta: int = Field(ge=-5, le=5)


class DecisionDeltas(BaseModel):
    model_config = ConfigDict(extra="forbid")

    self_mood: int = Field(default=0, ge=-10, le=10)
    relations: list[RelationDelta] = Field(default_factory=list, max_length=8)


class AgentDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    agent_id: str = Field(min_length=1, max_length=64)
    goal: str = Field(min_length=1, max_length=120)
    act: Literal["move", "say", "message", "idle"] = Field(validation_alias=AliasChoices("act", "action"))
    target_id: str | None = Field(default=None, max_length=64)
    say_text: str | None = Field(default=None, max_length=280)
    move_to: MoveTo | None = None
    deltas: DecisionDeltas | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> "AgentDecision":
        if self.act in {"say", "message"} and not (self.say_text and self.say_text.strip()):
            raise ValueError("say_text is required for say/message actions")
        if self.act == "message" and not self.target_id:
            raise ValueError("target_id is required for message actions")
        if self.act != "move" and self.move_to is not None:
            raise ValueError("move_to is allowed only for move action")
        if self.act not in {"say", "message"} and self.say_text is not None:
            raise ValueError("say_text is allowed only for say/message actions")
        return self


class TickDecisionEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decisions: list[AgentDecision] = Field(min_length=1, max_length=3)


@dataclass
class LLMTickDecider:
    client: LLMClient
    temperature: float = 0.2
    max_agents_per_tick: int = 3

    @classmethod
    def from_env(cls) -> "LLMTickDecider":
        client = LLMClient.from_env()

        try:
            temperature = float(os.getenv("LLM_DECIDER_TEMPERATURE", "0.2"))
        except ValueError:
            temperature = 0.2
        temperature = max(0.0, min(temperature, 1.0))

        try:
            max_agents = int(os.getenv("LLM_DECIDER_MAX_AGENTS_PER_TICK", "3"))
        except ValueError:
            max_agents = 3
        max_agents = max(1, min(max_agents, 3))

        return cls(
            client=client,
            temperature=temperature,
            max_agents_per_tick=max_agents,
        )

    @property
    def enabled(self) -> bool:
        return self.client.enabled

    def _debug_enabled(self) -> bool:
        return self.client.debug

    def _debug(self, message: str) -> None:
        if self._debug_enabled():
            logging.getLogger("app.sim.llm_decider").warning(message)

    def decide(
        self,
        tick: int,
        world_summary: dict[str, Any],
        agents_context: list[dict[str, Any]],
        expected_agent_ids: list[str],
    ) -> dict[str, AgentDecision]:
        if not self.enabled or not agents_context or not expected_agent_ids:
            return {}

        response_obj = self.client.request_json_object(
            system_prompt=self._system_prompt(),
            user_payload=self._user_payload(
                tick=tick,
                world_summary=world_summary,
                agents_context=agents_context,
            ),
            temperature=self.temperature,
            json_schema=TickDecisionEnvelope.model_json_schema(),
        )
        if not response_obj:
            self._debug("LLM decider got empty/invalid JSON object from client")
            return {}

        try:
            envelope = TickDecisionEnvelope.model_validate(response_obj)
        except ValidationError as exc:
            self._debug(f"LLM decider schema validation failed: {exc.errors()}")
            return {}

        expected = set(expected_agent_ids)
        actual_ids = [decision.agent_id for decision in envelope.decisions]
        if len(actual_ids) != len(expected_agent_ids):
            self._debug(
                f"LLM decider count mismatch expected={len(expected_agent_ids)} actual={len(actual_ids)}"
            )
            return {}
        if set(actual_ids) != expected:
            self._debug(f"LLM decider agent_id mismatch expected={sorted(expected)} actual={sorted(set(actual_ids))}")
            return {}
        if len(set(actual_ids)) != len(actual_ids):
            self._debug("LLM decider duplicate agent_id in decisions")
            return {}

        return {decision.agent_id: decision for decision in envelope.decisions}

    def _system_prompt(self) -> str:
        return (
            "Ты движок принятия решений для мультиагентной симуляции.\n"
            "Отвечай строго одним JSON-объектом и только по схеме ниже.\n"
            "Не используй markdown, пояснения, комментарии и code fences.\n"
            "LLM предлагает решение, но не управляет world state напрямую.\n"
            "Сервер сам применяет правила/ограничения и может отклонить решение.\n"
            "\n"
            "Глобальные законы мира:\n"
            "1) Конфликт — редкое событие. В большинстве ситуаций агенты стараются сотрудничать или быть нейтральными.\n"
            "2) Цель агента — поддерживать связность общества, избегать бессмысленного ухода в одиночество.\n"
            "3) Если агент уходит, он должен иметь причину.\n"
            "\n"
            "Схема ответа:\n"
            "{\n"
            '  "decisions": [\n'
            "    {\n"
            '      "agent_id": "string",\n'
            '      "goal": "short string",\n'
            '      "act": "move|say|message|idle",\n'
            '      "target_id": "string|null",\n'
            '      "say_text": "string|null",\n'
            '      "move_to": {"x": number, "z": number} | null,\n'
            '      "deltas": {\n'
            '        "self_mood": int(-10..10),\n'
            '        "relations": [{"to_id":"string","delta":int(-5..5)}]\n'
            "      } | null\n"
            "    }\n"
            "  ]\n"
            "}"
        )

    def _user_payload(
        self,
        *,
        tick: int,
        world_summary: dict[str, Any],
        agents_context: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "task": "На текущий тик верни по одному решению для каждого агента из списка agents.",
            "tick": tick,
            "world": world_summary,
            "agents": agents_context,
            "hard_limits": {
                "decisions_count_must_match_agents_input": True,
                "max_goal_len": 120,
                "max_say_text_len": 280,
            },
        }
