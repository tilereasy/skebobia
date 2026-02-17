from __future__ import annotations

import logging
import os
from collections.abc import Mapping
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

    decisions: list[AgentDecision] = Field(min_length=1, max_length=16)


@dataclass
class LLMTickDecider:
    client: LLMClient
    temperature: float = 0.2
    max_agents_per_tick: int = 4
    strict_schema_validation: bool = False

    @classmethod
    def from_env(cls) -> "LLMTickDecider":
        client = LLMClient.from_env()

        try:
            temperature = float(os.getenv("LLM_DECIDER_TEMPERATURE", "0.2"))
        except ValueError:
            temperature = 0.2
        temperature = max(0.0, min(temperature, 1.0))

        try:
            max_agents = int(os.getenv("LLM_DECIDER_MAX_AGENTS_PER_TICK", "4"))
        except ValueError:
            max_agents = 4
        max_agents = max(1, min(max_agents, 16))

        strict_schema_raw = os.getenv("LLM_DECIDER_STRICT_SCHEMA_VALIDATION")
        strict_schema_validation = (
            client.strict_json_schema if strict_schema_raw is None else strict_schema_raw.strip().lower() in {"1", "true", "yes", "on"}
        )

        return cls(
            client=client,
            temperature=temperature,
            max_agents_per_tick=max_agents,
            strict_schema_validation=strict_schema_validation,
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

        minimum_output_tokens = 220 + max(0, len(expected_agent_ids) - 1) * 140
        minimum_output_tokens = max(220, min(minimum_output_tokens, 2600))
        response_obj = self.client.request_json_object(
            system_prompt=self._system_prompt(),
            user_payload=self._user_payload(
                tick=tick,
                world_summary=world_summary,
                agents_context=agents_context,
                expected_agent_ids=expected_agent_ids,
            ),
            temperature=self.temperature,
            json_schema=TickDecisionEnvelope.model_json_schema(),
            minimum_output_tokens=minimum_output_tokens,
        )
        if not response_obj:
            self._debug("LLM decider got empty/invalid JSON object from client")
            return {}

        strict_decisions = self._validate_strict(response_obj=response_obj, expected_agent_ids=expected_agent_ids)
        if strict_decisions and len(strict_decisions) == len(expected_agent_ids):
            return strict_decisions
        if self.strict_schema_validation:
            if strict_decisions:
                self._debug("LLM decider strict schema mode: rejected partial strict decision set")
            return {}

        relaxed_decisions = self._validate_relaxed(response_obj=response_obj, expected_agent_ids=expected_agent_ids)
        if not relaxed_decisions:
            self._debug("LLM decider fallback: no valid decisions after relaxed parsing")
        return relaxed_decisions

    def _validate_strict(
        self,
        *,
        response_obj: dict[str, Any],
        expected_agent_ids: list[str],
    ) -> dict[str, AgentDecision]:
        try:
            envelope = TickDecisionEnvelope.model_validate(response_obj)
        except ValidationError as exc:
            self._debug(f"LLM decider schema validation failed: {exc.errors()}")
            return {}

        expected = set(expected_agent_ids)
        decisions_by_id: dict[str, AgentDecision] = {}
        ignored_ids: set[str] = set()

        for decision in envelope.decisions:
            if decision.agent_id not in expected:
                ignored_ids.add(decision.agent_id)
                continue
            if decision.agent_id in decisions_by_id:
                self._debug(f"LLM decider duplicate agent_id ignored: {decision.agent_id}")
                continue
            decisions_by_id[decision.agent_id] = decision

        if ignored_ids:
            self._debug(f"LLM decider ignored decisions for unknown ids: {sorted(ignored_ids)}")

        missing = [agent_id for agent_id in expected_agent_ids if agent_id not in decisions_by_id]
        if missing:
            self._debug(f"LLM decider strict parse missing ids: {missing}")
        return decisions_by_id

    def _validate_relaxed(
        self,
        *,
        response_obj: dict[str, Any],
        expected_agent_ids: list[str],
    ) -> dict[str, AgentDecision]:
        expected = set(expected_agent_ids)
        candidates = self._extract_decision_candidates(
            payload=response_obj,
            expected_agent_ids=expected_agent_ids,
        )
        if not candidates:
            self._debug("LLM decider relaxed parse found no decision candidates")
            return {}

        decisions_by_id: dict[str, AgentDecision] = {}
        for idx, candidate in enumerate(candidates):
            normalized = self._normalize_candidate(
                raw_candidate=candidate,
                expected_agent_ids=expected_agent_ids,
                index=idx,
            )
            if not normalized:
                continue

            decision = self._validate_normalized_decision(normalized)
            if decision is None:
                continue

            if decision.agent_id not in expected:
                if idx < len(expected_agent_ids):
                    fallback_id = expected_agent_ids[idx]
                    if fallback_id not in decisions_by_id:
                        patched = dict(normalized)
                        patched["agent_id"] = fallback_id
                        patched_decision = self._validate_normalized_decision(patched)
                        if patched_decision is not None:
                            decisions_by_id[fallback_id] = patched_decision
                            continue
                self._debug(f"LLM decider relaxed ignored unknown agent_id={decision.agent_id!r}")
                continue

            if decision.agent_id in decisions_by_id:
                self._debug(f"LLM decider relaxed duplicate agent_id ignored: {decision.agent_id}")
                continue
            decisions_by_id[decision.agent_id] = decision

        missing = [agent_id for agent_id in expected_agent_ids if agent_id not in decisions_by_id]
        if missing:
            self._debug(f"LLM decider relaxed parse missing ids: {missing}")
        return decisions_by_id

    def _validate_normalized_decision(self, normalized: dict[str, Any]) -> AgentDecision | None:
        try:
            return AgentDecision.model_validate(normalized)
        except ValidationError as exc:
            self._debug(f"LLM decider relaxed decision rejected: {exc.errors()} payload={normalized!r}")
            return None

    def _extract_decision_candidates(
        self,
        *,
        payload: Mapping[str, Any],
        expected_agent_ids: list[str],
        depth: int = 0,
    ) -> list[Mapping[str, Any]]:
        if depth > 3:
            return []

        for key in ("decisions", "actions", "items", "agents", "result", "data", "response"):
            value = payload.get(key)
            if isinstance(value, list):
                filtered = [
                    item
                    for item in value
                    if isinstance(item, Mapping) and self._looks_like_decision(item)
                ]
                if filtered:
                    return filtered
            if isinstance(value, Mapping):
                nested = self._extract_decision_candidates(
                    payload=value,
                    expected_agent_ids=expected_agent_ids,
                    depth=depth + 1,
                )
                if nested:
                    return nested

        nested_payload = payload.get("payload")
        if isinstance(nested_payload, Mapping):
            nested = self._extract_decision_candidates(
                payload=nested_payload,
                expected_agent_ids=expected_agent_ids,
                depth=depth + 1,
            )
            if nested:
                return nested

        single_decision = payload.get("decision")
        if isinstance(single_decision, Mapping):
            return [single_decision]

        by_agent: list[Mapping[str, Any]] = []
        for agent_id in expected_agent_ids:
            item = payload.get(agent_id)
            if not isinstance(item, Mapping):
                continue
            merged = dict(item)
            merged.setdefault("agent_id", agent_id)
            by_agent.append(merged)
        if by_agent:
            return by_agent

        if self._looks_like_decision(payload):
            return [payload]
        return []

    def _looks_like_decision(self, payload: Mapping[str, Any]) -> bool:
        keys = set(payload.keys())
        return bool(
            keys
            & {
                "act",
                "action",
                "kind",
                "goal",
                "plan",
                "say_text",
                "text",
                "message",
                "move_to",
                "destination",
                "deltas",
                "delta",
            }
        )

    def _normalize_candidate(
        self,
        *,
        raw_candidate: Mapping[str, Any],
        expected_agent_ids: list[str],
        index: int,
    ) -> dict[str, Any] | None:
        candidate = dict(raw_candidate)
        nested = candidate.get("decision")
        if isinstance(nested, Mapping):
            merged = dict(nested)
            for key, value in candidate.items():
                if key != "decision" and key not in merged:
                    merged[key] = value
            candidate = merged

        agent_id = self._first_str(
            candidate,
            ("agent_id", "agentId", "agent", "id", "for_agent", "agentID"),
        )
        if not agent_id and index < len(expected_agent_ids):
            agent_id = expected_agent_ids[index]
        if not agent_id:
            return None

        raw_act = self._first_str(candidate, ("act", "action", "kind", "type"))
        act = self._normalize_act(raw_act, candidate)
        goal = self._first_str(candidate, ("goal", "plan", "current_plan", "intent", "objective")) or "Keep social coherence"

        target_id = self._first_str(
            candidate,
            ("target_id", "targetId", "target", "to_id", "to", "recipient_id", "recipient"),
        )
        if target_id and target_id.lower() in {"none", "null"}:
            target_id = None

        say_text = self._first_str(
            candidate,
            ("say_text", "sayText", "text", "message", "utterance", "speech", "content", "reply"),
        )
        move_to = self._extract_move_to(candidate)
        deltas = self._extract_deltas(candidate)

        if act in {"say", "message"} and say_text:
            say_text = say_text[:280]
        if act not in {"say", "message"}:
            say_text = None
        if act != "move":
            move_to = None

        return {
            "agent_id": agent_id[:64],
            "goal": goal[:120],
            "act": act,
            "target_id": target_id[:64] if target_id else None,
            "say_text": say_text,
            "move_to": move_to,
            "deltas": deltas,
        }

    def _normalize_act(self, raw_act: str | None, payload: Mapping[str, Any]) -> str:
        if raw_act:
            normalized = raw_act.strip().lower()
            aliases = {
                "move": "move",
                "walk": "move",
                "go": "move",
                "relocate": "move",
                "say": "say",
                "speak": "say",
                "talk": "say",
                "reply": "say",
                "message": "message",
                "dm": "message",
                "send_message": "message",
                "idle": "idle",
                "wait": "idle",
                "observe": "idle",
                "noop": "idle",
                "none": "idle",
            }
            if normalized in aliases:
                return aliases[normalized]

        if self._first_value(payload, ("move_to", "moveTo", "destination", "position", "pos", "coords")) is not None:
            return "move"
        target_hint = self._first_str(
            payload,
            ("target_id", "targetId", "target", "to_id", "to", "recipient_id", "recipient"),
        )
        if target_hint and self._first_str(payload, ("say_text", "sayText", "text", "message", "utterance", "reply")):
            return "message"
        if self._first_str(payload, ("say_text", "sayText", "text", "message", "utterance", "reply")):
            return "say"
        return "idle"

    def _extract_move_to(self, payload: Mapping[str, Any]) -> dict[str, float] | None:
        raw = self._first_value(payload, ("move_to", "moveTo", "destination", "position", "pos", "coords"))
        x: float | None = None
        z: float | None = None

        if isinstance(raw, Mapping):
            x = self._to_float(raw.get("x"))
            z = self._to_float(raw.get("z"))
            if z is None:
                z = self._to_float(raw.get("y"))
        elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
            x = self._to_float(raw[0])
            z = self._to_float(raw[1])
        elif isinstance(raw, str):
            compact = raw.replace(" ", "")
            if "," in compact:
                left, right = compact.split(",", 1)
                x = self._to_float(left)
                z = self._to_float(right)

        if x is None:
            x = self._to_float(payload.get("x"))
        if z is None:
            z = self._to_float(payload.get("z"))

        if x is None or z is None:
            return None
        return {"x": x, "z": z}

    def _extract_deltas(self, payload: Mapping[str, Any]) -> dict[str, Any] | None:
        raw = self._first_value(payload, ("deltas", "delta", "effects", "changes"))
        if not isinstance(raw, Mapping):
            return None

        self_mood = self._to_int(
            self._first_value(raw, ("self_mood", "mood", "mood_delta", "selfMood")),
        )
        if self_mood is None:
            self_mood = 0

        relations_raw = self._first_value(raw, ("relations", "relation_deltas", "relationDeltas"))
        relations: list[dict[str, Any]] = []
        if isinstance(relations_raw, list):
            for rel_item in relations_raw[:8]:
                if not isinstance(rel_item, Mapping):
                    continue
                to_id = self._first_str(rel_item, ("to_id", "to", "target_id", "target"))
                delta = self._to_int(self._first_value(rel_item, ("delta", "value", "change")))
                if to_id is None or delta is None:
                    continue
                relations.append({"to_id": to_id[:64], "delta": delta})

        if self_mood == 0 and not relations:
            return None
        return {"self_mood": self_mood, "relations": relations}

    def _first_value(self, payload: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
        for key in keys:
            if key in payload:
                return payload.get(key)
        return None

    def _first_str(self, payload: Mapping[str, Any], keys: tuple[str, ...]) -> str | None:
        for key in keys:
            value = payload.get(key)
            if isinstance(value, str):
                trimmed = value.strip()
                if trimmed:
                    return trimmed
        return None

    def _to_float(self, value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return None
        return None

    def _to_int(self, value: Any) -> int | None:
        numeric = self._to_float(value)
        if numeric is None:
            return None
        return int(round(numeric))

    def _system_prompt(self) -> str:
        return (
            "You are a decision engine for a live multi-agent social simulation.\n"
            "Output strictly ONE minified JSON object on a single line.\n"
            "No markdown, no comments, no code fences, no trailing text.\n"
            "Use only ids from USER_CONTEXT_JSON.expected_agent_ids.\n"
            "Return exactly one decision per expected agent id.\n"
            "If a field is not used, set null.\n"
            "The server is authoritative and may reject unsafe choices.\n"
            "Conflicts are rare; prefer neutral or cooperative behavior.\n"
            "Actions must keep social coherence and avoid pointless isolation.\n"
            "Primary dialogue language is Russian unless incoming context is clearly another language.\n"
            "Use each agent's mood, traits and recent context to create distinct voice per agent.\n"
            "Avoid generic assistant boilerplate (e.g. 'I'm here for you').\n"
            "Do not mirror or copy recent messages verbatim; rephrase with new wording.\n"
            "say_text must reference a concrete context signal (topic, inbox item, relation, or world event).\n"
            "Keep strings short: goal up to 60 chars, say_text up to 90 chars.\n"
            "\n"
            "Format:\n"
            "{\n"
            '  "decisions": [\n'
            "    {\n"
            '      "agent_id": "expected id",\n'
            '      "goal": "1 short sentence",\n'
            '      "act": "move|say|message|idle",\n'
            '      "target_id": "agent id|null",\n'
            '      "say_text": "text|null",\n'
            '      "move_to": {"x": number, "z": number} | null,\n'
            '      "deltas": {"self_mood": int(-10..10), "relations":[{"to_id":"id","delta":int(-5..5)}]} | null\n'
            "    }\n"
            "  ]\n"
            "}\n"
        )

    def _user_payload(
        self,
        *,
        tick: int,
        world_summary: dict[str, Any],
        agents_context: list[dict[str, Any]],
        expected_agent_ids: list[str],
    ) -> dict[str, Any]:
        return {
            "task": "Return one decision per expected_agent_ids entry for the current tick.",
            "tick": tick,
            "expected_agent_ids": expected_agent_ids,
            "world": world_summary,
            "agents": agents_context,
            "style_hints": {
                "primary_language": "ru",
                "must_sound_in_world": True,
                "avoid_verbatim_repeat": True,
                "avoid_generic_assistant_tone": True,
            },
            "hard_limits": {
                "decisions_count_must_match_agents_input": True,
                "agent_ids_must_match_expected_agent_ids": True,
                "max_goal_len": 120,
                "max_say_text_len": 280,
                "prefer_goal_len": 60,
                "prefer_say_text_len": 90,
                "response_should_be_minified_json_single_line": True,
                "allowed_act_values": ["move", "say", "message", "idle"],
            },
        }
