from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
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
    say_text: str | None = Field(default=None, max_length=360)
    speech_intent: Literal["inform", "propose", "ask", "confirm", "coordinate"] | None = Field(
        default=None,
        validation_alias=AliasChoices("speech_intent", "speechIntent", "utterance_role", "reply_role", "speech_role"),
    )
    move_to: MoveTo | None = None
    evidence_ids: list[str] | None = Field(default=None, max_length=3)
    deltas: DecisionDeltas | None = None

    @model_validator(mode="after")
    def normalize_shape(self) -> "AgentDecision":
        if self.target_id is not None:
            target = self.target_id.strip()
            if not target or target.lower() in {"null", "none", "nil", "n/a"}:
                self.target_id = None
            else:
                self.target_id = target[:64]

        if self.say_text is not None:
            say_text = self.say_text.strip()
            if not say_text or say_text.lower() in {"null", "none", "nil", "n/a"}:
                self.say_text = None
            else:
                self.say_text = say_text[:360]
                words = [token for token in self.say_text.split() if any(ch.isalpha() for ch in token)]
                if len(words) < 2:
                    self.say_text = None

        if self.speech_intent is not None:
            role = self.speech_intent.strip().lower()
            allowed_roles = {"inform", "propose", "ask", "confirm", "coordinate"}
            if role in {"null", "none", "nil", "n/a", ""}:
                self.speech_intent = None
            elif role in allowed_roles:
                self.speech_intent = role  # type: ignore[assignment]
            else:
                self.speech_intent = None

        if self.evidence_ids is not None:
            normalized_evidence_ids: list[str] = []
            seen: set[str] = set()
            for item in self.evidence_ids:
                if not isinstance(item, str):
                    continue
                cleaned = item.strip()
                if not cleaned or cleaned.lower() in {"null", "none", "nil", "n/a"}:
                    continue
                key = cleaned.casefold()
                if key in seen:
                    continue
                seen.add(key)
                normalized_evidence_ids.append(cleaned[:64])
                if len(normalized_evidence_ids) >= 3:
                    break
            self.evidence_ids = normalized_evidence_ids or None

        if self.act == "message" and not self.target_id:
            self.act = "say"
        if self.act in {"say", "message"} and self.say_text is None:
            self.act = "idle"

        if self.act != "move":
            self.move_to = None
        if self.act not in {"say", "message"}:
            self.say_text = None
            self.speech_intent = None
        if self.act == "idle":
            self.target_id = None
        if self.act in {"say", "message"} and self.say_text and self.speech_intent is None:
            lowered = self.say_text.lower()
            if "?" in lowered:
                self.speech_intent = "ask"
            elif any(token in lowered for token in ("предлага", "давай", "план", "собер")):
                self.speech_intent = "propose"
            elif any(token in lowered for token in ("принял", "понял", "согласен", "подтвержда")):
                self.speech_intent = "confirm"
            elif any(token in lowered for token in ("в", "у", "через")) and self.act == "message":
                self.speech_intent = "coordinate"
            else:
                self.speech_intent = "inform"
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
    trace_callback: Callable[[dict[str, Any]], None] | None = None

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

    def _emit_trace(self, payload: dict[str, Any]) -> None:
        if self.trace_callback is None:
            return
        try:
            self.trace_callback(payload)
        except Exception as exc:
            self._debug(f"LLM decider trace callback failed: {exc!r}")

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
        system_prompt = self._system_prompt()
        user_payload = self._user_payload(
            tick=tick,
            world_summary=world_summary,
            agents_context=agents_context,
            expected_agent_ids=expected_agent_ids,
        )
        response_obj = self.client.request_json_object(
            system_prompt=system_prompt,
            user_payload=user_payload,
            temperature=self.temperature,
            json_schema=TickDecisionEnvelope.model_json_schema(),
            minimum_output_tokens=minimum_output_tokens,
        )
        self._emit_trace(
            {
                "tick": tick,
                "expected_agent_ids": list(expected_agent_ids),
                "prompt": {
                    "system_prompt": system_prompt,
                    "user_payload": user_payload,
                },
                "response": response_obj,
            }
        )
        if not response_obj:
            self._debug("LLM decider got empty/invalid JSON object from client")
            return {}

        strict_decisions = self._validate_strict(response_obj=response_obj, expected_agent_ids=expected_agent_ids)
        if strict_decisions and len(strict_decisions) == len(expected_agent_ids):
            self._debug(
                "LLM decider accepted decisions "
                f"branch=strict_full count={len(strict_decisions)} ids={sorted(strict_decisions.keys())}"
            )
            return strict_decisions
        relaxed_decisions = self._validate_relaxed(response_obj=response_obj, expected_agent_ids=expected_agent_ids)
        chosen: dict[str, AgentDecision] = {}
        branch = "none"
        if self.strict_schema_validation:
            if strict_decisions and relaxed_decisions:
                merged = dict(strict_decisions)
                for agent_id in expected_agent_ids:
                    if agent_id not in merged and agent_id in relaxed_decisions:
                        merged[agent_id] = relaxed_decisions[agent_id]
                if len(merged) != len(strict_decisions):
                    self._debug("LLM decider strict schema mode: filled missing strict decisions via relaxed parse")
                chosen = merged
                branch = "strict_plus_relaxed"
            elif strict_decisions:
                self._debug("LLM decider strict schema mode: using partial strict decision set")
                chosen = strict_decisions
                branch = "strict_partial"
            elif relaxed_decisions:
                self._debug("LLM decider strict schema mode: strict parse failed, using relaxed decisions")
                chosen = relaxed_decisions
                branch = "relaxed_in_strict_mode"
            else:
                self._debug("LLM decider strict schema mode: no valid decisions after strict/relaxed parsing")
                chosen = {}
                branch = "none"
        else:
            if strict_decisions and relaxed_decisions:
                merged = dict(strict_decisions)
                for agent_id in expected_agent_ids:
                    if agent_id not in merged and agent_id in relaxed_decisions:
                        merged[agent_id] = relaxed_decisions[agent_id]
                chosen = merged
                branch = "strict_plus_relaxed"
            elif strict_decisions:
                chosen = strict_decisions
                branch = "strict_partial"
            else:
                if not relaxed_decisions:
                    self._debug("LLM decider fallback: no valid decisions after relaxed parsing")
                chosen = relaxed_decisions
                branch = "relaxed_only" if relaxed_decisions else "none"

        self._debug(
            "LLM decider accepted decisions "
            f"branch={branch} count={len(chosen)} ids={sorted(chosen.keys()) if chosen else []}"
        )
        return chosen

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
        for candidate in candidates:
            normalized = self._normalize_candidate(
                raw_candidate=candidate,
            )
            if not normalized:
                continue

            decision = self._validate_normalized_decision(normalized)
            if decision is None:
                continue

            if decision.agent_id not in expected:
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
                "speech_intent",
                "speechIntent",
                "utterance_role",
                "reply_role",
                "goal",
                "plan",
                "say_text",
                "text",
                "message",
                "move_to",
                "destination",
                "evidence_ids",
                "evidenceIds",
                "evidence",
                "support_ids",
                "supportIds",
                "deltas",
                "delta",
            }
        )

    def _normalize_candidate(
        self,
        *,
        raw_candidate: Mapping[str, Any],
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
        speech_intent = self._first_str(
            candidate,
            ("speech_intent", "speechIntent", "utterance_role", "reply_role", "speech_role"),
        )
        if speech_intent:
            speech_intent = speech_intent.lower()
        move_to = self._extract_move_to(candidate)
        evidence_ids = self._extract_evidence_ids(candidate)
        deltas = self._extract_deltas(candidate)

        if act in {"say", "message"} and say_text:
            say_text = say_text[:360]
        if act not in {"say", "message"}:
            say_text = None
            speech_intent = None
        if act != "move":
            move_to = None

        return {
            "agent_id": agent_id[:64],
            "goal": goal[:120],
            "act": act,
            "target_id": target_id[:64] if target_id else None,
            "say_text": say_text,
            "speech_intent": speech_intent,
            "move_to": move_to,
            "evidence_ids": evidence_ids,
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

    def _extract_evidence_ids(self, payload: Mapping[str, Any]) -> list[str] | None:
        raw = self._first_value(
            payload,
            ("evidence_ids", "evidenceIds", "evidence", "support_ids", "supportIds"),
        )
        if raw is None:
            return None

        candidates: list[str] = []
        if isinstance(raw, str):
            candidates = [part.strip() for part in raw.split(",")]
        elif isinstance(raw, list | tuple):
            candidates = [str(item).strip() for item in raw if isinstance(item, str)]
        else:
            return None

        normalized: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            if not item or item.lower() in {"null", "none", "nil", "n/a"}:
                continue
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(item[:64])
            if len(normalized) >= 3:
                break
        return normalized or None

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
            "You choose actions for agents in a live social simulation.\n"
            "Return strictly ONE minified JSON object on a single line.\n"
            "No markdown, no comments, no trailing text.\n"
            "Use JSON null values, never the string \"null\".\n"
            "Return exactly one decision for each id in USER_CONTEXT_JSON.expected_agent_ids.\n"
            "Use only those ids.\n"
            "If a field is not used, set null.\n"
            "Use agent.allowed_actions and state.cooldowns as hard constraints.\n"
            "If an action is unavailable, choose idle.\n"
            "If queue.must_message_source is true, act must be 'message' and target_id must equal queue.source_id.\n"
            "If queue.answer_first is true, answer queue.text directly and do not ask a new question.\n"
            "If queue.question_allowed_now is false, say_text must not contain '?'.\n"
            "Primary dialogue language is Russian unless context is clearly another language.\n"
            "Agents must speak in first person and never describe themselves in third person.\n"
            "Avoid bureaucratic or project-management tone.\n"
            "Never say: 'принял', 'задача', 'синхронизировать шаги', 'уточню факт', 'вернусь с результатом'.\n"
            "Do not summon, call over, or reposition others (no 'подойди', 'иди сюда', 'встретимся', 'подтянись').\n"
            "Do not repeat interpretation already expressed by another agent.\n"
            "Choose a unique stance.\n"
            "Avoid meta-talk about communication process itself and abstract 'steps'.\n"
            "If speaking, prefer emotion, concrete observation, and personal thought.\n"
            "Do not copy recent messages verbatim.\n"
            "One-word replies are forbidden.\n"
            "Use evidence_ids (up to 3 ids) when relevant.\n"
            "If world.recent_events contains world+micro, at least one eligible agent should react via say/message.\n"
            "When reacting to world+micro, include that world event id in evidence_ids whenever relevant.\n"
            "If queue.first_world_reaction is true, say_text must not contain '?'.\n"
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
            '      "speech_intent": "inform|propose|ask|confirm|coordinate|null",\n'
            '      "move_to": {"x": number, "z": number} | null,\n'
            '      "evidence_ids": ["id1","id2"] | null,\n'
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
        dialogue_capable = 0
        selected_for_reply = 0
        can_skip_selected = 0
        must_text_reply_selected = 0
        must_message_selected = 0
        world_action_reply_selected = 0
        answer_first_selected = 0
        question_blocked_agents = 0
        impulse_agents_without_inbox = 0
        for agent_ctx in agents_context:
            if not isinstance(agent_ctx, dict):
                continue
            allowed = agent_ctx.get("allowed_actions")
            if isinstance(allowed, list) and ("say" in allowed or "message" in allowed):
                dialogue_capable += 1
            queue = agent_ctx.get("queue")
            if not isinstance(queue, dict):
                continue
            if (
                bool(queue.get("internal_impulse_20pct"))
                and int(queue.get("pending_inbox_count", 0)) == 0
            ):
                impulse_agents_without_inbox += 1
            if bool(queue.get("answer_first")) or queue.get("question_allowed_now") is False:
                question_blocked_agents += 1
            if queue.get("selected_for_reply") is not True:
                continue
            selected_for_reply += 1
            if bool(queue.get("answer_first")):
                answer_first_selected += 1
            if bool(queue.get("allow_move_instead_of_say")):
                world_action_reply_selected += 1
                continue
            if bool(queue.get("must_message_source")):
                must_message_selected += 1
                must_text_reply_selected += 1
                continue
            reply_policy = queue.get("reply_policy")
            can_skip = isinstance(reply_policy, dict) and bool(reply_policy.get("can_skip"))
            if can_skip:
                can_skip_selected += 1
            else:
                must_text_reply_selected += 1

        if selected_for_reply > 0:
            # Разрешаем пропуски только для меньшинства выбранных агентов, но большинство действий остаются ответами.
            max_skip_actions = min(can_skip_selected, max(1, selected_for_reply // 3))
            min_dialogue_actions = max(0, max(
                must_text_reply_selected,
                selected_for_reply - max_skip_actions - world_action_reply_selected,
            )
            )
        else:
            min_dialogue_actions = 0 if dialogue_capable == 0 else max(1, dialogue_capable // 2)

        recent_world_micro_event_ids: list[str] = []
        world_recent_events = world_summary.get("recent_events")
        if isinstance(world_recent_events, list):
            for event in reversed(world_recent_events):
                if not isinstance(event, dict):
                    continue
                tags = event.get("tags")
                if not isinstance(tags, list):
                    continue
                normalized_tags = {str(tag).lower() for tag in tags}
                if "world" not in normalized_tags or "micro" not in normalized_tags:
                    continue
                event_id = event.get("id")
                if isinstance(event_id, str) and event_id.strip():
                    recent_world_micro_event_ids.append(event_id.strip())
                    if len(recent_world_micro_event_ids) >= 4:
                        break

        if recent_world_micro_event_ids and dialogue_capable > 0:
            # Микро-бонус: если world/micro только что произошел, подталкиваем хотя бы еще одно диалоговое действие.
            min_dialogue_actions = max(
                min_dialogue_actions,
                min(dialogue_capable, min_dialogue_actions + 1),
            )
        if impulse_agents_without_inbox > 0 and dialogue_capable > 0:
            min_dialogue_actions = min(
                dialogue_capable,
                min_dialogue_actions + impulse_agents_without_inbox,
            )

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
                "first_person_only": True,
                "avoid_process_talk": True,
                "do_not_repeat_interpretation": True,
                "choose_unique_stance": True,
                "prefer_react_to_world_micro": bool(recent_world_micro_event_ids),
            },
            "hard_limits": {
                "decisions_count_must_match_agents_input": True,
                "agent_ids_must_match_expected_agent_ids": True,
                "max_goal_len": 120,
                "max_say_text_len": 360,
                "prefer_goal_len": 60,
                "prefer_say_text_len": 180,
                "min_words_for_reply_text": 8,
                "max_evidence_ids": 3,
                "allowed_speech_intents": ["inform", "propose", "ask", "confirm", "coordinate"],
                "anti_jitter_move_when_last_action_move": True,
                "min_say_or_message_actions_if_allowed": min_dialogue_actions,
                "selected_for_reply_agents": selected_for_reply,
                "selected_reply_answer_first_agents": answer_first_selected,
                "selected_reply_must_message_agents": must_message_selected,
                "selected_reply_agents_allowing_move_action": world_action_reply_selected,
                "recent_world_micro_event_ids": recent_world_micro_event_ids,
                "prefer_world_micro_reaction": bool(recent_world_micro_event_ids),
                "impulse_agents_without_inbox": impulse_agents_without_inbox,
                "internal_impulse_probability_if_no_inbox": 0.2,
                "max_question_actions_for_tick": max(0, len(expected_agent_ids) - question_blocked_agents),
                "max_idle_in_selected_queue": 0 if selected_for_reply == 0 else max(0, selected_for_reply - min_dialogue_actions),
                "response_should_be_minified_json_single_line": True,
                "allowed_act_values": ["move", "say", "message", "idle"],
                "forbidden_phrases": [
                    "принял",
                    "задача",
                    "синхронизировать шаги",
                    "уточню факт",
                    "вернусь с результатом",
                    "подойди",
                    "иди сюда",
                    "встретимся",
                    "подтянись",
                ],
            },
        }
