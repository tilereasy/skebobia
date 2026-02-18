from __future__ import annotations

import json
import logging
import math
import os
import random
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Deque

from app.agents.agent import AgentState, Vec3, plan_for_mood
from app.memory.store import EpisodicMemoryStore, _tokenize
from app.sim import templates
from app.sim.llm_decider import AgentDecision, LLMTickDecider
from app.sim.movement import SAFE_POINT, clamp_position, pick_wander_target, step_towards
from app.sim.relations import apply_ignored_inbox_penalty, clamp as clamp_relation, update_relations_from_event
from app.sim.rules import parse_traits, pick_best_friend, text_sentiment
from app.sim.world_events import GeneratedWorldEvent, WorldEventGenerator


def _utc_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _clamp_float(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(low, min(high, value))


def _env_float(name: str, default: float, low: float, high: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(low, min(high, value))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class TickResult:
    events: list[dict] = field(default_factory=list)
    relations_changed: bool = False


@dataclass
class WorldState:
    agents: dict[str, AgentState]
    relations: dict[tuple[str, str], int]
    event_log: Deque[dict]
    speed: float = 1.0
    tick: int = 0
    next_event_id: int = 0


@dataclass
class ActionIntent:
    kind: str
    text: str = ""
    tags: list[str] = field(default_factory=list)
    target_id: str | None = None
    destination: Vec3 | None = None
    text_kind: str | None = None
    llm_generated: bool = False
    thread_id: str | None = None
    expects_reply: bool | None = None
    force_non_question: bool = False
    speech_intent: str | None = None
    evidence_ids: list[str] = field(default_factory=list)


@dataclass
class ReplyTask:
    id: str
    agent_id: str
    inbox_id: str
    source_id: str | None
    source_type: str
    thread_id: str | None
    expects_reply: bool
    can_reply: bool
    text: str
    tags: tuple[str, ...]
    created_tick: int
    priority: int
    retries: int = 0
    skips: int = 0
    last_attempt_tick: int = 0


@dataclass
class ReplyPolicy:
    can_skip: bool
    skip_chance: float


class StubWorld:
    _WORLD_REACTION_STANCES = ("alarm", "curiosity", "skeptic", "practical", "empathy")

    def __init__(
        self,
        relations_interval_ticks: int = 5,
        history_limit: int = 300,
        memory_short_limit: int = 20,
        recent_window: int = 20,
    ):
        agents = self._build_agents()
        relations = self._build_relations(list(agents.keys()))
        self.relations_interval_ticks = max(1, relations_interval_ticks)
        self.memory_short_limit = max(5, memory_short_limit)
        self.recent_window = max(10, recent_window)
        self.state = WorldState(
            agents=agents,
            relations=relations,
            event_log=deque(maxlen=history_limit),
        )
        self.agent_home_positions: dict[str, Vec3] = {
            agent_id: Vec3(x=agent.pos.x, y=agent.pos.y, z=agent.pos.z)
            for agent_id, agent in agents.items()
        }

        self.llm_decider = LLMTickDecider.from_env()
        self.llm_first_hard_mode = _env_bool("LLM_FIRST_HARD_MODE", True)
        self.llm_force_user_reply_via_llm = _env_bool("LLM_FORCE_USER_REPLY_VIA_LLM", True)
        self.llm_trace_events_enabled = _env_bool("LLM_TRACE_EVENTS_ENABLED", True)
        self.llm_trace_logs_enabled = _env_bool("LLM_TRACE_LOGS_ENABLED", True)
        self.llm_trace_max_chars = _env_int("LLM_TRACE_EVENT_MAX_CHARS", 6000, 512, 50000)
        if self.llm_trace_events_enabled or self.llm_trace_logs_enabled:
            self.llm_decider.trace_callback = self._emit_llm_trace_events
        self.memory_store = EpisodicMemoryStore.from_env()

        self.prompt_recent_events_limit = _env_int("LLM_PROMPT_RECENT_EVENTS", 6, 2, 24)
        self.prompt_inbox_limit = _env_int("LLM_PROMPT_INBOX", 4, 1, 12)
        self.prompt_memories_limit = _env_int("LLM_PROMPT_MEMORIES", 5, 1, 12)
        self.single_agent_backfill_retries = _env_int("LLM_DECIDER_BACKFILL_RETRIES", 2, 0, 4)

        self.target_llm_response_ratio = _env_float("LLM_TARGET_RESPONSE_RATIO", 0.9, 0.1, 0.99)
        self.response_ratio_window = _env_int("LLM_RESPONSE_RATIO_WINDOW", 240, 20, 1000)
        self.dialogue_llm_window: Deque[int] = deque(maxlen=self.response_ratio_window)
        self.llm_retry_on_must_answer = _env_bool("LLM_RETRY_ON_MUST_ANSWER", False)

        self.max_replies_per_tick = _env_int("REPLY_QUEUE_MAX_REPLIES_PER_TICK", 2, 1, 16)
        self.reply_queue_max_wait_ticks = _env_int("REPLY_QUEUE_MAX_WAIT_TICKS", 10, 2, 60)
        self.reply_queue_max_skips = _env_int("REPLY_QUEUE_MAX_SKIPS", 2, 0, 8)
        self.world_reply_ttl_ticks = _env_int("REPLY_WORLD_TTL_TICKS", 8, 1, 120)
        self.unqueued_reply_release_ticks = _env_int("REPLY_UNQUEUED_RELEASE_TICKS", 4, 1, 32)
        self.question_min_interval_ticks = _env_int("QUESTION_MIN_INTERVAL_TICKS", 6, 1, 120)
        self.question_max_interval_ticks = _env_int("QUESTION_MAX_INTERVAL_TICKS", 10, self.question_min_interval_ticks, 240)
        self.reply_min_words_default = _env_int("REPLY_MIN_WORDS", 8, 3, 30)
        self.reply_min_words_danger = _env_int("REPLY_MIN_WORDS_DANGER", 4, 1, 20)
        self.reply_queue: Deque[ReplyTask] = deque(maxlen=_env_int("REPLY_QUEUE_MAX_SIZE", 512, 32, 4096))
        self.reply_task_by_inbox_id: dict[str, ReplyTask] = {}
        self.reply_task_seq = 0
        self.inbox_seq = 0
        self.proactive_llm_agents_per_tick = _env_int("LLM_PROACTIVE_AGENTS_PER_TICK", 1, 0, 8)
        self.startup_world_event_enabled = _env_bool("STARTUP_WORLD_EVENT_ENABLED", True)
        self.startup_world_event_importance = _env_float("STARTUP_WORLD_EVENT_IMPORTANCE", 0.85, 0.2, 1.0)
        self.chaotic_move_radius = _env_float("CHAOTIC_MOVE_RADIUS", 8.0, 2.0, 18.0)
        self.chaotic_move_min_distance = _env_float("CHAOTIC_MOVE_MIN_DISTANCE", 1.4, 0.4, 4.0)
        self.chaotic_move_step = _env_float("CHAOTIC_MOVE_STEP", 2.2, 0.3, 6.0)
        self.chaotic_background_move_step = _env_float("CHAOTIC_BACKGROUND_MOVE_STEP", 0.38, 0.05, 2.2)
        self.relations_passive_step_ticks = _env_int(
            "RELATIONS_PASSIVE_STEP_TICKS",
            max(2, self.relations_interval_ticks * 2),
            1,
            240,
        )
        self.relations_passive_max_delta = _env_int("RELATIONS_PASSIVE_MAX_DELTA", 2, 1, 6)
        self.relations_recent_window_ticks = _env_int("RELATIONS_RECENT_WINDOW_TICKS", 24, 4, 240)
        self.relations_recent_event_weight = _env_int("RELATIONS_RECENT_EVENT_WEIGHT", 3, 0, 12)
        self.relations_proximity_radius = _env_float("RELATIONS_PROXIMITY_RADIUS", 3.4, 0.5, 20.0)
        self.relations_distance_penalty_radius = _env_float("RELATIONS_DISTANCE_PENALTY_RADIUS", 8.0, 1.0, 30.0)
        self.world_event_generator = WorldEventGenerator.from_env(client=self.llm_decider.client)
        self.world_events_logger = logging.getLogger("app.sim.world_events")

        self.reply_policy_by_agent: dict[str, ReplyPolicy] = {}
        for agent in agents.values():
            self.reply_policy_by_agent[agent.id] = self._build_reply_policy(agent)

        self._seed_initial_events()
        self._trigger_startup_world_event()

    @property
    def speed(self) -> float:
        return self.state.speed

    @property
    def world_state(self) -> WorldState:
        return self.state

    def _ordered_agents(self) -> list[AgentState]:
        return [self.state.agents[agent_id] for agent_id in sorted(self.state.agents.keys())]

    def _agent_name(self, agent_id: str | None) -> str | None:
        if not agent_id:
            return None
        agent = self.state.agents.get(agent_id)
        return agent.name if agent else None

    def _distance_2d(self, lhs: Vec3, rhs: Vec3) -> float:
        dx = lhs.x - rhs.x
        dz = lhs.z - rhs.z
        return (dx * dx + dz * dz) ** 0.5

    def _clamp_to_home_radius(self, agent_id: str, position: Vec3) -> Vec3:
        home = self.agent_home_positions.get(agent_id)
        if home is None:
            return clamp_position(position)
        dx = position.x - home.x
        dz = position.z - home.z
        distance = math.sqrt(dx * dx + dz * dz)
        if distance <= self.chaotic_move_radius:
            return clamp_position(position)
        ratio = self.chaotic_move_radius / max(distance, 1e-6)
        return clamp_position(Vec3(home.x + dx * ratio, 0.0, home.z + dz * ratio))

    def _separate_from_other_agents(self, agent_id: str, position: Vec3) -> Vec3:
        adjusted = Vec3(position.x, 0.0, position.z)
        min_distance = self.chaotic_move_min_distance
        for _ in range(2):
            moved = False
            for other in self._ordered_agents():
                if other.id == agent_id:
                    continue
                distance = self._distance_2d(adjusted, other.pos)
                if distance >= min_distance:
                    continue

                push = (min_distance - distance) + 0.05
                if distance < 1e-6:
                    angle_seed = (sum(ord(ch) for ch in agent_id) * 73 + self.state.tick * 29) % 6283
                    angle = angle_seed / 1000.0
                    away_x = math.cos(angle)
                    away_z = math.sin(angle)
                else:
                    away_x = (adjusted.x - other.pos.x) / distance
                    away_z = (adjusted.z - other.pos.z) / distance

                adjusted = Vec3(adjusted.x + away_x * push, 0.0, adjusted.z + away_z * push)
                adjusted = self._clamp_to_home_radius(agent_id, adjusted)
                moved = True
            if not moved:
                break
        return clamp_position(adjusted)

    def _chaotic_destination(self, agent: AgentState) -> Vec3:
        seed = (
            self.state.tick * 131
            + sum(ord(ch) for ch in agent.id) * 17
            + len(agent.inbox) * 29
            + len(agent.memory_short) * 7
        )
        angle = ((seed % 6283) / 1000.0) % (2 * math.pi)
        distance = 0.55 + (((seed // 11) % 100) / 100.0) * self.chaotic_move_step
        candidate = Vec3(
            agent.pos.x + math.cos(angle) * distance,
            0.0,
            agent.pos.z + math.sin(angle) * distance,
        )
        candidate = self._clamp_to_home_radius(agent.id, candidate)
        candidate = self._separate_from_other_agents(agent.id, candidate)
        return candidate

    def _question_interval_for_agent(self, agent_id: str) -> int:
        span = max(0, self.question_max_interval_ticks - self.question_min_interval_ticks)
        if span == 0:
            return self.question_min_interval_ticks
        return self.question_min_interval_ticks + (sum(ord(ch) for ch in agent_id) % (span + 1))

    @staticmethod
    def _text_has_question(text: str) -> bool:
        return "?" in str(text).strip()

    def _can_ask_question_now(self, agent: AgentState) -> bool:
        interval = self._question_interval_for_agent(agent.id)
        return (self.state.tick - agent.last_question_tick) >= interval

    @staticmethod
    def _strip_question_form(text: str) -> str:
        cleaned = " ".join(str(text).replace("?", ".").split()).strip()
        if not cleaned:
            return "Понял."
        if cleaned[-1] not in ".!":
            cleaned = f"{cleaned}."
        return cleaned

    def _enforce_question_policy_text(
        self,
        agent: AgentState,
        text: str,
        *,
        force_non_question: bool = False,
    ) -> str:
        if not self._text_has_question(text):
            return text
        if force_non_question or not self._can_ask_question_now(agent):
            return self._strip_question_form(text)
        return text

    def _mark_question_emitted(self, agent: AgentState, text: str) -> None:
        if self._text_has_question(text):
            agent.last_question_tick = self.state.tick

    @staticmethod
    def _word_count(text: str) -> int:
        return sum(1 for token in str(text).split() if any(ch.isalpha() for ch in token))

    @staticmethod
    def _contains_operational_tone(text: str) -> bool:
        lowered = str(text).lower()
        blocked_markers = (
            "принял",
            "задача",
            "синхронизировать шаги",
            "синхрониз",
            "уточню факт",
            "вернусь с результат",
        )
        return any(marker in lowered for marker in blocked_markers)

    @staticmethod
    def _contains_summon_language(text: str) -> bool:
        lowered = str(text).lower()
        blocked_markers = (
            "подойди",
            "подходи",
            "подтянись",
            "иди сюда",
            "иди ко мне",
            "встретимся",
            "давай встретимся",
            "подойду к тебе",
            "подойду ближе",
            "подожди у",
            "жду тебя у",
        )
        return any(marker in lowered for marker in blocked_markers)

    @staticmethod
    def _has_first_person_marker(text: str) -> bool:
        lowered = f" {str(text).lower()} "
        markers = (" я ", " мне ", " меня ", " мой ", " моя ", " мое ", " моё ", " мы ", " нам ", " нас ")
        return any(marker in lowered for marker in markers)

    def _is_third_person_self_reference(self, agent: AgentState, text: str) -> bool:
        lowered = str(text).lower()
        name_marker = agent.name.lower()
        if not name_marker or name_marker not in lowered:
            return False
        if self._has_first_person_marker(text):
            return False
        return True

    def _internal_impulse_20pct(self, agent: AgentState) -> bool:
        if agent.inbox:
            return False
        seed = self.state.tick * 97 + sum(ord(ch) for ch in agent.id) * 31
        return (seed % 100) < 20

    @staticmethod
    def _has_action_or_proposal(text: str) -> bool:
        lowered = str(text).lower()
        markers = (
            "иду",
            "пойду",
            "двигаюсь",
            "проверю",
            "сделаю",
            "осмотрю",
            "вижу",
            "слышу",
            "заметил",
            "нашел",
            "нашёл",
            "чувствую",
            "кажется",
        )
        return any(marker in lowered for marker in markers)

    @staticmethod
    def _has_fact_link(text: str, evidence_ids: list[str]) -> bool:
        if not evidence_ids:
            return False
        lowered = str(text).lower()
        if any(str(eid).lower() in lowered for eid in evidence_ids):
            return True
        fact_markers = ("событ", "видел", "по памяти", "в лог", "по факту", "наблюдал", "отметил")
        return any(marker in lowered for marker in fact_markers)

    def _is_danger_reply_context(self, message: dict[str, Any] | None) -> bool:
        if not message:
            return False
        tags = {str(tag).lower() for tag in message.get("tags", [])}
        if tags & {"panic", "danger", "conflict", "urgent"}:
            return True
        lowered = str(message.get("text", "")).lower()
        danger_markers = ("опасн", "тревог", "пожар", "угроз", "срочн", "конфликт", "шторм")
        return any(marker in lowered for marker in danger_markers)

    @staticmethod
    def _is_user_message_payload(message: dict[str, Any] | None) -> bool:
        if not message:
            return False
        tags = {str(tag).lower() for tag in message.get("tags", [])}
        return "user_message" in tags

    def _looks_like_looping_coordination_question(self, text: str) -> bool:
        normalized = self._normalize_dialogue_text(text)
        patterns = (
            "что нам делать дальше",
            "что делать дальше",
            "какой план дальше",
            "что дальше",
            "что будем делать",
            "как поступим дальше",
        )
        return any(pattern in normalized for pattern in patterns)

    def _reply_substance_ok(
        self,
        *,
        agent: AgentState,
        intent: ActionIntent,
        source_message: dict[str, Any] | None,
    ) -> tuple[bool, str]:
        if intent.kind not in {"say", "message"}:
            return False, "non_text_action"

        reply_text = str(intent.text).strip()
        if not reply_text:
            return False, "empty_text"

        if self._looks_like_looping_coordination_question(reply_text):
            recent = self._recent_dialogue_texts(limit=4, source_id=agent.id)
            if any(self._looks_like_looping_coordination_question(previous) for previous in recent):
                return False, "repetitive_loop"

        min_words = self.reply_min_words_danger if self._is_danger_reply_context(source_message) else self.reply_min_words_default
        if self._word_count(reply_text) < min_words:
            return False, "too_short"

        source_text = ""
        if source_message is not None:
            source_text = str(source_message.get("text", ""))
        topic_text = source_text or str(agent.last_topic or "")
        source_tokens = _tokenize(topic_text)
        reply_tokens = _tokenize(reply_text)
        needed_overlap = 2 if len(source_tokens) >= 5 else 1
        overlap_ok = len(source_tokens & reply_tokens) >= needed_overlap if source_tokens else False

        if overlap_ok:
            return True, "topic_overlap"
        if self._has_action_or_proposal(reply_text):
            return True, "actionable"
        if self._has_fact_link(reply_text, intent.evidence_ids):
            return True, "fact_link"

        return False, "low_substance"

    def _fallback_move_intent(self, agent: AgentState, reply_task: ReplyTask | None = None) -> ActionIntent:
        if reply_task is not None and reply_task.source_id and reply_task.source_id in self.state.agents:
            return ActionIntent(kind="move", target_id=reply_task.source_id)
        friend_id = pick_best_friend(self.state.relations, agent.id, sorted(self.state.agents.keys()))
        if friend_id:
            return ActionIntent(kind="move", target_id=friend_id)
        return ActionIntent(kind="move", destination=pick_wander_target(agent.id, self.state.tick))

    def _build_reply_policy(self, agent: AgentState) -> ReplyPolicy:
        traits = parse_traits(agent.traits)
        skip_bias = 0.0
        if traits.sociability < 45:
            skip_bias += 0.18
        if traits.aggression > 60:
            skip_bias += 0.12
        normalized_traits = agent.traits.lower()
        if "загад" in normalized_traits or "импульс" in normalized_traits:
            skip_bias += 0.1
        if "эмпат" in normalized_traits or "забот" in normalized_traits:
            skip_bias -= 0.08
        skip_bias = _clamp_float(skip_bias, 0.0, 0.55)
        return ReplyPolicy(can_skip=skip_bias >= 0.12, skip_chance=skip_bias)

    def _find_inbox_message(self, agent: AgentState, inbox_id: str) -> dict[str, Any] | None:
        for message in agent.inbox:
            if message.get("inbox_id") == inbox_id:
                return message
        return None

    def _selected_required_inbox_message(
        self,
        agent: AgentState,
        reply_task: ReplyTask | None = None,
    ) -> dict[str, Any] | None:
        if reply_task is not None:
            return self._find_inbox_message(agent, reply_task.inbox_id)
        for message in reversed(agent.inbox):
            if self._is_inbox_message_reply_required(message, mutate=False):
                return message
        return None

    @staticmethod
    def _reply_task_requires_source_message(reply_task: ReplyTask | None) -> bool:
        return bool(
            reply_task is not None
            and reply_task.source_type == "agent"
            and reply_task.source_id
        )

    def _classify_requires_reply(self, source_type: str, tags: list[str]) -> bool:
        tags_set = set(tags)
        if "user_message" in tags_set or "important" in tags_set:
            return True
        if source_type == "agent":
            return True
        if source_type == "world":
            return False
        return "dialogue" in tags_set or "agent_message" in tags_set or "reply" in tags_set

    def _is_inbox_message_reply_required(self, message: dict[str, Any], *, mutate: bool = False) -> bool:
        expects_reply = bool(message.get("expects_reply", message.get("requires_reply")))
        can_reply = bool(message.get("can_reply", True))
        if not expects_reply or not can_reply:
            return False
        source_type = str(message.get("source_type", ""))
        if source_type != "world":
            return True
        received_tick = int(message.get("received_tick", self.state.tick))
        if self.state.tick - received_tick <= self.world_reply_ttl_ticks:
            return True
        if mutate:
            message["expects_reply"] = False
            message["requires_reply"] = False
        return False

    def _inbox_must_reply_count(self, agent: AgentState, *, mutate_expired: bool = False) -> int:
        count = 0
        for message in agent.inbox:
            if self._is_inbox_message_reply_required(message, mutate=mutate_expired):
                count += 1
                continue
            if mutate_expired:
                self._remove_reply_task(message.get("inbox_id"))
        return count

    def _has_pending_must_reply(self, agent: AgentState) -> bool:
        return self._inbox_must_reply_count(agent, mutate_expired=True) > 0

    def _should_apply_inbox_sentiment_on_receive(self, source_type: str, tags: list[str]) -> bool:
        _ = tags
        return source_type == "agent"

    def _apply_inbox_sentiment_once(self, agent: AgentState, message: dict[str, Any], multiplier: int = 3) -> None:
        if bool(message.get("mood_applied")):
            return
        text = str(message.get("text", "")).strip()
        if not text:
            message["mood_applied"] = True
            return
        sentiment = text_sentiment(text)
        agent.mood = _clamp_int(agent.mood + sentiment * multiplier, -100, 100)
        message["mood_applied"] = True

    def _reply_priority(self, source_type: str, tags: list[str]) -> int:
        tags_set = set(tags)
        if "user_message" in tags_set:
            return 120
        if "important" in tags_set:
            return 100
        if source_type == "agent":
            return 90
        if source_type == "world":
            return 75
        return 60

    def _queue_reply_task(self, agent: AgentState, message: dict[str, Any]) -> None:
        inbox_id = str(message.get("inbox_id", "")).strip()
        if not inbox_id or inbox_id in self.reply_task_by_inbox_id:
            return

        can_reply = bool(message.get("can_reply", True))
        if not can_reply:
            return

        tags = list(message.get("tags", []))
        source_type = str(message.get("source_type", ""))
        source_id = message.get("source_id")
        if source_type == "agent":
            if (
                not isinstance(source_id, str)
                or source_id not in self.state.agents
                or source_id == agent.id
            ):
                return
        if not self._is_inbox_message_reply_required(message, mutate=True):
            return

        self.reply_task_seq += 1
        task = ReplyTask(
            id=f"q{self.reply_task_seq}",
            agent_id=agent.id,
            inbox_id=inbox_id,
            source_id=source_id if isinstance(source_id, str) else None,
            source_type=source_type,
            thread_id=message.get("thread_id"),
            expects_reply=bool(message.get("expects_reply", message.get("requires_reply"))),
            can_reply=can_reply,
            text=str(message.get("text", ""))[:220],
            tags=tuple(str(tag) for tag in tags)[:8],
            created_tick=self.state.tick,
            priority=self._reply_priority(source_type=source_type, tags=tags),
            retries=0,
            skips=0,
            last_attempt_tick=self.state.tick,
        )
        self.reply_queue.append(task)
        self.reply_task_by_inbox_id[inbox_id] = task

    def _remove_reply_task(self, inbox_id: str | None) -> None:
        if not inbox_id:
            return
        if inbox_id not in self.reply_task_by_inbox_id:
            return
        self.reply_task_by_inbox_id.pop(inbox_id, None)
        self.reply_queue = deque(
            (task for task in self.reply_queue if task.inbox_id != inbox_id),
            maxlen=self.reply_queue.maxlen,
        )

    def _refresh_reply_queue(self) -> None:
        alive: Deque[ReplyTask] = deque(maxlen=self.reply_queue.maxlen)
        alive_map: dict[str, ReplyTask] = {}
        for task in self.reply_queue:
            agent = self.state.agents.get(task.agent_id)
            if agent is None:
                continue
            message = self._find_inbox_message(agent, task.inbox_id)
            if message is None:
                continue
            if not self._is_inbox_message_reply_required(message, mutate=True):
                continue
            alive.append(task)
            alive_map[task.inbox_id] = task
        self.reply_queue = alive
        self.reply_task_by_inbox_id = alive_map

    def _should_skip_reply(self, agent: AgentState, task: ReplyTask, message: dict[str, Any]) -> bool:
        if task.source_type == "agent":
            return False

        policy = self.reply_policy_by_agent.get(agent.id)
        if policy is None or not policy.can_skip:
            return False

        tags = set(message.get("tags", []))
        if "user_message" in tags:
            return False

        wait_ticks = self.state.tick - task.created_tick
        if wait_ticks >= self.reply_queue_max_wait_ticks:
            return False
        if task.skips >= self.reply_queue_max_skips:
            return False

        source_hash = sum(ord(ch) for ch in (task.source_id or "world"))
        seed = self.state.tick * 41 + sum(ord(ch) for ch in agent.id) * 17 + source_hash + task.retries * 23
        roll = seed % 100
        return roll < int(policy.skip_chance * 100)

    def _select_reply_tasks(self) -> list[ReplyTask]:
        if not self.reply_queue:
            return []

        def rank(task: ReplyTask) -> tuple[int, int, int]:
            wait_ticks = self.state.tick - task.created_tick
            score = task.priority + wait_ticks * 10 - task.retries * 6 - task.skips * 4
            return (score, wait_ticks, -task.retries)

        selected: list[ReplyTask] = []
        used_agents: set[str] = set()

        ordered = sorted(self.reply_queue, key=rank, reverse=True)
        for task in ordered:
            if len(selected) >= self.max_replies_per_tick:
                break
            if task.agent_id in used_agents:
                continue

            agent = self.state.agents.get(task.agent_id)
            if agent is None:
                self._remove_reply_task(task.inbox_id)
                continue
            message = self._find_inbox_message(agent, task.inbox_id)
            if message is None:
                self._remove_reply_task(task.inbox_id)
                continue
            if not self._is_inbox_message_reply_required(message, mutate=True):
                self._remove_reply_task(task.inbox_id)
                continue

            source_id = message.get("source_id")
            source_is_agent = str(message.get("source_type", "")) == "agent"
            can_say = agent.say_cooldown == 0
            can_message = (
                agent.message_cooldown == 0
                and source_is_agent
                and isinstance(source_id, str)
                and source_id in self.state.agents
                and source_id != agent.id
            )
            if source_is_agent:
                if not can_message:
                    continue
            elif not can_say and not can_message:
                continue

            if self._should_skip_reply(agent, task, message):
                task.skips += 1
                task.retries += 1
                task.last_attempt_tick = self.state.tick
                agent.current_plan = "Пропускаю реплику и наблюдаю очередь"
                continue

            selected.append(task)
            used_agents.add(task.agent_id)
            task.last_attempt_tick = self.state.tick

        return selected

    def _select_proactive_agents(self, excluded_agent_ids: set[str], selected_replies_count: int) -> list[str]:
        if not self.llm_decider.enabled:
            return []
        if self.proactive_llm_agents_per_tick <= 0:
            return []

        free_slots = max(0, self.max_replies_per_tick - selected_replies_count)
        if free_slots <= 0:
            return []

        limit = min(self.proactive_llm_agents_per_tick, free_slots)
        if limit <= 0:
            return []

        pending_reply_agents = {task.agent_id for task in self.reply_queue}
        candidates = [
            agent.id
            for agent in self._ordered_agents()
            if (
                agent.id not in excluded_agent_ids
                and agent.id not in pending_reply_agents
                and (agent.say_cooldown == 0 or agent.message_cooldown == 0)
            )
        ]
        if not candidates:
            return []

        offset = self.state.tick % len(candidates)
        rotated = candidates[offset:] + candidates[:offset]
        return rotated[:limit]

    def _dialogue_ratio(self) -> float:
        if not self.dialogue_llm_window:
            return 1.0
        return sum(self.dialogue_llm_window) / len(self.dialogue_llm_window)

    def _llm_stats_payload(self) -> dict[str, Any]:
        return {
            "enabled": self.llm_decider.enabled,
            "llm_first_hard_mode": self.llm_first_hard_mode,
            "force_user_reply_via_llm": self.llm_force_user_reply_via_llm,
            "target_response_ratio": self.target_llm_response_ratio,
            "dialogue_ratio_window": len(self.dialogue_llm_window),
            "dialogue_ratio_llm": round(self._dialogue_ratio(), 3),
            "reply_queue": {
                "pending": len(self.reply_queue),
                "max_replies_per_tick": self.max_replies_per_tick,
                "proactive_agents_per_tick": self.proactive_llm_agents_per_tick,
            },
            "world_events": {
                "enabled": self.world_event_generator.enabled,
                "anchors": list(self.world_event_generator.anchors),
                "last_should_emit_reason": self.world_event_generator.last_should_emit_reason,
                "last_reject_reason": self.world_event_generator.last_reject_reason,
            },
        }

    def _world_event_stats_payload(self) -> dict[str, Any]:
        return self.world_event_generator.metrics(
            tick=self.state.tick,
            events=list(self.state.event_log),
        )

    def _normalize_dialogue_text(self, text: str) -> str:
        lowered = text.lower()
        cleaned = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in lowered)
        return " ".join(cleaned.split())

    def _recent_dialogue_texts(self, limit: int = 12, source_id: str | None = None) -> list[str]:
        items: list[str] = []
        for event in reversed(self.state.event_log):
            if source_id is not None and event.get("source_id") != source_id:
                continue
            tags = set(event.get("tags", []))
            if "dialogue" not in tags and "agent_message" not in tags and "reply" not in tags:
                continue
            text = str(event.get("text", "")).strip()
            if not text:
                continue
            items.append(text)
            if len(items) >= limit:
                break
        return items

    def _is_repetitive_dialogue(self, text: str, recent_texts: list[str]) -> bool:
        candidate = self._normalize_dialogue_text(text)
        if not candidate:
            return True

        candidate_tokens = set(candidate.split())
        candidate_char_len = len(candidate)
        for previous in recent_texts:
            previous_normalized = self._normalize_dialogue_text(previous)
            if not previous_normalized:
                continue
            if candidate == previous_normalized:
                return True

            previous_tokens = set(previous_normalized.split())
            if len(candidate_tokens) < 3 or len(previous_tokens) < 3:
                continue
            if candidate_char_len < 18 or len(previous_normalized) < 18:
                continue
            min_token_count = min(len(candidate_tokens), len(previous_tokens))
            if min_token_count < 4:
                continue
            overlap = len(candidate_tokens & previous_tokens) / max(len(candidate_tokens), len(previous_tokens))
            threshold = 0.97 if min_token_count < 6 else 0.92
            if overlap >= threshold:
                return True
        return False

    def _guess_text_kind(self, tags: list[str], target_id: str | None) -> str | None:
        tags_set = set(tags)
        if "conflict" in tags_set:
            return "conflict"
        if "help" in tags_set:
            return "support"
        if "memory" in tags_set:
            return "memory"
        if "agent_message" in tags_set:
            return "agent_message" if target_id else "respond_agent"
        if "user_message" in tags_set:
            return "respond_user"
        if "dialogue" in tags_set:
            return "explore"
        return None

    def _render_text_variant(self, text_kind: str, selector: int, agent: AgentState, target_id: str | None) -> str:
        target_name = self._agent_name(target_id) or target_id or "друг"
        return templates.render(
            text_kind,
            selector,
            target_name=target_name,
            topic=(agent.last_topic or "текущей ситуации")[:45],
            name=agent.name,
        )

    def _voice_template_kinds(self, agent: AgentState) -> list[str]:
        profile = parse_traits(agent.traits)
        if profile.aggression >= 60:
            return ["voice_aggressive", "voice_cool", "voice_empathic"]
        if profile.sociability >= 55:
            return ["voice_empathic", "voice_cool", "voice_aggressive"]
        return ["voice_cool", "voice_empathic", "voice_aggressive"]

    def _dedupe_dialogue_text(
        self,
        *,
        agent: AgentState,
        text: str,
        tags: list[str],
        target_id: str | None = None,
        text_kind: str | None = None,
    ) -> str:
        candidate = text.strip()
        if not candidate:
            return candidate

        recent = self._recent_dialogue_texts(limit=8, source_id=agent.id) + self._recent_dialogue_texts(limit=12)
        if not self._is_repetitive_dialogue(candidate, recent):
            return candidate

        kind = text_kind or self._guess_text_kind(tags, target_id)
        selector_base = (
            self.state.tick * 113
            + sum(ord(ch) for ch in agent.id) * 19
            + len(candidate) * 7
            + len(recent) * 5
        )
        if kind:
            for offset in range(1, 10):
                variant = self._render_text_variant(
                    text_kind=kind,
                    selector=selector_base + offset,
                    agent=agent,
                    target_id=target_id,
                ).strip()
                if variant and not self._is_repetitive_dialogue(variant, recent):
                    return variant[:360]

        stem = candidate.rstrip(".!? ")
        if not stem:
            stem = candidate
        fallback_variants: list[str] = []
        for offset, voice_kind in enumerate(self._voice_template_kinds(agent), start=1):
            voice_line = templates.render(
                voice_kind,
                selector_base + 50 + offset,
                name=agent.name,
                topic=(agent.last_topic or "текущей ситуации")[:45],
            ).strip()
            if voice_line:
                fallback_variants.append(f"{stem}. {voice_line}")
        fallback_variants.extend(
            [
                f"{stem}. Я до сих пор чувствую тревогу от этого.",
                f"{stem}. Я заметил это и не могу выбросить из головы.",
                f"{stem}. Мне это не нравится, здесь что-то не так.",
                f"{stem}. Странное чувство, будто сейчас случится что-то ещё.",
            ]
        )
        for variant in fallback_variants:
            if not self._is_repetitive_dialogue(variant, recent):
                return variant[:360]
        return candidate[:360]

    def _event_prompt_item(self, event: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "id": event.get("id"),
            "source_id": event.get("source_id"),
            "target_id": event.get("target_id"),
            "text": str(event.get("text", ""))[:160],
            "tags": list(event.get("tags", []))[:4],
            "tick": event.get("tick"),
        }
        if "anchor" in event:
            payload["anchor"] = event.get("anchor")
        if "severity" in event:
            payload["severity"] = event.get("severity")
        if "importance" in event:
            payload["importance"] = event.get("importance")
        if "evidence_ids" in event:
            payload["evidence_ids"] = list(event.get("evidence_ids", []))[:3]
        return payload

    @staticmethod
    def _world_reaction_stance_from_text(text: str) -> str:
        lowered = str(text).lower()
        if any(token in lowered for token in ("опас", "тревог", "жут", "страш", "напряг")):
            return "alarm"
        if any(token in lowered for token in ("не вер", "сомнева", "странно", "подозр")):
            return "skeptic"
        if any(token in lowered for token in ("провер", "осмотр", "след", "детал", "факт")):
            return "practical"
        if any(token in lowered for token in ("чувств", "поддерж", "пережива", "жалко", "эмоци")):
            return "empathy"
        return "curiosity"

    def _world_primary_reactor_count(self, *, importance: float | None, severity: str | None) -> int:
        if str(severity or "").strip().lower() == "danger":
            return 2
        if importance is not None and importance >= 0.8:
            return 2
        return 1

    def _primary_stances_for_event(self, *, event_id: str, count: int) -> list[str]:
        count = max(0, min(count, len(self._WORLD_REACTION_STANCES)))
        if count == 0:
            return []
        base_idx = (sum(ord(ch) for ch in event_id) + self.state.tick * 7) % len(self._WORLD_REACTION_STANCES)
        return [
            self._WORLD_REACTION_STANCES[(base_idx + offset) % len(self._WORLD_REACTION_STANCES)]
            for offset in range(count)
        ]

    def _reply_task_allows_move_instead_of_text(self, reply_task: ReplyTask | None) -> bool:
        if reply_task is None:
            return False
        if reply_task.source_type != "world":
            return False
        if "primary_world_reaction" in set(reply_task.tags):
            return False
        if reply_task.source_id:
            return False
        return "user_message" not in set(reply_task.tags)

    def _safe_json_text(self, value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return repr(value)

    def _truncate_llm_trace_text(self, text: str) -> str:
        if len(text) <= self.llm_trace_max_chars:
            return text
        truncated = len(text) - self.llm_trace_max_chars
        suffix = f"... [truncated {truncated} chars]"
        head_limit = max(0, self.llm_trace_max_chars - len(suffix))
        return f"{text[:head_limit]}{suffix}"

    def _emit_llm_trace_events(self, payload: dict[str, Any]) -> None:
        if not self.llm_trace_events_enabled and not self.llm_trace_logs_enabled:
            return

        tick = payload.get("tick", self.state.tick)
        expected_raw = payload.get("expected_agent_ids")
        expected_agent_ids = list(expected_raw) if isinstance(expected_raw, list | tuple) else []

        prompt_text = self._truncate_llm_trace_text(
            "LLM_PROMPT "
            + self._safe_json_text(
                {
                    "tick": tick,
                    "expected_agent_ids": expected_agent_ids,
                    "prompt": payload.get("prompt"),
                }
            )
        )
        response_text = self._truncate_llm_trace_text(
            "LLM_RESPONSE "
            + self._safe_json_text(
                {
                    "tick": tick,
                    "expected_agent_ids": expected_agent_ids,
                    "response": payload.get("response"),
                }
            )
        )

        if self.llm_trace_logs_enabled:
            trace_logger = logging.getLogger("app.sim.llm_trace")
            trace_logger.warning(prompt_text)
            trace_logger.warning(response_text)

        if self.llm_trace_events_enabled:
            self._append_event(
                source_type="system",
                source_id=None,
                text=prompt_text,
                tags=["system", "llm", "llm_trace", "llm_prompt"],
            )
            self._append_event(
                source_type="system",
                source_id=None,
                text=response_text,
                tags=["system", "llm", "llm_trace", "llm_response"],
            )

    def _recent_events_for_prompt(self, limit: int) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for event in reversed(self.state.event_log):
            tags = set(event.get("tags", []))
            if "llm_trace" in tags:
                continue
            items.append(event)
            if len(items) >= limit:
                break
        items.reverse()
        return items

    def _fallback_memories(self, agent_id: str, limit: int) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for event in reversed(self.state.event_log):
            if event.get("source_id") != agent_id and event.get("target_id") != agent_id:
                continue
            items.append(
                {
                    "event_id": event.get("id"),
                    "text": str(event.get("text", ""))[:140],
                    "tags": list(event.get("tags", []))[:3],
                    "score": 0.2,
                }
            )
            if len(items) >= limit:
                break
        items.reverse()
        return items

    def _memories_for_prompt(self, agent_id: str, query: str, limit: int) -> list[dict[str, Any]]:
        memories = self.memory_store.recall(agent_id=agent_id, query=query, limit=limit)
        if memories:
            return memories
        return self._fallback_memories(agent_id, limit)

    def _agent_context_for_prompt(
        self,
        agent: AgentState,
        reply_task: ReplyTask | None = None,
        proactive_selected: bool = False,
    ) -> dict[str, Any]:
        allowed_actions = ["idle"]
        if agent.say_cooldown == 0:
            allowed_actions.append("say")
        if agent.message_cooldown == 0:
            allowed_actions.append("message")

        inbox_items = [
            {
                "inbox_id": message.get("inbox_id"),
                "thread_id": message.get("thread_id"),
                "target_id": message.get("target_id"),
                "source_id": message.get("source_id"),
                "source_type": message.get("source_type"),
                "text": str(message.get("text", ""))[:140],
                "tags": list(message.get("tags", []))[:3],
                "expects_reply": bool(message.get("expects_reply", message.get("requires_reply"))),
                "can_reply": bool(message.get("can_reply", True)),
                "directed": bool(message.get("directed", False)),
                "world_event_id": message.get("world_event_id"),
                "world_reaction_role": message.get("world_reaction_role"),
                "world_reaction_stance": message.get("world_reaction_stance"),
                "world_reaction_forbidden_stances": list(message.get("world_reaction_forbidden_stances", []))[:4],
                "world_reaction_primary_agents": list(message.get("world_reaction_primary_agents", []))[:2],
                "no_question_required": bool(message.get("no_question_required", False)),
            }
            for message in agent.inbox[-self.prompt_inbox_limit :]
        ]

        selected_message = self._selected_required_inbox_message(agent, reply_task=reply_task)
        answer_first = bool(
            selected_message is not None and self._text_has_question(str(selected_message.get("text", "")))
        )
        must_message_source = bool(
            self._reply_task_requires_source_message(reply_task) and "message" in allowed_actions
        )

        query = ""
        if reply_task is not None:
            query = reply_task.text
        elif inbox_items:
            query = str(inbox_items[-1].get("text", ""))
        elif agent.last_topic:
            query = agent.last_topic
        else:
            query = agent.current_plan

        others: list[dict[str, Any]] = []
        for other in self._ordered_agents():
            if other.id == agent.id:
                continue
            others.append(
                {
                    "id": other.id,
                    "name": other.name,
                    "distance": round(self._distance_2d(agent.pos, other.pos), 2),
                    "relation": self.state.relations.get((agent.id, other.id), 0),
                }
            )
        others.sort(key=lambda item: item["distance"])

        queue_payload: dict[str, Any] = {
            "selected_for_reply": reply_task is not None,
            "proactive_selected": proactive_selected,
            "pending_inbox_count": len(agent.inbox),
            "pending_must_reply_count": self._inbox_must_reply_count(agent, mutate_expired=False),
            "question_allowed_now": self._can_ask_question_now(agent),
            "question_interval_ticks": self._question_interval_for_agent(agent.id),
            "answer_first": answer_first,
            "internal_impulse_20pct": self._internal_impulse_20pct(agent),
        }
        first_world_reaction = bool(
            selected_message is not None
            and str(selected_message.get("source_type", "")).lower() == "world"
            and str(selected_message.get("world_reaction_role", "")) == "primary"
        )
        if first_world_reaction:
            queue_payload["first_world_reaction"] = True
            queue_payload["world_reaction"] = {
                "event_id": selected_message.get("world_event_id"),
                "role": selected_message.get("world_reaction_role"),
                "assigned_stance": selected_message.get("world_reaction_stance"),
                "forbidden_stances": list(selected_message.get("world_reaction_forbidden_stances", []))[:4],
                "primary_reactors": list(selected_message.get("world_reaction_primary_agents", []))[:2],
            }
        if reply_task is not None:
            queue_payload.update(
                {
                    "task_id": reply_task.id,
                    "inbox_id": reply_task.inbox_id,
                    "thread_id": reply_task.thread_id,
                    "source_id": reply_task.source_id,
                    "source_type": reply_task.source_type,
                    "tags": list(reply_task.tags)[:4],
                    "text": reply_task.text[:180],
                    "expects_reply": reply_task.expects_reply,
                    "can_reply": reply_task.can_reply,
                    "wait_ticks": self.state.tick - reply_task.created_tick,
                    "retries": reply_task.retries,
                    "skips": reply_task.skips,
                    "allow_move_instead_of_say": self._reply_task_allows_move_instead_of_text(reply_task),
                    "must_message_source": must_message_source,
                    "reply_policy": {
                        "can_skip": self.reply_policy_by_agent.get(agent.id, ReplyPolicy(False, 0.0)).can_skip,
                        "max_skips": self.reply_queue_max_skips,
                    },
                }
            )

        return {
            "agent_id": agent.id,
            "traits": agent.traits,
            "mood": agent.mood,
            "mood_label": agent.mood_label,
            "state": {
                "mood": agent.mood,
                "mood_label": agent.mood_label,
                "plan": agent.current_plan,
                "pos": {"x": round(agent.pos.x, 2), "z": round(agent.pos.z, 2)},
                "cooldowns": {
                    "say": agent.say_cooldown,
                    "message": agent.message_cooldown,
                },
                "last_action": agent.last_action,
                "last_say": (agent.last_say or "")[:140],
            },
            "allowed_actions": allowed_actions,
            "recent_events": [
                self._event_prompt_item(event)
                for event in self._recent_events_for_prompt(self.prompt_recent_events_limit)
            ],
            "inbox": inbox_items,
            "memories_top": self._memories_for_prompt(agent.id, query=query, limit=self.prompt_memories_limit),
            "others": others,
            "queue": queue_payload,
        }

    def _world_summary_for_prompt(self, selected_reply_agent_ids: list[str] | None = None) -> dict[str, Any]:
        return {
            "tick": self.state.tick,
            "agents_count": len(self.state.agents),
            "llm_response_ratio": round(self._dialogue_ratio(), 3),
            "reply_queue": {
                "pending": len(self.reply_queue),
                "selected_agent_ids": selected_reply_agent_ids or [],
                "max_replies_per_tick": self.max_replies_per_tick,
                "max_wait_ticks": self.reply_queue_max_wait_ticks,
            },
            "recent_events": [
                self._event_prompt_item(event)
                for event in self._recent_events_for_prompt(self.prompt_recent_events_limit)
            ],
        }

    def _llm_debug(self, message: str) -> None:
        if self.llm_decider.enabled and self.llm_decider.client.debug:
            logging.getLogger("app.sim.engine").warning(message)

    def _llm_decision_for_single_agent(
        self,
        agent_id: str,
        retries: int = 0,
        reply_task: ReplyTask | None = None,
        proactive_selected: bool = False,
    ) -> AgentDecision | None:
        if not self.llm_decider.enabled:
            return None

        agent = self.state.agents.get(agent_id)
        if agent is None:
            return None

        attempts = max(1, retries + 1)
        for _ in range(attempts):
            decisions = self.llm_decider.decide(
                tick=self.state.tick,
                world_summary=self._world_summary_for_prompt(
                    selected_reply_agent_ids=[agent_id],
                ),
                agents_context=[
                    self._agent_context_for_prompt(
                        agent,
                        reply_task=reply_task,
                        proactive_selected=proactive_selected,
                    )
                ],
                expected_agent_ids=[agent_id],
            )
            single = decisions.get(agent_id)
            if single is not None:
                return single
        return None

    def _llm_decisions_for_selected_agents(
        self,
        reply_tasks: list[ReplyTask],
        proactive_agent_ids: list[str],
    ) -> dict[str, AgentDecision]:
        if not self.llm_decider.enabled:
            return {}

        reply_task_by_agent = {task.agent_id: task for task in reply_tasks}
        expected_agent_ids = [task.agent_id for task in reply_tasks]
        for agent_id in proactive_agent_ids:
            if agent_id not in reply_task_by_agent:
                expected_agent_ids.append(agent_id)
        if not expected_agent_ids:
            return {}

        context_by_agent: dict[str, dict[str, Any]] = {}
        proactive_set = set(proactive_agent_ids)
        for agent_id in expected_agent_ids:
            agent = self.state.agents.get(agent_id)
            if agent is None:
                continue
            context_by_agent[agent_id] = self._agent_context_for_prompt(
                agent,
                reply_task=reply_task_by_agent.get(agent_id),
                proactive_selected=agent_id in proactive_set,
            )
        expected_agent_ids = [agent_id for agent_id in expected_agent_ids if agent_id in context_by_agent]
        if not expected_agent_ids:
            return {}

        decisions = self.llm_decider.decide(
            tick=self.state.tick,
            world_summary=self._world_summary_for_prompt(selected_reply_agent_ids=expected_agent_ids),
            agents_context=[context_by_agent[agent_id] for agent_id in expected_agent_ids],
            expected_agent_ids=expected_agent_ids,
        )

        if len(decisions) == len(expected_agent_ids):
            return decisions

        missing = [agent_id for agent_id in expected_agent_ids if agent_id not in decisions]
        self._llm_debug(
            "LLM batch coverage "
            f"{len(decisions)}/{len(expected_agent_ids)} missing={missing}"
        )

        for agent_id in missing:
            single = self._llm_decision_for_single_agent(
                agent_id,
                retries=self.single_agent_backfill_retries,
                reply_task=reply_task_by_agent.get(agent_id),
                proactive_selected=agent_id in proactive_set,
            )
            if single is not None:
                decisions[agent_id] = single

        still_missing = [agent_id for agent_id in expected_agent_ids if agent_id not in decisions]
        self._llm_debug(
            "LLM coverage after backfill "
            f"{len(decisions)}/{len(expected_agent_ids)} missing={still_missing}"
        )
        return decisions

    def _build_agents(self) -> dict[str, AgentState]:
        base = [
            ("a1", "Скебоб", "эмпатичный, любопытный", "#f4a261", -10),
            ("a2", "Бобекс", "загадочный, решительный", "#2a9d8f", 5),
            ("a3", "Боб", "злой, импульсивный", "#e76f51", -20),
            ("a4", "Скебобиха", "кокетливая, заботливая", "#457b9d", 18),
        ]

        agents: dict[str, AgentState] = {}
        for idx, (agent_id, name, traits, avatar, mood) in enumerate(base):
            agent = AgentState(
                id=agent_id,
                name=name,
                traits=traits,
                mood=mood,
                avatar=avatar,
                pos=Vec3(x=-8 + idx * 5, z=-3 + idx),
                current_plan="Синхронизация мира",
                last_action="idle",
                last_interaction_tick=0,
            )
            agent.current_plan = plan_for_mood(agent.mood_label, selector=idx)
            agents[agent_id] = agent
        return agents

    def _build_relations(self, ids: list[str]) -> dict[tuple[str, str], int]:
        relations: dict[tuple[str, str], int] = {}
        for src_idx, src in enumerate(ids):
            for dst_idx, dst in enumerate(ids):
                if src == dst:
                    continue
                base = 30 - abs(src_idx - dst_idx) * 12 + (dst_idx - src_idx) * 3
                relations[(src, dst)] = clamp_relation(base)
        return relations

    @staticmethod
    def _fallback_avatar(agent_id: str) -> str:
        palette = ("#f4a261", "#2a9d8f", "#e76f51", "#457b9d", "#8ecae6", "#f77f00")
        idx = abs(sum(ord(ch) for ch in agent_id)) % len(palette)
        return palette[idx]

    def _next_agent_id(self) -> str:
        occupied = {agent_id for agent_id in self.state.agents}
        idx = 1
        while True:
            candidate = f"a{idx}"
            if candidate not in occupied:
                return candidate
            idx += 1

    def _suggest_agent_position(self) -> Vec3:
        if not self.state.agents:
            return Vec3(0.0, 0.0, 0.0)
        ordered = self._ordered_agents()
        pivot = ordered[-1].pos
        offset_seed = (self.state.tick + len(ordered) * 37) % 6
        offsets = (
            (2.6, 0.0),
            (-2.6, 0.0),
            (0.0, 2.4),
            (0.0, -2.4),
            (1.8, 1.8),
            (-1.8, -1.8),
        )
        dx, dz = offsets[offset_seed]
        return clamp_position(Vec3(pivot.x + dx, 0.0, pivot.z + dz))

    def add_agent(
        self,
        *,
        name: str,
        traits: str,
        mood: int = 0,
        avatar: str | None = None,
        agent_id: str | None = None,
        pos_x: float | None = None,
        pos_z: float | None = None,
    ) -> AgentState:
        normalized_id = (agent_id or "").strip()
        if not normalized_id:
            normalized_id = self._next_agent_id()
        if normalized_id in self.state.agents:
            raise ValueError(f"agent already exists: {normalized_id}")

        normalized_name = name.strip()[:64] or normalized_id
        normalized_traits = traits.strip()[:256] or "нейтральный"
        normalized_mood = _clamp_int(int(mood), -100, 100)
        normalized_avatar = (avatar or "").strip() or self._fallback_avatar(normalized_id)

        if pos_x is None or pos_z is None:
            suggested = self._suggest_agent_position()
            initial_pos = suggested
        else:
            initial_pos = clamp_position(Vec3(float(pos_x), 0.0, float(pos_z)))
        initial_pos = self._separate_from_other_agents(normalized_id, initial_pos)

        new_agent = AgentState(
            id=normalized_id,
            name=normalized_name,
            traits=normalized_traits,
            mood=normalized_mood,
            avatar=normalized_avatar,
            pos=initial_pos,
            current_plan="наблюдать мир",
            last_action="idle",
            last_interaction_tick=self.state.tick,
        )
        new_agent.current_plan = plan_for_mood(new_agent.mood_label, selector=len(self.state.agents))
        self.state.agents[normalized_id] = new_agent
        self.agent_home_positions[normalized_id] = Vec3(initial_pos.x, 0.0, initial_pos.z)

        for other_id, other in self.state.agents.items():
            if other_id == normalized_id:
                continue
            base_forward = clamp_relation(12 - abs(new_agent.mood - other.mood) // 12)
            base_backward = clamp_relation(10 - abs(other.mood - new_agent.mood) // 14)
            self.state.relations[(normalized_id, other_id)] = base_forward
            self.state.relations[(other_id, normalized_id)] = base_backward

        self._append_event(
            source_type="system",
            source_id=None,
            text=f"В мир добавлен агент {new_agent.name}.",
            tags=["system", "world", "agent_lifecycle"],
        )
        return new_agent

    def remove_agent(self, agent_id: str) -> AgentState:
        normalized_id = str(agent_id).strip()
        if normalized_id not in self.state.agents:
            raise KeyError(normalized_id)
        if len(self.state.agents) <= 1:
            raise ValueError("cannot remove the last agent")

        removed = self.state.agents.pop(normalized_id)
        self.agent_home_positions.pop(normalized_id, None)

        self.state.relations = {
            key: value
            for key, value in self.state.relations.items()
            if normalized_id not in key
        }

        self.reply_queue = deque(
            (
                task
                for task in self.reply_queue
                if task.agent_id != normalized_id and task.source_id != normalized_id
            ),
            maxlen=self.reply_queue.maxlen,
        )
        self.reply_task_by_inbox_id = {
            inbox_id: task
            for inbox_id, task in self.reply_task_by_inbox_id.items()
            if task.agent_id != normalized_id and task.source_id != normalized_id
        }

        for agent in self.state.agents.values():
            agent.inbox = [
                item
                for item in agent.inbox
                if item.get("source_id") != normalized_id and item.get("target_id") != normalized_id
            ]
            if agent.target_id == normalized_id:
                agent.target_id = None

        self._append_event(
            source_type="system",
            source_id=None,
            text=f"Агент {removed.name} покинул мир.",
            tags=["system", "world", "agent_lifecycle"],
        )
        return removed

    def _seed_initial_events(self) -> None:
        self._append_event(
            source_type="world",
            source_id=None,
            text="Загрузка мира завершена. LLM-first мозг активен.",
            tags=["system", "world"],
        )
        self._append_event(
            source_type="world",
            source_id=None,
            text="Приоритет: ответы через LLM; fallback только для отказоустойчивости.",
            tags=["system", "world"],
        )

    def _startup_world_event_text(self) -> str:
        topics = (
            "Сегодня был сильный ливень.",
            "Сегодня хороший день.",
            "Сегодня холодно.",
            "Сегодня жарко.",
            "Сегодня ничего не произошло.",
        )
        return topics[random.randrange(len(topics))]

    def _trigger_startup_world_event(self) -> None:
        if not self.startup_world_event_enabled:
            return
        if not self.state.agents:
            return

        generated_startup = self.world_event_generator.generate(
            tick=self.state.tick,
            events=list(self.state.event_log),
            agents=self._world_event_agents_snapshot(),
            reply_queue_pending=len(self.reply_queue),
        )
        if generated_startup is not None:
            self.add_world_event(
                text=generated_startup.text,
                importance=generated_startup.importance,
                emit_immediate_reactions=False,
                extra_tags=generated_startup.tags,
                anchor=generated_startup.anchor,
                severity=generated_startup.severity,
                ensure_agent_reaction=False,
            )
            self.world_events_logger.warning(
                "startup_world_event_emit tick=%s anchor=%s severity=%s importance=%.2f text=%s",
                self.state.tick,
                generated_startup.anchor,
                generated_startup.severity,
                generated_startup.importance,
                generated_startup.text,
            )
            return

        self.add_world_event(
            text=self._startup_world_event_text(),
            importance=self.startup_world_event_importance,
            emit_immediate_reactions=False,
        )

    def _remember_event(self, event_id: str) -> None:
        for agent in self.state.agents.values():
            agent.memory_short.append(event_id)
            if len(agent.memory_short) > self.memory_short_limit:
                agent.memory_short.pop(0)

    def _remember_event_in_store(self, event: dict[str, Any]) -> None:
        if not self.memory_store.enabled:
            return

        source_id = event.get("source_id")
        target_id = event.get("target_id")
        source_type = event.get("source_type")
        text = str(event.get("text", ""))
        tags = list(event.get("tags", []))

        participants: set[str] = set()
        if source_id in self.state.agents:
            participants.add(source_id)
        if target_id in self.state.agents:
            participants.add(target_id)

        if source_type == "world":
            if target_id and target_id in self.state.agents:
                participants.add(target_id)
            else:
                participants.update(self.state.agents.keys())

        if not participants:
            return

        importance = 0.45
        if isinstance(event.get("importance"), (int, float)):
            importance = _clamp_float(float(event.get("importance", 0.45)), 0.0, 1.0)
        else:
            if "important" in tags:
                importance += 0.3
            if "user_message" in tags:
                importance += 0.2
            importance = _clamp_float(importance, 0.0, 1.0)

        for agent_id in participants:
            self.memory_store.remember(
                agent_id=agent_id,
                text=text,
                tags=tags,
                tick=self.state.tick,
                event_id=event.get("id"),
                source_id=source_id,
                target_id=target_id,
                importance=importance,
            )

    def _append_event(
        self,
        source_type: str,
        source_id: str | None,
        text: str,
        tags: list[str],
        target_id: str | None = None,
        llm_generated: bool = False,
        thread_id: str | None = None,
        expects_reply: bool | None = None,
        can_reply: bool | None = None,
        importance: float | None = None,
        anchor: str | None = None,
        severity: str | None = None,
        evidence_ids: list[str] | None = None,
    ) -> dict:
        self.state.next_event_id += 1
        normalized_tags = list(tags)

        event = {
            "id": f"e{self.state.next_event_id}",
            "ts": _utc_iso(),
            "source_type": source_type,
            "source_id": source_id,
            "text": text,
            "tags": normalized_tags,
            "tick": self.state.tick,
        }
        if target_id is not None:
            event["target_id"] = target_id
        effective_thread_id = (thread_id or "").strip()
        if not effective_thread_id and target_id is not None:
            effective_thread_id = f"t{self.state.next_event_id}"
        if effective_thread_id:
            event["thread_id"] = effective_thread_id
        if expects_reply is not None:
            event["expects_reply"] = bool(expects_reply)
        if can_reply is not None:
            event["can_reply"] = bool(can_reply)
        if importance is not None:
            event["importance"] = _clamp_float(float(importance), 0.0, 1.0)
        if anchor:
            event["anchor"] = str(anchor)
        if severity:
            event["severity"] = str(severity)
        if evidence_ids:
            normalized_evidence_ids: list[str] = []
            seen: set[str] = set()
            for item in evidence_ids:
                if not isinstance(item, str):
                    continue
                cleaned = item.strip()
                if not cleaned:
                    continue
                lowered = cleaned.casefold()
                if lowered in seen:
                    continue
                seen.add(lowered)
                normalized_evidence_ids.append(cleaned[:64])
                if len(normalized_evidence_ids) >= 3:
                    break
            if normalized_evidence_ids:
                event["evidence_ids"] = normalized_evidence_ids

        self.state.event_log.append(event)
        self._remember_event(event["id"])
        self._remember_event_in_store(event)
        if source_type == "world":
            self.world_event_generator.observe_world_event(event)

        if source_type == "agent" and ({"dialogue", "reply", "agent_message"} & set(normalized_tags)):
            is_llm = llm_generated or ("llm" in normalized_tags)
            self.dialogue_llm_window.append(1 if is_llm else 0)

        return event

    def _enqueue_inbox(
        self,
        target_id: str,
        source_type: str,
        source_id: str | None,
        text: str,
        tags: list[str],
        thread_id: str | None = None,
        expects_reply: bool | None = None,
        can_reply: bool | None = None,
        directed: bool = False,
    ) -> dict[str, Any] | None:
        target = self.state.agents.get(target_id)
        if not target:
            return None
        source_name = self._agent_name(source_id) if source_id else None
        tags_list = list(tags)
        effective_expects_reply = (
            self._classify_requires_reply(source_type=source_type, tags=tags_list)
            if expects_reply is None
            else bool(expects_reply)
        )
        effective_can_reply = effective_expects_reply if can_reply is None else bool(can_reply)
        if not effective_expects_reply:
            effective_can_reply = False
        apply_sentiment_on_receive = self._should_apply_inbox_sentiment_on_receive(source_type=source_type, tags=tags_list)
        self.inbox_seq += 1
        inbox_message = {
            "inbox_id": f"i{self.inbox_seq}",
            "thread_id": (thread_id or "").strip() or None,
            "target_id": target_id,
            "source_type": source_type,
            "source_id": source_id,
            "source_name": source_name,
            "text": text,
            "tags": tags_list,
            "received_tick": self.state.tick,
            "penalized": False,
            "directed": directed,
            "expects_reply": effective_expects_reply,
            "requires_reply": effective_expects_reply,
            "can_reply": effective_can_reply,
            "mood_applied": not apply_sentiment_on_receive,
        }
        target.inbox.append(inbox_message)
        if apply_sentiment_on_receive:
            self._apply_inbox_sentiment_once(target, inbox_message, multiplier=3)
        self._queue_reply_task(target, inbox_message)
        return inbox_message

    def _decrement_cooldowns(self) -> None:
        for agent in self._ordered_agents():
            agent.say_cooldown = max(0, agent.say_cooldown - 1)
            agent.message_cooldown = max(0, agent.message_cooldown - 1)

    def _apply_passive_mood(self, agent: AgentState) -> None:
        if not agent.inbox and (self.state.tick - agent.last_interaction_tick) >= 4:
            agent.mood = _clamp_int(agent.mood - 1, -100, 100)
        if self.state.tick % 8 == 0:
            if agent.mood > 0:
                agent.mood -= 1
            elif agent.mood < 0:
                agent.mood += 1

    def _execute_move(self, agent: AgentState, intent: ActionIntent) -> None:
        _ = intent
        destination = self._chaotic_destination(agent)

        old_pos = agent.pos
        max_step = 0.6 + 0.45 * max(0.5, self.state.speed)
        new_pos = step_towards(agent.pos, destination, max_step=max_step)
        new_pos = self._clamp_to_home_radius(agent.id, new_pos)
        new_pos = self._separate_from_other_agents(agent.id, new_pos)
        agent.pos = new_pos

        dx = new_pos.x - old_pos.x
        dz = new_pos.z - old_pos.z
        if abs(dx) > 1e-6 or abs(dz) > 1e-6:
            agent.look_at = Vec3(dx, 0.0, dz)
        agent.last_action = "move"
        agent.target_id = intent.target_id

    def _apply_background_movement(self, agent: AgentState) -> None:
        if agent.last_action == "move":
            return

        destination = self._chaotic_destination(agent)
        old_pos = agent.pos
        max_step = max(0.05, self.chaotic_background_move_step * max(0.5, self.state.speed))
        new_pos = step_towards(agent.pos, destination, max_step=max_step)
        new_pos = self._clamp_to_home_radius(agent.id, new_pos)
        new_pos = self._separate_from_other_agents(agent.id, new_pos)
        agent.pos = new_pos

        dx = new_pos.x - old_pos.x
        dz = new_pos.z - old_pos.z
        if abs(dx) > 1e-6 or abs(dz) > 1e-6:
            agent.look_at = Vec3(dx, 0.0, dz)

    def _execute_say(self, agent: AgentState, intent: ActionIntent) -> dict | None:
        if agent.say_cooldown > 0:
            agent.last_action = "idle"
            return None

        text = intent.text.strip() if intent.text else templates.render("explore", self.state.tick)
        tags = intent.tags if intent.tags else ["dialogue"]
        text = self._dedupe_dialogue_text(
            agent=agent,
            text=text,
            tags=tags,
            target_id=intent.target_id,
            text_kind=intent.text_kind,
        )
        text = self._enforce_question_policy_text(
            agent,
            text,
            force_non_question=intent.force_non_question,
        )
        directed = intent.target_id is not None
        expects_reply = (
            bool(intent.expects_reply)
            if intent.expects_reply is not None
            else directed
        )
        event = self._append_event(
            source_type="agent",
            source_id=agent.id,
            text=text,
            tags=tags,
            target_id=intent.target_id,
            llm_generated=intent.llm_generated,
            thread_id=intent.thread_id,
            expects_reply=expects_reply if directed else None,
            can_reply=True if directed else None,
            evidence_ids=list(intent.evidence_ids or []),
        )

        agent.last_action = "say"
        agent.last_say = text
        agent.say_cooldown = 2
        agent.last_interaction_tick = self.state.tick
        agent.target_id = intent.target_id
        self._mark_question_emitted(agent, text)

        if intent.target_id:
            self._enqueue_inbox(
                target_id=intent.target_id,
                source_type="agent",
                source_id=agent.id,
                text=text,
                tags=tags,
                thread_id=event.get("thread_id"),
                expects_reply=expects_reply,
                can_reply=True,
                directed=True,
            )
        update_relations_from_event(self.state.relations, event)
        return event

    def _execute_message(self, agent: AgentState, intent: ActionIntent) -> dict | None:
        if not intent.target_id or intent.target_id not in self.state.agents:
            agent.last_action = "idle"
            return None
        if agent.message_cooldown > 0:
            agent.last_action = "idle"
            return None

        target_name = self._agent_name(intent.target_id) or intent.target_id
        text = (
            intent.text.strip()
            if intent.text
            else templates.render(
                "agent_message",
                self.state.tick,
                target_name=target_name,
                topic=agent.last_topic or "текущей ситуации",
            )
        )
        tags = intent.tags if intent.tags else ["agent_message", "dialogue"]
        text = self._dedupe_dialogue_text(
            agent=agent,
            text=text,
            tags=tags,
            target_id=intent.target_id,
            text_kind=intent.text_kind,
        )
        text = self._enforce_question_policy_text(
            agent,
            text,
            force_non_question=intent.force_non_question,
        )
        expects_reply = bool(intent.expects_reply) if intent.expects_reply is not None else True
        event = self._append_event(
            source_type="agent",
            source_id=agent.id,
            text=text,
            tags=tags,
            target_id=intent.target_id,
            llm_generated=intent.llm_generated,
            thread_id=intent.thread_id,
            expects_reply=expects_reply,
            can_reply=True,
            evidence_ids=list(intent.evidence_ids or []),
        )
        self._enqueue_inbox(
            target_id=intent.target_id,
            source_type="agent",
            source_id=agent.id,
            text=text,
            tags=tags,
            thread_id=event.get("thread_id"),
            expects_reply=expects_reply,
            can_reply=True,
            directed=True,
        )

        agent.last_action = "say"
        agent.last_say = text
        agent.target_id = intent.target_id
        agent.say_cooldown = max(agent.say_cooldown, 1)
        agent.message_cooldown = 3
        agent.last_interaction_tick = self.state.tick
        self._mark_question_emitted(agent, text)
        update_relations_from_event(self.state.relations, event)
        return event

    def _execute_intent(self, agent: AgentState, intent: ActionIntent) -> dict | None:
        if intent.kind == "move":
            self._execute_move(agent, intent)
            return None
        if intent.kind == "say":
            return self._execute_say(agent, intent)
        if intent.kind == "message":
            return self._execute_message(agent, intent)
        agent.last_action = "idle"
        agent.target_id = None
        return None

    def _intent_from_llm_decision(self, agent: AgentState, decision: AgentDecision) -> ActionIntent | None:
        if decision.act == "idle":
            return ActionIntent(kind="idle", llm_generated=True)

        if decision.act == "move":
            # Перемещение из LLM намеренно игнорируется: движение автономное и хаотичное.
            return None

        if decision.act == "say":
            if agent.say_cooldown > 0:
                return ActionIntent(kind="idle", llm_generated=True)
            text = (decision.say_text or "").strip()
            if not text:
                return ActionIntent(kind="idle", llm_generated=True)
            if self._contains_operational_tone(text):
                return None
            if self._contains_summon_language(text):
                return None
            if self._is_third_person_self_reference(agent, text):
                return None

            target_id: str | None = None
            if (
                decision.target_id
                and decision.target_id in self.state.agents
                and decision.target_id != agent.id
            ):
                target_id = decision.target_id

            return ActionIntent(
                kind="say",
                text=text,
                tags=["dialogue", "llm"],
                target_id=target_id,
                llm_generated=True,
                speech_intent=decision.speech_intent,
                evidence_ids=list(decision.evidence_ids or []),
            )

        if decision.act == "message":
            if agent.message_cooldown > 0:
                return ActionIntent(kind="idle", llm_generated=True)
            target_id = decision.target_id
            text = (decision.say_text or "").strip()
            if not text:
                return ActionIntent(kind="idle", llm_generated=True)
            if self._contains_operational_tone(text):
                return None
            if self._contains_summon_language(text):
                return None
            if self._is_third_person_self_reference(agent, text):
                return None

            if not target_id or target_id not in self.state.agents or target_id == agent.id:
                return ActionIntent(kind="idle", llm_generated=True)
            return ActionIntent(
                kind="message",
                text=text,
                tags=["agent_message", "dialogue", "llm"],
                target_id=target_id,
                llm_generated=True,
                speech_intent=decision.speech_intent,
                evidence_ids=list(decision.evidence_ids or []),
            )

        return None

    def _apply_llm_deltas(self, agent: AgentState, decision: AgentDecision) -> bool:
        if not decision.deltas:
            return False

        mood_delta = _clamp_int(decision.deltas.self_mood, -6, 6)
        if mood_delta != 0:
            agent.mood = _clamp_int(agent.mood + mood_delta, -100, 100)

        relations_changed = False
        for rel_delta in decision.deltas.relations[:3]:
            if rel_delta.to_id == agent.id or rel_delta.to_id not in self.state.agents:
                continue
            key = (agent.id, rel_delta.to_id)
            if key not in self.state.relations:
                continue
            delta = _clamp_int(rel_delta.delta, -3, 3)
            if delta == 0:
                continue
            self.state.relations[key] = clamp_relation(self.state.relations[key] + delta)
            relations_changed = True
        return relations_changed

    def _intent_satisfies_must_answer(
        self,
        intent: ActionIntent,
        allow_action_reply: bool,
        reply_task: ReplyTask | None = None,
        answer_first_required: bool = False,
    ) -> bool:
        if self._reply_task_requires_source_message(reply_task):
            if intent.kind != "message":
                return False
            if intent.target_id != reply_task.source_id:
                return False
            if answer_first_required and self._text_has_question(intent.text):
                return False
            return True

        if intent.kind in {"say", "message"}:
            if answer_first_required and self._text_has_question(intent.text):
                return False
            return True
        if allow_action_reply and intent.kind == "move":
            return True
        return False

    def _consume_inbox(
        self,
        agent: AgentState,
        intent: ActionIntent,
        inbox_id: str | None = None,
        allow_nonverbal: bool = False,
    ) -> None:
        if not agent.inbox:
            return
        if intent.kind not in {"say", "message"} and not allow_nonverbal:
            return

        index = -1
        if inbox_id:
            for idx, message in enumerate(agent.inbox):
                if message.get("inbox_id") == inbox_id:
                    index = idx
                    break

        if index < 0 and intent.kind == "message" and intent.target_id:
            for idx in range(len(agent.inbox) - 1, -1, -1):
                source_id = agent.inbox[idx].get("source_id")
                if source_id and source_id == intent.target_id:
                    index = idx
                    break
            if index < 0:
                return

        if index < 0:
            for idx in range(len(agent.inbox) - 1, -1, -1):
                if self._is_inbox_message_reply_required(agent.inbox[idx], mutate=False):
                    index = idx
                    break

        if index < 0:
            index = len(agent.inbox) - 1

        processed = agent.inbox.pop(index)
        self._apply_inbox_sentiment_once(agent, processed, multiplier=3)

        self._remove_reply_task(processed.get("inbox_id"))
        agent.last_topic = str(processed.get("text", ""))[:60]
        agent.last_interaction_tick = self.state.tick

    def _fallback_response_intent(self, agent: AgentState, reply_task: ReplyTask | None = None) -> ActionIntent:
        latest_message = self._selected_required_inbox_message(agent, reply_task=reply_task) or {}
        if not latest_message and agent.inbox:
            latest_message = agent.inbox[-1]

        thread_id = str(latest_message.get("thread_id") or "").strip() or None
        answer_first_required = self._text_has_question(str(latest_message.get("text", ""))) or bool(
            latest_message.get("no_question_required")
        )
        source_id = latest_message.get("source_id")
        source_type = str(latest_message.get("source_type", ""))
        source_tags = {str(tag).lower() for tag in latest_message.get("tags", [])}
        is_user_message = "user_message" in source_tags
        is_world_message = source_type == "world" and not is_user_message
        world_event_id = str(latest_message.get("world_event_id") or "").strip() or None
        source_name = latest_message.get("source_name") or self._agent_name(source_id) or "друг"
        topic = str(latest_message.get("text", ""))[:48]
        agent.last_topic = topic

        selector = self.state.tick * 109 + sum(ord(ch) for ch in agent.id)
        if (
            source_type == "agent"
            and source_id
            and source_id in self.state.agents
            and agent.message_cooldown == 0
        ):
            text = templates.render("respond_agent", selector, target_name=source_name, topic=topic)
            return ActionIntent(
                kind="message",
                text=text,
                tags=["agent_message", "dialogue", "reply"],
                target_id=source_id,
                text_kind="respond_agent",
                llm_generated=False,
                thread_id=thread_id,
                expects_reply=True,
                force_non_question=answer_first_required,
                evidence_ids=[world_event_id] if world_event_id else [],
            )

        text = templates.render("respond_user", selector)
        tags = ["dialogue", "reply"]
        if is_user_message:
            tags.append("user_message")
        if is_world_message:
            tags.append("world_reply")
        return ActionIntent(
            kind="say",
            text=text,
            tags=tags,
            text_kind="respond_user",
            llm_generated=False,
            thread_id=thread_id,
            expects_reply=False,
            force_non_question=answer_first_required,
            evidence_ids=[world_event_id] if world_event_id else [],
        )

    def _fallback_intent(
        self,
        agent: AgentState,
        must_answer: bool,
        reply_task: ReplyTask | None = None,
        allow_action_reply: bool = False,
    ) -> ActionIntent:
        if must_answer and agent.inbox and not allow_action_reply:
            return self._fallback_response_intent(agent, reply_task=reply_task)
        if must_answer and allow_action_reply:
            friend_id = pick_best_friend(self.state.relations, agent.id, sorted(self.state.agents.keys()))
            if friend_id:
                return ActionIntent(kind="move", target_id=friend_id)
            return ActionIntent(kind="move", destination=SAFE_POINT)

        if not self.llm_decider.enabled and agent.say_cooldown == 0 and self.state.tick % 6 == 0:
            text = templates.render("explore", self.state.tick + len(agent.memory_short))
            return ActionIntent(
                kind="say",
                text=text,
                tags=["dialogue"],
                text_kind="explore",
                llm_generated=False,
            )

        if agent.mood < -55:
            return ActionIntent(kind="move", destination=SAFE_POINT)

        friend_id = pick_best_friend(self.state.relations, agent.id, sorted(self.state.agents.keys()))
        if friend_id:
            return ActionIntent(kind="move", target_id=friend_id)
        return ActionIntent(kind="move", destination=pick_wander_target(agent.id, self.state.tick))

    def _fallback_plan(self, agent: AgentState, must_answer: bool, allow_action_reply: bool = False) -> str:
        if must_answer and not allow_action_reply:
            return "Respond to incoming message"
        if must_answer and allow_action_reply:
            return "Handle world cue with movement"
        if agent.mood < -55:
            return "Move to safer place"
        if agent.mood > 40:
            return "Coordinate with allies"
        return "Observe and reposition"

    def _llm_first_failsafe_intent(
        self,
        agent: AgentState,
        must_answer: bool,
        reply_task: ReplyTask | None = None,
        allow_action_reply: bool = False,
    ) -> ActionIntent:
        if must_answer and agent.inbox and not allow_action_reply:
            return self._fallback_response_intent(agent, reply_task=reply_task)
        if must_answer and allow_action_reply:
            friend_id = pick_best_friend(self.state.relations, agent.id, sorted(self.state.agents.keys()))
            if friend_id:
                return ActionIntent(kind="move", target_id=friend_id)
            return ActionIntent(kind="move", destination=SAFE_POINT)
        if agent.mood < -55:
            return ActionIntent(kind="move", destination=SAFE_POINT)
        return ActionIntent(kind="move", destination=pick_wander_target(agent.id, self.state.tick))

    def _llm_first_failsafe_plan(self, must_answer: bool, allow_action_reply: bool = False) -> str:
        if must_answer and not allow_action_reply:
            return "LLM timeout: sending fail-safe reply"
        if must_answer and allow_action_reply:
            return "LLM timeout: fail-safe move for world cue"
        return "Awaiting LLM directive"

    def _should_retry_llm_before_fallback(self, must_answer: bool) -> bool:
        if not self.llm_decider.enabled:
            return False
        if must_answer:
            return self.llm_retry_on_must_answer
        return self._dialogue_ratio() < self.target_llm_response_ratio

    def _periodic_unqueued_reply_allowed(self, agent: AgentState) -> bool:
        interval = max(1, self.unqueued_reply_release_ticks)
        phase = sum(ord(ch) for ch in agent.id) % interval
        return (self.state.tick % interval) == phase

    def _can_agent_reply_now(self, agent: AgentState, message: dict[str, Any]) -> bool:
        source_is_agent = str(message.get("source_type", "")) == "agent"
        if source_is_agent:
            source_id = message.get("source_id")
            return (
                agent.message_cooldown == 0
                and isinstance(source_id, str)
                and source_id in self.state.agents
                and source_id != agent.id
            )
        return agent.say_cooldown == 0

    def _can_penalize_ignored_message(
        self,
        *,
        agent: AgentState,
        message: dict[str, Any],
        reply_task: ReplyTask | None,
        allow_unqueued_reply: bool,
        periodic_unqueued_release: bool,
    ) -> bool:
        if not self._is_inbox_message_reply_required(message, mutate=True):
            return False
        if not self._can_agent_reply_now(agent, message):
            return False
        if reply_task is not None:
            return str(message.get("inbox_id", "")) == reply_task.inbox_id
        return allow_unqueued_reply or periodic_unqueued_release

    @staticmethod
    def _relation_signal_from_event(tags: set[str], text: str) -> int:
        if "conflict" in tags:
            return -3
        if "help" in tags:
            return 2
        if "agent_message" in tags or "reply" in tags or "dialogue" in tags:
            sentiment = text_sentiment(text)
            if sentiment < 0:
                return -1
            return 1
        return 0

    def _recent_pair_relation_bias(self, src_id: str, dst_id: str) -> int:
        if self.relations_recent_event_weight <= 0:
            return 0

        start_tick = self.state.tick - self.relations_recent_window_ticks
        score = 0
        for event in reversed(self.state.event_log):
            event_tick = int(event.get("tick", self.state.tick))
            if event_tick < start_tick:
                break
            if event.get("source_id") != src_id:
                continue
            if event.get("target_id") != dst_id:
                continue

            tags = {str(tag).lower() for tag in event.get("tags", [])}
            text = str(event.get("text", ""))
            score += self._relation_signal_from_event(tags, text)
            score = _clamp_int(score, -self.relations_recent_event_weight, self.relations_recent_event_weight)
            if abs(score) >= self.relations_recent_event_weight:
                break
        return score

    def _step_relations(self) -> bool:
        if self.state.tick % self.relations_passive_step_ticks != 0:
            return False
        changed = False
        for src_id, src_agent in self.state.agents.items():
            for dst_id, dst_agent in self.state.agents.items():
                if src_id == dst_id:
                    continue
                key = (src_id, dst_id)
                current = self.state.relations[key]

                mood_gap = abs(src_agent.mood - dst_agent.mood)
                mood_alignment = 1 if mood_gap <= 35 else (-1 if mood_gap >= 60 else 0)

                plan_src = (src_agent.current_plan or "").strip()
                plan_dst = (dst_agent.current_plan or "").strip()
                plan_alignment = 1 if plan_src and plan_dst and plan_src[:10] == plan_dst[:10] else 0

                distance = self._distance_2d(src_agent.pos, dst_agent.pos)
                distance_signal = 1 if distance <= self.relations_proximity_radius else (
                    -1 if distance >= self.relations_distance_penalty_radius else 0
                )

                history_signal = self._recent_pair_relation_bias(src_id, dst_id)
                decay_to_center = -1 if current > 85 else (1 if current < -85 else 0)

                delta = mood_alignment + plan_alignment + distance_signal + history_signal + decay_to_center
                if delta == 0 and current != 0:
                    # Предотвращаем "замороженные" матрицы: очень редкий сдвиг к центру.
                    seed = (self.state.tick * 17 + sum(ord(ch) for ch in src_id) * 7 + sum(ord(ch) for ch in dst_id) * 11) % 13
                    if seed == 0:
                        delta = -1 if current > 0 else 1

                delta = _clamp_int(delta, -self.relations_passive_max_delta, self.relations_passive_max_delta)
                if delta == 0:
                    continue
                updated = clamp_relation(current + delta)
                if updated != current:
                    self.state.relations[key] = updated
                    changed = True
        return changed

    def _run_agent_brain(
        self,
        agent: AgentState,
        llm_decision: AgentDecision | None = None,
        reply_task: ReplyTask | None = None,
        allow_unqueued_reply: bool = True,
    ) -> tuple[list[dict], bool]:
        periodic_unqueued_release = self._periodic_unqueued_reply_allowed(agent)
        must_answer = reply_task is not None or self._has_pending_must_reply(agent)
        allow_action_reply = self._reply_task_allows_move_instead_of_text(reply_task)
        selected_required_message = self._selected_required_inbox_message(agent, reply_task=reply_task)
        strict_user_reply_via_llm = (
            self.llm_force_user_reply_via_llm and self._is_user_message_payload(selected_required_message)
        )
        answer_first_required = bool(
            selected_required_message is not None
            and self._text_has_question(str(selected_required_message.get("text", "")))
        )
        if selected_required_message is not None and bool(selected_required_message.get("no_question_required")):
            answer_first_required = True

        apply_ignored_inbox_penalty(
            relations=self.state.relations,
            inbox=agent.inbox,
            owner_agent_id=agent.id,
            now_tick=self.state.tick,
            can_penalize=lambda message: self._can_penalize_ignored_message(
                agent=agent,
                message=message,
                reply_task=reply_task,
                allow_unqueued_reply=allow_unqueued_reply,
                periodic_unqueued_release=periodic_unqueued_release,
            ),
        )

        if reply_task is None and must_answer and not allow_unqueued_reply and not periodic_unqueued_release:
            required_ticks = [
                int(message.get("received_tick", self.state.tick))
                for message in agent.inbox
                if self._is_inbox_message_reply_required(message, mutate=False)
            ]
            oldest_tick = min(required_ticks) if required_ticks else self.state.tick
            wait_ticks = self.state.tick - oldest_tick
            pending_must_reply = self._inbox_must_reply_count(agent, mutate_expired=False)
            agent.current_plan = f"В очереди ответов: {pending_must_reply}"
            agent.last_action = "idle"
            if wait_ticks > self.reply_queue_max_wait_ticks:
                agent.mood = _clamp_int(agent.mood - 1, -100, 100)
            return [], False

        intent: ActionIntent | None = None
        used_llm = False
        llm_first_mode = self.llm_decider.enabled and self.llm_first_hard_mode

        if strict_user_reply_via_llm and not self.llm_decider.enabled:
            agent.current_plan = "Ожидаю ответ через LLM"
            agent.last_action = "idle"
            if reply_task is not None:
                reply_task.retries += 1
            return [], False

        if llm_decision is None and self.llm_decider.enabled and (must_answer or llm_first_mode):
            llm_decision = self._llm_decision_for_single_agent(
                agent.id,
                retries=self.single_agent_backfill_retries,
                reply_task=reply_task,
            )

        if llm_decision is not None:
            llm_intent = self._intent_from_llm_decision(agent, llm_decision)
            if llm_intent is not None:
                if must_answer and not self._intent_satisfies_must_answer(
                    llm_intent,
                    allow_action_reply,
                    reply_task=reply_task,
                    answer_first_required=answer_first_required,
                ):
                    llm_intent = None
                else:
                    intent = llm_intent
                    used_llm = True
                    agent.current_plan = llm_decision.goal.strip() or agent.current_plan
                    agent.target_id = llm_intent.target_id

        if intent is None:
            if strict_user_reply_via_llm:
                intent = ActionIntent(kind="idle")
                agent.current_plan = "Ожидаю ответ через LLM"
                agent.target_id = None
            elif llm_first_mode:
                retry_decision: AgentDecision | None = None
                if must_answer and llm_decision is None:
                    retry_decision = self._llm_decision_for_single_agent(
                        agent.id,
                        retries=self.single_agent_backfill_retries,
                        reply_task=reply_task,
                    )
                retry_intent = self._intent_from_llm_decision(agent, retry_decision) if retry_decision is not None else None
                if retry_intent is not None and (
                    not must_answer
                    or self._intent_satisfies_must_answer(
                        retry_intent,
                        allow_action_reply,
                        reply_task=reply_task,
                        answer_first_required=answer_first_required,
                    )
                ):
                    intent = retry_intent
                    used_llm = True
                    llm_decision = retry_decision
                    agent.current_plan = retry_decision.goal.strip() or agent.current_plan
                    agent.target_id = retry_intent.target_id
                else:
                    intent = self._llm_first_failsafe_intent(
                        agent,
                        must_answer=must_answer,
                        reply_task=reply_task,
                        allow_action_reply=allow_action_reply,
                    )
                    agent.current_plan = self._llm_first_failsafe_plan(
                        must_answer=must_answer,
                        allow_action_reply=allow_action_reply,
                    )
                    agent.target_id = intent.target_id
            else:
                fallback_intent = self._fallback_intent(
                    agent,
                    must_answer=must_answer,
                    reply_task=reply_task,
                    allow_action_reply=allow_action_reply,
                )
                agent.current_plan = self._fallback_plan(
                    agent,
                    must_answer=must_answer,
                    allow_action_reply=allow_action_reply,
                )
                agent.target_id = fallback_intent.target_id

                if (
                    fallback_intent.kind in {"say", "message"}
                    and self._should_retry_llm_before_fallback(must_answer=must_answer)
                ):
                    retry_decision = self._llm_decision_for_single_agent(
                        agent.id,
                        retries=self.single_agent_backfill_retries,
                        reply_task=reply_task,
                    )
                    retry_intent = None
                    if retry_decision is not None:
                        retry_intent = self._intent_from_llm_decision(agent, retry_decision)

                    if retry_intent is not None and (
                        not must_answer
                        or self._intent_satisfies_must_answer(
                            retry_intent,
                            allow_action_reply,
                            reply_task=reply_task,
                            answer_first_required=answer_first_required,
                        )
                    ):
                        intent = retry_intent
                        used_llm = True
                        llm_decision = retry_decision
                        agent.current_plan = retry_decision.goal.strip() or agent.current_plan
                        agent.target_id = retry_intent.target_id
                    elif self.llm_decider.enabled and not must_answer:
                        intent = ActionIntent(kind="move", destination=pick_wander_target(agent.id, self.state.tick))
                    else:
                        intent = fallback_intent
                else:
                    intent = fallback_intent

        if intent is None:
            intent = ActionIntent(kind="idle")

        if selected_required_message is not None:
            selected_thread_id = str(selected_required_message.get("thread_id") or "").strip() or None
            if selected_thread_id and intent.thread_id is None:
                intent.thread_id = selected_thread_id

        if answer_first_required and intent.kind in {"say", "message"}:
            intent.force_non_question = True

        if self._reply_task_requires_source_message(reply_task):
            if intent.kind != "message" or intent.target_id != reply_task.source_id:
                fallback_intent = self._fallback_response_intent(agent, reply_task=reply_task)
                if fallback_intent.kind == "message" and fallback_intent.target_id == reply_task.source_id:
                    intent = fallback_intent
                else:
                    selector = self.state.tick * 109 + sum(ord(ch) for ch in agent.id)
                    source_name = self._agent_name(reply_task.source_id) or "друг"
                    intent = ActionIntent(
                        kind="message",
                        text=templates.render(
                            "respond_agent",
                            selector,
                            target_name=source_name,
                            topic=(agent.last_topic or "текущей ситуации")[:48],
                        ),
                        tags=["agent_message", "dialogue", "reply"],
                        target_id=reply_task.source_id,
                        text_kind="respond_agent",
                        llm_generated=False,
                        thread_id=reply_task.thread_id,
                        expects_reply=True,
                        force_non_question=answer_first_required,
                    )
            if reply_task is not None and reply_task.thread_id and intent.thread_id is None:
                intent.thread_id = reply_task.thread_id
            intent.expects_reply = True
            if answer_first_required:
                intent.force_non_question = True

        if must_answer and intent.kind in {"say", "message"} and intent.llm_generated:
            first_looping = self._looks_like_looping_coordination_question(intent.text)
            substance_ok, substance_reason = self._reply_substance_ok(
                agent=agent,
                intent=intent,
                source_message=selected_required_message,
            )
            if not substance_ok:
                retry_reason = "no_retry"
                accepted_retry = False
                retry_looping = False
                retry_decision = self._llm_decision_for_single_agent(
                    agent.id,
                    retries=0,
                    reply_task=reply_task,
                )
                retry_intent = self._intent_from_llm_decision(agent, retry_decision) if retry_decision is not None else None
                if retry_intent is not None and self._intent_satisfies_must_answer(
                    retry_intent,
                    allow_action_reply,
                    reply_task=reply_task,
                    answer_first_required=answer_first_required,
                ):
                    if selected_required_message is not None:
                        selected_thread_id = str(selected_required_message.get("thread_id") or "").strip() or None
                        if selected_thread_id and retry_intent.thread_id is None:
                            retry_intent.thread_id = selected_thread_id
                    if answer_first_required and retry_intent.kind in {"say", "message"}:
                        retry_intent.force_non_question = True
                    if self._reply_task_requires_source_message(reply_task):
                        if retry_intent.kind == "message" and retry_intent.target_id == reply_task.source_id:
                            retry_intent.expects_reply = True
                            if reply_task.thread_id and retry_intent.thread_id is None:
                                retry_intent.thread_id = reply_task.thread_id
                        else:
                            retry_intent = None

                    if retry_intent is not None:
                        retry_looping = self._looks_like_looping_coordination_question(retry_intent.text)
                        retry_ok, retry_reason = self._reply_substance_ok(
                            agent=agent,
                            intent=retry_intent,
                            source_message=selected_required_message,
                        )
                        if retry_ok:
                            intent = retry_intent
                            llm_decision = retry_decision
                            used_llm = True
                            if retry_decision is not None:
                                agent.current_plan = retry_decision.goal.strip() or agent.current_plan
                            agent.target_id = retry_intent.target_id
                            accepted_retry = True

                if not accepted_retry:
                    if strict_user_reply_via_llm:
                        intent = ActionIntent(kind="idle")
                        agent.current_plan = "Ожидаю ответ через LLM"
                    else:
                        if (first_looping and retry_looping) or (
                            substance_reason == "repetitive_loop" and retry_reason == "repetitive_loop"
                        ):
                            intent = self._fallback_move_intent(agent, reply_task=reply_task)
                        else:
                            intent = self._fallback_response_intent(agent, reply_task=reply_task)
                    used_llm = False

        if (
            selected_required_message is not None
            and str(selected_required_message.get("source_type", "")).lower() == "world"
            and intent.kind in {"say", "message"}
        ):
            forbidden_stances = [
                str(item).strip().lower()
                for item in selected_required_message.get("world_reaction_forbidden_stances", [])
                if str(item).strip()
            ]
            if forbidden_stances:
                intent_stance = self._world_reaction_stance_from_text(intent.text)
                if intent_stance in set(forbidden_stances):
                    fallback_intent = self._fallback_response_intent(agent, reply_task=reply_task)
                    fallback_intent.force_non_question = True
                    intent = fallback_intent

            world_event_id = str(selected_required_message.get("world_event_id") or "").strip()
            if world_event_id and not intent.evidence_ids:
                intent.evidence_ids = [world_event_id]

        event = self._execute_intent(agent, intent)

        if event is not None and agent.inbox and intent.kind in {"say", "message"}:
            consumed_inbox_id = reply_task.inbox_id if reply_task is not None else None
            self._consume_inbox(agent, intent, inbox_id=consumed_inbox_id)
        elif (
            reply_task is not None
            and event is None
            and intent.kind == "move"
            and allow_action_reply
        ):
            self._consume_inbox(
                agent,
                intent,
                inbox_id=reply_task.inbox_id,
                allow_nonverbal=True,
            )
        elif reply_task is not None and event is None:
            reply_task.retries += 1

        llm_relations_changed = used_llm and llm_decision is not None and self._apply_llm_deltas(agent, llm_decision)
        return ([event] if event is not None else []), llm_relations_changed

    def _immediate_world_reactions(self, text: str, sentiment: int, count: int) -> list[dict]:
        reactions: list[dict] = []
        ordered_agents = self._ordered_agents()
        if not ordered_agents:
            return reactions

        seed = sum(ord(ch) for ch in text) + self.state.tick
        primary_count = _clamp_int(count, 1, 2)
        primary_stances = self._primary_stances_for_event(event_id=f"immediate-{self.state.tick}", count=primary_count)
        used_stances: set[str] = set()
        visited: set[str] = set()
        for idx in range(len(ordered_agents)):
            if len(reactions) >= primary_count:
                break
            speaker = ordered_agents[(seed + idx) % len(ordered_agents)]
            if speaker.id in visited:
                continue
            visited.add(speaker.id)
            if speaker.say_cooldown > 0:
                continue

            llm_decision = self._llm_decision_for_single_agent(
                speaker.id,
                retries=self.single_agent_backfill_retries,
            )
            intent = self._intent_from_llm_decision(speaker, llm_decision) if llm_decision else None

            if intent is not None and intent.kind not in {"say", "message"}:
                intent = None

            if intent is None:
                if self.llm_decider.enabled:
                    continue
                if sentiment < 0 and parse_traits(speaker.traits).courage < 55:
                    text_line = templates.render("panic", seed + idx)
                    intent = ActionIntent(
                        kind="say",
                        text=text_line,
                        tags=["dialogue", "world", "conflict"],
                        text_kind="panic",
                    )
                else:
                    text_line = templates.render("support", seed + idx)
                    intent = ActionIntent(
                        kind="say",
                        text=text_line,
                        tags=["dialogue", "world", "help"],
                        text_kind="support",
                    )
            else:
                merged_tags = set(intent.tags)
                merged_tags.add("world")
                if intent.kind in {"say", "message"}:
                    merged_tags.add("dialogue")
                intent.tags = list(merged_tags)

            if len(reactions) == 0 and intent.kind in {"say", "message"}:
                intent.force_non_question = True

            if intent.kind in {"say", "message"}:
                current_stance = self._world_reaction_stance_from_text(intent.text)
                if current_stance in used_stances:
                    continue
                if len(reactions) < len(primary_stances):
                    assigned_stance = primary_stances[len(reactions)]
                    # Сохраняем маркер stance для нижестоящего контекста промпта и отладки.
                    tag_set = set(intent.tags)
                    tag_set.add("world")
                    tag_set.add(f"stance:{assigned_stance}")
                    intent.tags = list(tag_set)
                used_stances.add(current_stance)

            event = self._execute_intent(speaker, intent)
            if event is not None:
                reactions.append(event)
                if llm_decision is not None and intent.llm_generated:
                    speaker.current_plan = llm_decision.goal.strip() or speaker.current_plan
                    self._apply_llm_deltas(speaker, llm_decision)

        return reactions

    def _world_event_agents_snapshot(self) -> list[dict[str, Any]]:
        snapshot: list[dict[str, Any]] = []
        for agent in self._ordered_agents():
            snapshot.append(
                {
                    "id": agent.id,
                    "name": agent.name,
                    "traits": agent.traits,
                    "mood": agent.mood,
                    "mood_label": agent.mood_label,
                    "pos": {"x": round(agent.pos.x, 2), "z": round(agent.pos.z, 2)},
                }
            )
        return snapshot

    def _pick_world_reaction_agents(
        self,
        inbox_by_agent: dict[str, dict[str, Any]],
        *,
        count: int,
    ) -> list[tuple[AgentState, dict[str, Any]]]:
        candidates: list[tuple[tuple[int, int, str], AgentState, dict[str, Any]]] = []
        for agent_id, inbox_message in inbox_by_agent.items():
            agent = self.state.agents.get(agent_id)
            if agent is None:
                continue
            pending_must_reply = self._inbox_must_reply_count(agent, mutate_expired=False)
            cooldown_sum = agent.say_cooldown + agent.message_cooldown
            rank = (pending_must_reply, cooldown_sum, agent.id)
            candidates.append((rank, agent, inbox_message))

        if not candidates:
            return []
        candidates.sort(key=lambda item: item[0])
        selected: list[tuple[AgentState, dict[str, Any]]] = []
        for _rank, selected_agent, selected_message in candidates[: max(1, count)]:
            selected.append((selected_agent, selected_message))
        return selected

    def _maybe_emit_generated_world_event(self) -> tuple[dict, list[dict]] | None:
        events_snapshot = list(self.state.event_log)
        agents_snapshot = self._world_event_agents_snapshot()
        pending = len(self.reply_queue)
        if not self.world_event_generator.should_emit(
            tick=self.state.tick,
            events=events_snapshot,
            agents=agents_snapshot,
            reply_queue_pending=pending,
        ):
            return None

        generated: GeneratedWorldEvent | None = self.world_event_generator.generate(
            tick=self.state.tick,
            events=events_snapshot,
            agents=agents_snapshot,
            reply_queue_pending=pending,
        )
        if generated is None:
            return None

        event, reactions = self.add_world_event(
            text=generated.text,
            importance=generated.importance,
            emit_immediate_reactions=False,
            extra_tags=generated.tags,
            anchor=generated.anchor,
            severity=generated.severity,
            ensure_agent_reaction=True,
        )
        self.world_events_logger.warning(
            "world_event_emit tick=%s anchor=%s severity=%s importance=%.2f text=%s",
            self.state.tick,
            generated.anchor,
            generated.severity,
            generated.importance,
            generated.text,
        )
        return event, reactions

    def step(self) -> TickResult:
        self.state.tick += 1
        self._decrement_cooldowns()

        self._refresh_reply_queue()
        selected_reply_tasks = self._select_reply_tasks()
        selected_reply_tasks_by_agent = {task.agent_id: task for task in selected_reply_tasks}
        proactive_agent_ids = self._select_proactive_agents(
            excluded_agent_ids=set(selected_reply_tasks_by_agent.keys()),
            selected_replies_count=len(selected_reply_tasks),
        )
        proactive_agent_set = set(proactive_agent_ids)

        llm_decisions = self._llm_decisions_for_selected_agents(
            selected_reply_tasks,
            proactive_agent_ids=proactive_agent_ids,
        )
        if self.llm_decider.enabled:
            self._llm_debug(
                "LLM decision coverage "
                f"tick={self.state.tick} queue_selected={len(selected_reply_tasks)} "
                f"proactive_selected={len(proactive_agent_ids)} "
                f"decisions={len(llm_decisions)} queue_pending={len(self.reply_queue)}"
            )

        events: list[dict] = []
        llm_relations_changed = False
        for agent in self._ordered_agents():
            self._apply_passive_mood(agent)
            reply_task = selected_reply_tasks_by_agent.get(agent.id)
            agent_events, agent_relations_changed = self._run_agent_brain(
                agent=agent,
                llm_decision=llm_decisions.get(agent.id),
                reply_task=reply_task,
                allow_unqueued_reply=(
                    not self.llm_decider.enabled
                    or agent.id in proactive_agent_set
                ),
            )
            events.extend(agent_events)
            llm_relations_changed = llm_relations_changed or agent_relations_changed
            self._apply_background_movement(agent)

        generated_world = self._maybe_emit_generated_world_event()
        if generated_world is not None:
            world_event, world_reactions = generated_world
            events.append(world_event)
            events.extend(world_reactions)

        passive_relations_changed = self._step_relations()
        relations_changed = llm_relations_changed or passive_relations_changed or any(
            event.get("target_id") is not None for event in events
        )
        relations_changed = relations_changed or (self.state.tick % self.relations_interval_ticks == 0)

        return TickResult(events=events, relations_changed=relations_changed)

    def agents_state_payload(self) -> list[dict]:
        payload: list[dict] = []
        for agent in self._ordered_agents():
            item = agent.to_state_payload()
            item["tick"] = self.state.tick
            payload.append(item)
        return payload

    def relations_payload(self) -> dict:
        return {
            "nodes": [{"id": agent.id, "name": agent.name} for agent in self._ordered_agents()],
            "edges": [
                {"from": src, "to": dst, "value": value}
                for (src, dst), value in sorted(self.state.relations.items())
            ],
        }

    def agents_list_payload(self) -> list[dict]:
        return [agent.to_agent_summary() for agent in self._ordered_agents()]

    def state_payload(self) -> dict:
        return {
            "tick": self.state.tick,
            "speed": self.state.speed,
            "agents": self.agents_state_payload(),
            "relations": self.relations_payload(),
            "events": list(self.state.event_log)[-200:],
            "llm_stats": self._llm_stats_payload(),
            "world_event_stats": self._world_event_stats_payload(),
            "memory_stats": self.memory_store.stats(),
        }

    def events_payload(self, limit: int = 200, agent_id: str | None = None) -> list[dict]:
        items = list(self.state.event_log)
        if agent_id:
            items = [event for event in items if event.get("source_id") == agent_id]
        return items[-max(1, min(limit, 500)) :]

    def events_since(self, after_event_id: int) -> list[dict]:
        threshold = max(0, int(after_event_id))

        def _event_id_as_int(event: dict[str, Any]) -> int:
            raw = str(event.get("id", ""))
            if raw.startswith("e"):
                raw = raw[1:]
            try:
                return int(raw)
            except ValueError:
                return 0

        return [event for event in self.state.event_log if _event_id_as_int(event) > threshold]

    def update_speed(self, speed: float) -> float:
        self.state.speed = _clamp_float(speed, 0.1, 5.0)
        return self.state.speed

    def add_world_event(
        self,
        text: str,
        importance: float | None = None,
        emit_immediate_reactions: bool = True,
        extra_tags: list[str] | None = None,
        anchor: str | None = None,
        severity: str | None = None,
        ensure_agent_reaction: bool = False,
    ) -> tuple[dict, list[dict]]:
        normalized_importance = None
        if importance is not None:
            normalized_importance = _clamp_float(float(importance), 0.0, 1.0)

        normalized_tags: list[str] = ["world"]
        if extra_tags:
            normalized_tags.extend(str(tag).strip().lower() for tag in extra_tags if str(tag).strip())
        if anchor:
            normalized_tags.append(f"anchor:{anchor}")
        if severity:
            normalized_tags.append(str(severity).strip().lower())
        if normalized_importance is not None and normalized_importance >= 0.7:
            normalized_tags.append("important")
        if str(severity).strip().lower() == "danger":
            normalized_tags.append("important")

        deduped_tags: list[str] = []
        seen_tags: set[str] = set()
        for tag in normalized_tags:
            if tag in seen_tags:
                continue
            seen_tags.add(tag)
            deduped_tags.append(tag)
        if "world" not in seen_tags:
            deduped_tags.insert(0, "world")

        event = self._append_event(
            source_type="world",
            source_id=None,
            text=text,
            tags=deduped_tags,
            importance=normalized_importance,
            anchor=anchor,
            severity=severity,
        )
        sentiment = text_sentiment(text)
        intensity = max(1, int(round((normalized_importance if normalized_importance is not None else 0.5) * 10)))
        mood_shift = sentiment * max(2, intensity // 2)
        inbox_by_agent: dict[str, dict[str, Any]] = {}

        for idx, agent in enumerate(self._ordered_agents()):
            spread = idx if mood_shift < 0 else -idx
            agent.mood = _clamp_int(agent.mood + mood_shift + spread, -100, 100)
            agent.last_topic = text[:60]
            inbox_message = self._enqueue_inbox(
                target_id=agent.id,
                source_type="world",
                source_id=None,
                text=text,
                tags=deduped_tags,
                expects_reply=False,
                can_reply=False,
            )
            if inbox_message is not None:
                inbox_message["world_event_id"] = event["id"]
                inbox_by_agent[agent.id] = inbox_message

        if ensure_agent_reaction and inbox_by_agent:
            primary_count = self._world_primary_reactor_count(
                importance=normalized_importance,
                severity=severity,
            )
            primary_agents = self._pick_world_reaction_agents(
                inbox_by_agent,
                count=primary_count,
            )
            primary_agent_ids = [agent.id for agent, _ in primary_agents]
            primary_stances = self._primary_stances_for_event(
                event_id=str(event.get("id", "")),
                count=len(primary_agents),
            )

            for agent_id, inbox_message in inbox_by_agent.items():
                inbox_message["world_reaction_primary_agents"] = list(primary_agent_ids)

            for idx, (selected_agent, selected_message) in enumerate(primary_agents):
                selected_tags = [str(tag) for tag in selected_message.get("tags", []) if str(tag).strip()]
                if "primary_world_reaction" not in selected_tags:
                    selected_tags.append("primary_world_reaction")
                selected_message["tags"] = selected_tags
                selected_message["world_reaction_role"] = "primary"
                selected_message["world_reaction_stance"] = (
                    primary_stances[idx] if idx < len(primary_stances) else "curiosity"
                )
                selected_message["world_reaction_forbidden_stances"] = list(primary_stances[:idx])
                selected_message["no_question_required"] = True
                selected_message["expects_reply"] = True
                selected_message["requires_reply"] = True
                selected_message["can_reply"] = True
                self._queue_reply_task(selected_agent, selected_message)

            for agent_id, inbox_message in inbox_by_agent.items():
                if agent_id in primary_agent_ids:
                    continue
                inbox_message["world_reaction_role"] = "observer"
                inbox_message["world_reaction_forbidden_stances"] = list(primary_stances)

        reactions: list[dict] = []
        if emit_immediate_reactions:
            reaction_count = 2 if (normalized_importance is not None and normalized_importance >= 0.8) else 1 + (sum(ord(ch) for ch in text) % 2)
            reaction_count = _clamp_int(reaction_count, 1, 2)
            reactions = self._immediate_world_reactions(text=text, sentiment=sentiment, count=reaction_count)
        return event, reactions

    def add_agent_message(self, agent_id: str, text: str) -> tuple[dict, dict | None]:
        if agent_id not in self.state.agents:
            raise KeyError(agent_id)

        target = self.state.agents[agent_id]
        event = self._append_event(
            source_type="world",
            source_id=None,
            text=f"User -> {target.name}: {text}",
            tags=["user_message", "dialogue"],
            target_id=agent_id,
            expects_reply=True,
            can_reply=True,
        )
        self._enqueue_inbox(
            target_id=agent_id,
            source_type="world",
            source_id=None,
            text=text,
            tags=["user_message", "dialogue"],
            thread_id=event.get("thread_id"),
            expects_reply=True,
            can_reply=True,
            directed=True,
        )

        sentiment = text_sentiment(text)
        target.mood = _clamp_int(target.mood + sentiment * 8, -100, 100)
        target.current_plan = "Respond to user"
        target.target_id = None
        target.last_topic = text[:60]
        answer_first_required = self._text_has_question(text)
        reply_task = None
        if target.inbox:
            latest_inbox = target.inbox[-1]
            reply_task = self.reply_task_by_inbox_id.get(latest_inbox.get("inbox_id"))

        if self.llm_force_user_reply_via_llm and not self.llm_decider.enabled:
            raise RuntimeError("LLM is required for user replies. Enable LLM_DECIDER_ENABLED=1.")

        llm_decision = self._llm_decision_for_single_agent(
            target.id,
            retries=self.single_agent_backfill_retries,
            reply_task=reply_task,
        )
        intent = self._intent_from_llm_decision(target, llm_decision) if llm_decision is not None else None

        if intent is not None and intent.kind == "message":
            intent.kind = "say"
            intent.target_id = None

        if intent is not None and intent.kind == "say":
            intent.target_id = None
            merged_tags = set(intent.tags)
            merged_tags.update({"dialogue", "reply", "user_message", "llm"})
            intent.tags = list(merged_tags)
            intent.llm_generated = True
            intent.thread_id = event.get("thread_id")
            intent.expects_reply = False
            if answer_first_required:
                intent.force_non_question = True
        elif self.llm_force_user_reply_via_llm:
            target.current_plan = "Ожидаю ответ через LLM"
            target.last_action = "idle"
            if reply_task is not None:
                reply_task.retries += 1
            return event, None
        else:
            selector = self.state.tick + self.state.next_event_id
            if sentiment < 0 and parse_traits(target.traits).aggression > 60:
                reply_text = templates.render("conflict", selector)
                reply_tags = ["dialogue", "reply", "user_message", "conflict"]
                reply_kind = "conflict"
            else:
                reply_text = templates.render("respond_user", selector)
                reply_tags = ["dialogue", "reply", "user_message"]
                reply_kind = "respond_user"
            intent = ActionIntent(
                kind="say",
                text=reply_text,
                tags=reply_tags,
                text_kind=reply_kind,
                llm_generated=False,
                thread_id=event.get("thread_id"),
                expects_reply=False,
                force_non_question=answer_first_required,
            )

        target.say_cooldown = 0
        reply = self._execute_intent(target, intent)

        if reply is None and self.llm_force_user_reply_via_llm:
            target.current_plan = "Ожидаю ответ через LLM"
            target.last_action = "idle"
            if reply_task is not None:
                reply_task.retries += 1
            return event, None

        if reply is None:
            fallback_text = templates.render("respond_user", self.state.tick + self.state.next_event_id)
            reply = self._append_event(
                source_type="agent",
                source_id=target.id,
                text=fallback_text,
                tags=["dialogue", "reply", "user_message"],
                llm_generated=False,
                thread_id=event.get("thread_id"),
                expects_reply=False,
                can_reply=False,
            )
            target.last_action = "say"
            target.last_say = fallback_text
            target.say_cooldown = 2
            target.last_interaction_tick = self.state.tick

        if reply is not None and llm_decision is not None and intent.llm_generated:
            target.current_plan = llm_decision.goal.strip() or target.current_plan
            self._apply_llm_deltas(target, llm_decision)

        if reply is not None and target.inbox:
            self._consume_inbox(
                target,
                intent,
                inbox_id=reply_task.inbox_id if reply_task is not None else None,
            )
        return event, reply

    def agent_details(self, agent_id: str) -> dict | None:
        agent = self.state.agents.get(agent_id)
        if not agent:
            return None

        key_memories = self.memory_store.recall(
            agent_id=agent.id,
            query=agent.last_topic or agent.current_plan,
            limit=5,
        )
        if not key_memories:
            key_memories = self._fallback_memories(agent.id, 5)

        recent_events = [
            event
            for event in self.state.event_log
            if event.get("source_id") == agent.id or event.get("target_id") == agent.id
        ][-10:]

        relation_items: list[dict[str, Any]] = []
        for other in self._ordered_agents():
            if other.id == agent.id:
                continue
            relation_items.append(
                {
                    "agent_id": other.id,
                    "name": other.name,
                    "value": self.state.relations.get((agent.id, other.id), 0),
                }
            )
        relation_items.sort(key=lambda item: item["value"], reverse=True)
        top_positive = [item for item in relation_items if item["value"] >= 0][:3]
        top_negative = list(reversed([item for item in relation_items if item["value"] < 0][-3:]))

        return {
            "id": agent.id,
            "name": agent.name,
            "traits": agent.traits,
            "mood": agent.mood,
            "mood_label": agent.mood_label,
            "current_plan": agent.current_plan,
            "target_id": agent.target_id,
            "cooldowns": {
                "say_cooldown": agent.say_cooldown,
                "message_cooldown": agent.message_cooldown,
            },
            "inbox_size": len(agent.inbox),
            "inbox_preview": agent.inbox[-5:],
            "reply_queue_pending": sum(1 for task in self.reply_queue if task.agent_id == agent.id),
            "memory_short": list(agent.memory_short[-10:]),
            "key_memories": key_memories,
            "recent_events": recent_events,
            "relations_snapshot": {
                "top_positive": top_positive,
                "top_negative": top_negative,
            },
            "llm_stats": self._llm_stats_payload(),
            "world_event_stats": self._world_event_stats_payload(),
        }
