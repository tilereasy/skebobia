from __future__ import annotations

import logging
import os
import random
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Deque

from app.agents.agent import AgentState, Vec3, plan_for_mood
from app.memory.store import EpisodicMemoryStore
from app.sim import templates
from app.sim.llm_decider import AgentDecision, LLMTickDecider
from app.sim.movement import SAFE_POINT, pick_wander_target, step_towards
from app.sim.relations import apply_ignored_inbox_penalty, clamp as clamp_relation, update_relations_from_event
from app.sim.rules import parse_traits, pick_best_friend, text_sentiment


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


@dataclass
class ReplyTask:
    id: str
    agent_id: str
    inbox_id: str
    source_id: str | None
    source_type: str
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

        self.llm_decider = LLMTickDecider.from_env()
        self.memory_store = EpisodicMemoryStore.from_env()

        self.prompt_recent_events_limit = _env_int("LLM_PROMPT_RECENT_EVENTS", 6, 2, 24)
        self.prompt_inbox_limit = _env_int("LLM_PROMPT_INBOX", 4, 1, 12)
        self.prompt_memories_limit = _env_int("LLM_PROMPT_MEMORIES", 5, 1, 12)
        self.single_agent_backfill_retries = _env_int("LLM_DECIDER_BACKFILL_RETRIES", 2, 0, 4)

        self.target_llm_response_ratio = _env_float("LLM_TARGET_RESPONSE_RATIO", 0.9, 0.1, 0.99)
        self.response_ratio_window = _env_int("LLM_RESPONSE_RATIO_WINDOW", 240, 20, 1000)
        self.dialogue_llm_window: Deque[int] = deque(maxlen=self.response_ratio_window)

        self.max_replies_per_tick = _env_int("REPLY_QUEUE_MAX_REPLIES_PER_TICK", 2, 1, 16)
        self.reply_queue_max_wait_ticks = _env_int("REPLY_QUEUE_MAX_WAIT_TICKS", 10, 2, 60)
        self.reply_queue_max_skips = _env_int("REPLY_QUEUE_MAX_SKIPS", 2, 0, 8)
        self.reply_queue: Deque[ReplyTask] = deque(maxlen=_env_int("REPLY_QUEUE_MAX_SIZE", 512, 32, 4096))
        self.reply_task_by_inbox_id: dict[str, ReplyTask] = {}
        self.reply_task_seq = 0
        self.inbox_seq = 0
        self.proactive_llm_agents_per_tick = _env_int("LLM_PROACTIVE_AGENTS_PER_TICK", 1, 0, 8)
        self.startup_world_event_enabled = _env_bool("STARTUP_WORLD_EVENT_ENABLED", True)
        self.startup_world_event_importance = _env_float("STARTUP_WORLD_EVENT_IMPORTANCE", 0.85, 0.2, 1.0)

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

        tags = list(message.get("tags", []))
        source_type = str(message.get("source_type", ""))
        if source_type == "world":
            requires_reply = ("user_message" in tags) or ("important" in tags)
        else:
            requires_reply = ("dialogue" in tags) or ("agent_message" in tags) or (source_type == "agent")
        if not requires_reply:
            return

        self.reply_task_seq += 1
        task = ReplyTask(
            id=f"q{self.reply_task_seq}",
            agent_id=agent.id,
            inbox_id=inbox_id,
            source_id=message.get("source_id"),
            source_type=source_type,
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
            if self._find_inbox_message(agent, task.inbox_id) is None:
                continue
            alive.append(task)
            alive_map[task.inbox_id] = task
        self.reply_queue = alive
        self.reply_task_by_inbox_id = alive_map

    def _should_skip_reply(self, agent: AgentState, task: ReplyTask, message: dict[str, Any]) -> bool:
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

            can_say = agent.say_cooldown == 0
            can_message = agent.message_cooldown == 0 and message.get("source_type") == "agent"
            if not can_say and not can_message:
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
            "target_response_ratio": self.target_llm_response_ratio,
            "dialogue_ratio_window": len(self.dialogue_llm_window),
            "dialogue_ratio_llm": round(self._dialogue_ratio(), 3),
            "reply_queue": {
                "pending": len(self.reply_queue),
                "max_replies_per_tick": self.max_replies_per_tick,
                "proactive_agents_per_tick": self.proactive_llm_agents_per_tick,
            },
        }

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
        for previous in recent_texts:
            previous_normalized = self._normalize_dialogue_text(previous)
            if not previous_normalized:
                continue
            if candidate == previous_normalized:
                return True

            previous_tokens = set(previous_normalized.split())
            if len(candidate_tokens) < 3 or len(previous_tokens) < 3:
                continue
            overlap = len(candidate_tokens & previous_tokens) / max(len(candidate_tokens), len(previous_tokens))
            if overlap >= 0.85:
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
                    return variant[:280]

        stem = candidate.rstrip(".!? ")
        if not stem:
            stem = candidate
        fallback_variants = [
            f"{stem}. Мм.",
            f"{stem}. Угу.",
            f"{stem}. Ну да.",
            f"{stem}. Хм.",
        ]
        for variant in fallback_variants:
            if not self._is_repetitive_dialogue(variant, recent):
                return variant[:280]
        return candidate[:280]

    def _event_prompt_item(self, event: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": event.get("id"),
            "source_id": event.get("source_id"),
            "target_id": event.get("target_id"),
            "text": str(event.get("text", ""))[:160],
            "tags": list(event.get("tags", []))[:4],
            "tick": event.get("tick"),
        }

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
        allowed_actions = ["move", "idle"]
        if agent.say_cooldown == 0:
            allowed_actions.append("say")
        if agent.message_cooldown == 0:
            allowed_actions.append("message")

        inbox_items = [
            {
                "source_id": message.get("source_id"),
                "source_type": message.get("source_type"),
                "text": str(message.get("text", ""))[:140],
                "tags": list(message.get("tags", []))[:3],
            }
            for message in agent.inbox[-self.prompt_inbox_limit :]
        ]

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
        }
        if reply_task is not None:
            queue_payload.update(
                {
                    "task_id": reply_task.id,
                    "inbox_id": reply_task.inbox_id,
                    "source_id": reply_task.source_id,
                    "source_type": reply_task.source_type,
                    "text": reply_task.text[:180],
                    "wait_ticks": self.state.tick - reply_task.created_tick,
                    "retries": reply_task.retries,
                    "skips": reply_task.skips,
                    "reply_policy": {
                        "can_skip": self.reply_policy_by_agent.get(agent.id, ReplyPolicy(False, 0.0)).can_skip,
                        "max_skips": self.reply_queue_max_skips,
                    },
                }
            )

        return {
            "agent_id": agent.id,
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
                for event in list(self.state.event_log)[-self.prompt_recent_events_limit :]
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
                for event in list(self.state.event_log)[-self.prompt_recent_events_limit :]
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
            ("a3", "Эпштейн", "злой, импульсивный", "#e76f51", -20),
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

        self.state.event_log.append(event)
        self._remember_event(event["id"])
        self._remember_event_in_store(event)

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
    ) -> None:
        target = self.state.agents.get(target_id)
        if not target:
            return
        source_name = self._agent_name(source_id) if source_id else None
        self.inbox_seq += 1
        inbox_message = {
            "inbox_id": f"i{self.inbox_seq}",
            "source_type": source_type,
            "source_id": source_id,
            "source_name": source_name,
            "text": text,
            "tags": list(tags),
            "received_tick": self.state.tick,
            "penalized": False,
        }
        target.inbox.append(inbox_message)
        self._queue_reply_task(target, inbox_message)

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
        if intent.target_id and intent.target_id in self.state.agents:
            destination = self.state.agents[intent.target_id].pos
        elif intent.destination is not None:
            destination = intent.destination
        else:
            destination = pick_wander_target(agent.id, self.state.tick)

        old_pos = agent.pos
        max_step = 0.6 + 0.45 * max(0.5, self.state.speed)
        new_pos = step_towards(agent.pos, destination, max_step=max_step)
        agent.pos = new_pos

        dx = new_pos.x - old_pos.x
        dz = new_pos.z - old_pos.z
        if abs(dx) > 1e-6 or abs(dz) > 1e-6:
            agent.look_at = Vec3(dx, 0.0, dz)
        agent.last_action = "move"
        agent.target_id = intent.target_id

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
        event = self._append_event(
            source_type="agent",
            source_id=agent.id,
            text=text,
            tags=tags,
            target_id=intent.target_id,
            llm_generated=intent.llm_generated,
        )

        agent.last_action = "say"
        agent.last_say = text
        agent.say_cooldown = 2
        agent.last_interaction_tick = self.state.tick
        agent.target_id = intent.target_id

        if intent.target_id:
            self._enqueue_inbox(
                target_id=intent.target_id,
                source_type="agent",
                source_id=agent.id,
                text=text,
                tags=tags,
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
        event = self._append_event(
            source_type="agent",
            source_id=agent.id,
            text=text,
            tags=tags,
            target_id=intent.target_id,
            llm_generated=intent.llm_generated,
        )
        self._enqueue_inbox(
            target_id=intent.target_id,
            source_type="agent",
            source_id=agent.id,
            text=text,
            tags=tags,
        )

        agent.last_action = "say"
        agent.last_say = text
        agent.target_id = intent.target_id
        agent.say_cooldown = max(agent.say_cooldown, 1)
        agent.message_cooldown = 3
        agent.last_interaction_tick = self.state.tick
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
            target_id: str | None = None
            if (
                decision.target_id
                and decision.target_id in self.state.agents
                and decision.target_id != agent.id
            ):
                target_id = decision.target_id

            destination = None
            if decision.move_to is not None:
                destination = Vec3(x=decision.move_to.x, z=decision.move_to.z)
            if target_id is None and destination is None:
                destination = pick_wander_target(agent.id, self.state.tick)

            return ActionIntent(kind="move", target_id=target_id, destination=destination, llm_generated=True)

        if decision.act == "say":
            if agent.say_cooldown > 0:
                return ActionIntent(kind="idle", llm_generated=True)
            text = (decision.say_text or "").strip()
            if not text:
                return ActionIntent(kind="idle", llm_generated=True)

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
            )

        if decision.act == "message":
            if agent.message_cooldown > 0:
                return ActionIntent(kind="idle", llm_generated=True)
            target_id = decision.target_id
            text = (decision.say_text or "").strip()
            if not text:
                return ActionIntent(kind="idle", llm_generated=True)

            if not target_id or target_id not in self.state.agents or target_id == agent.id:
                return ActionIntent(
                    kind="say",
                    text=text,
                    tags=["dialogue", "llm"],
                    target_id=None,
                    llm_generated=True,
                )
            return ActionIntent(
                kind="message",
                text=text,
                tags=["agent_message", "dialogue", "llm"],
                target_id=target_id,
                llm_generated=True,
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

    def _consume_inbox(self, agent: AgentState, intent: ActionIntent, inbox_id: str | None = None) -> None:
        if not agent.inbox:
            return
        if intent.kind not in {"say", "message"}:
            return

        index = -1
        if inbox_id:
            for idx, message in enumerate(agent.inbox):
                if message.get("inbox_id") == inbox_id:
                    index = idx
                    break

        if index >= 0:
            processed = agent.inbox.pop(index)
        else:
            latest_message = agent.inbox[-1]
            source_id = latest_message.get("source_id")
            if intent.kind == "message" and source_id and intent.target_id and intent.target_id != source_id:
                return
            processed = agent.inbox.pop(-1)

        self._remove_reply_task(processed.get("inbox_id"))
        agent.last_topic = str(processed.get("text", ""))[:60]
        agent.last_interaction_tick = self.state.tick

    def _fallback_response_intent(self, agent: AgentState, reply_task: ReplyTask | None = None) -> ActionIntent:
        latest_message: dict[str, Any] = {}
        if reply_task is not None:
            found = self._find_inbox_message(agent, reply_task.inbox_id)
            if found is not None:
                latest_message = found
        if not latest_message:
            latest_message = agent.inbox[-1] if agent.inbox else {}
        source_id = latest_message.get("source_id")
        source_type = str(latest_message.get("source_type", ""))
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
            )

        text = templates.render("respond_user", selector)
        tags = ["dialogue", "reply"]
        if source_type != "agent":
            tags.append("user_message")
        return ActionIntent(
            kind="say",
            text=text,
            tags=tags,
            text_kind="respond_user",
            llm_generated=False,
        )

    def _fallback_intent(
        self,
        agent: AgentState,
        must_answer: bool,
        reply_task: ReplyTask | None = None,
    ) -> ActionIntent:
        if must_answer and agent.inbox:
            return self._fallback_response_intent(agent, reply_task=reply_task)

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

    def _fallback_plan(self, agent: AgentState, must_answer: bool) -> str:
        if must_answer:
            return "Respond to incoming message"
        if agent.mood < -55:
            return "Move to safer place"
        if agent.mood > 40:
            return "Coordinate with allies"
        return "Observe and reposition"

    def _should_retry_llm_before_fallback(self, must_answer: bool) -> bool:
        if not self.llm_decider.enabled:
            return False
        if must_answer:
            return True
        return self._dialogue_ratio() < self.target_llm_response_ratio

    def _step_relations(self) -> None:
        for src_id, src_agent in self.state.agents.items():
            for dst_id, dst_agent in self.state.agents.items():
                if src_id == dst_id:
                    continue
                key = (src_id, dst_id)
                current = self.state.relations[key]
                mood_alignment = 1 if (src_agent.mood + dst_agent.mood) >= 0 else -1
                plan_alignment = 1 if src_agent.current_plan[:12] == dst_agent.current_plan[:12] else 0
                decay_to_center = -1 if current > 0 else (1 if current < 0 else 0)
                self.state.relations[key] = clamp_relation(current + mood_alignment + plan_alignment + decay_to_center)

    def _run_agent_brain(
        self,
        agent: AgentState,
        llm_decision: AgentDecision | None = None,
        reply_task: ReplyTask | None = None,
        allow_unqueued_reply: bool = True,
    ) -> tuple[list[dict], bool]:
        apply_ignored_inbox_penalty(
            relations=self.state.relations,
            inbox=agent.inbox,
            owner_agent_id=agent.id,
            now_tick=self.state.tick,
        )

        if agent.inbox:
            latest_sentiment = text_sentiment(str(agent.inbox[-1].get("text", "")))
            agent.mood = _clamp_int(agent.mood + latest_sentiment * 3, -100, 100)

        if reply_task is None and agent.inbox and not allow_unqueued_reply:
            oldest_tick = min(int(message.get("received_tick", self.state.tick)) for message in agent.inbox)
            wait_ticks = self.state.tick - oldest_tick
            agent.current_plan = f"В очереди ответов: {len(agent.inbox)}"
            agent.last_action = "idle"
            if wait_ticks > self.reply_queue_max_wait_ticks:
                agent.mood = _clamp_int(agent.mood - 1, -100, 100)
            return [], False

        must_answer = reply_task is not None or bool(agent.inbox)
        intent: ActionIntent | None = None
        used_llm = False

        if llm_decision is None and must_answer and self.llm_decider.enabled:
            llm_decision = self._llm_decision_for_single_agent(
                agent.id,
                retries=self.single_agent_backfill_retries,
                reply_task=reply_task,
            )

        if llm_decision is not None:
            llm_intent = self._intent_from_llm_decision(agent, llm_decision)
            if llm_intent is not None:
                if must_answer and llm_intent.kind not in {"say", "message"}:
                    llm_intent = None
                else:
                    intent = llm_intent
                    used_llm = True
                    agent.current_plan = llm_decision.goal.strip() or agent.current_plan
                    agent.target_id = llm_intent.target_id

        if intent is None:
            fallback_intent = self._fallback_intent(agent, must_answer=must_answer, reply_task=reply_task)
            agent.current_plan = self._fallback_plan(agent, must_answer=must_answer)
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

                if retry_intent is not None and (not must_answer or retry_intent.kind in {"say", "message"}):
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

        event = self._execute_intent(agent, intent)

        if event is not None and agent.inbox and intent.kind in {"say", "message"}:
            consumed_inbox_id = reply_task.inbox_id if reply_task is not None else None
            self._consume_inbox(agent, intent, inbox_id=consumed_inbox_id)
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
        for idx in range(count):
            speaker = ordered_agents[(seed + idx) % len(ordered_agents)]
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

            event = self._execute_intent(speaker, intent)
            if event is not None:
                reactions.append(event)
                if llm_decision is not None and intent.llm_generated:
                    speaker.current_plan = llm_decision.goal.strip() or speaker.current_plan
                    self._apply_llm_deltas(speaker, llm_decision)

        return reactions

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

        relations_changed = llm_relations_changed or any(event.get("target_id") is not None for event in events)
        relations_changed = relations_changed or (self.state.tick % self.relations_interval_ticks == 0)
        if relations_changed:
            self._step_relations()

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
            "memory_stats": self.memory_store.stats(),
        }

    def events_payload(self, limit: int = 200, agent_id: str | None = None) -> list[dict]:
        items = list(self.state.event_log)
        if agent_id:
            items = [event for event in items if event.get("source_id") == agent_id]
        return items[-max(1, min(limit, 500)) :]

    def update_speed(self, speed: float) -> float:
        self.state.speed = _clamp_float(speed, 0.1, 5.0)
        return self.state.speed

    def add_world_event(
        self,
        text: str,
        importance: float | None = None,
        emit_immediate_reactions: bool = True,
    ) -> tuple[dict, list[dict]]:
        tags = ["world"]
        if importance is not None and importance >= 0.7:
            tags.append("important")

        event = self._append_event(source_type="world", source_id=None, text=text, tags=tags)
        sentiment = text_sentiment(text)
        intensity = max(1, int(round((importance if importance is not None else 0.5) * 10)))
        mood_shift = sentiment * max(2, intensity // 2)

        for idx, agent in enumerate(self._ordered_agents()):
            spread = idx if mood_shift < 0 else -idx
            agent.mood = _clamp_int(agent.mood + mood_shift + spread, -100, 100)
            agent.last_topic = text[:60]
            self._enqueue_inbox(
                target_id=agent.id,
                source_type="world",
                source_id=None,
                text=text,
                tags=tags,
            )

        reactions: list[dict] = []
        if emit_immediate_reactions:
            reaction_count = 2 if (importance is not None and importance >= 0.8) else 1 + (sum(ord(ch) for ch in text) % 2)
            reaction_count = _clamp_int(reaction_count, 1, 2)
            reactions = self._immediate_world_reactions(text=text, sentiment=sentiment, count=reaction_count)
        return event, reactions

    def add_agent_message(self, agent_id: str, text: str) -> tuple[dict, dict]:
        if agent_id not in self.state.agents:
            raise KeyError(agent_id)

        target = self.state.agents[agent_id]
        event = self._append_event(
            source_type="world",
            source_id=None,
            text=f"User -> {target.name}: {text}",
            tags=["user_message", "dialogue"],
            target_id=agent_id,
        )
        self._enqueue_inbox(
            target_id=agent_id,
            source_type="world",
            source_id=None,
            text=text,
            tags=["user_message", "dialogue"],
        )

        sentiment = text_sentiment(text)
        target.mood = _clamp_int(target.mood + sentiment * 8, -100, 100)
        target.current_plan = "Respond to user"
        target.target_id = None
        target.last_topic = text[:60]
        reply_task = None
        if target.inbox:
            latest_inbox = target.inbox[-1]
            reply_task = self.reply_task_by_inbox_id.get(latest_inbox.get("inbox_id"))

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
            )

        target.say_cooldown = 0
        reply = self._execute_intent(target, intent)

        if reply is None:
            fallback_text = templates.render("respond_user", self.state.tick + self.state.next_event_id)
            reply = self._append_event(
                source_type="agent",
                source_id=target.id,
                text=fallback_text,
                tags=["dialogue", "reply", "user_message"],
                llm_generated=False,
            )
            target.last_action = "say"
            target.last_say = fallback_text
            target.say_cooldown = 2
            target.last_interaction_tick = self.state.tick

        if llm_decision is not None and intent.llm_generated:
            target.current_plan = llm_decision.goal.strip() or target.current_plan
            self._apply_llm_deltas(target, llm_decision)

        if target.inbox:
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
            "llm_stats": self._llm_stats_payload(),
        }
