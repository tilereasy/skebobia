from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Deque

from app.agents.agent import AgentState, Vec3, plan_for_mood
from app.sim import templates
from app.sim.llm_decider import AgentDecision, LLMTickDecider
from app.sim.movement import SAFE_POINT, pick_wander_target, step_towards
from app.sim.relations import apply_ignored_inbox_penalty, clamp as clamp_relation, update_relations_from_event
from app.sim.rules import (
    GOAL_EXPLORE,
    GOAL_HELP,
    GOAL_PANIC,
    GOAL_RESPOND,
    GOAL_SOCIAL,
    ActionIntent,
    act,
    choose_goal,
    goal_to_plan,
    parse_traits,
    pick_best_friend,
    reflect,
    text_sentiment,
)


def _utc_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _clamp_float(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


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
        self.prompt_recent_events_limit = 5
        self.prompt_inbox_limit = 3
        self.prompt_memories_limit = 3
        self._seed_initial_events()

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
        topic_hint = (agent.last_topic or "обстановке")[:32]
        fallback_variants = [
            f"{stem}. Уточняю: фокус на {topic_hint}.",
            f"{stem}. Переформулирую: держим внимание на {topic_hint}.",
            f"{stem}. Конкретика: проверяем {topic_hint}.",
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
        }

    def _fake_memories(self, agent_id: str, limit: int) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for event in reversed(self.state.event_log):
            if event.get("source_id") != agent_id and event.get("target_id") != agent_id:
                continue
            items.append(
                {
                    "event_id": event.get("id"),
                    "text": str(event.get("text", ""))[:140],
                    "tags": list(event.get("tags", []))[:3],
                }
            )
            if len(items) >= limit:
                break
        items.reverse()
        return items

    def _agent_context_for_prompt(self, agent: AgentState) -> dict[str, Any]:
        inbox_items = [
            {
                "source_id": message.get("source_id"),
                "source_type": message.get("source_type"),
                "text": str(message.get("text", ""))[:140],
                "tags": list(message.get("tags", []))[:3],
            }
            for message in agent.inbox[-self.prompt_inbox_limit :]
        ]

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

        return {
            "agent_id": agent.id,
            "state": {
                "mood": agent.mood,
                "mood_label": agent.mood_label,
                "plan": agent.current_plan,
                "pos": {"x": round(agent.pos.x, 2), "z": round(agent.pos.z, 2)},
            },
            "recent_events": [
                self._event_prompt_item(event)
                for event in list(self.state.event_log)[-self.prompt_recent_events_limit :]
            ],
            "inbox": inbox_items,
            "memories_top": self._fake_memories(agent.id, self.prompt_memories_limit),
            "others": others,
        }

    def _world_summary_for_prompt(self) -> dict[str, Any]:
        return {
            "tick": self.state.tick,
            "agents_count": len(self.state.agents),
            "recent_events": [
                self._event_prompt_item(event)
                for event in list(self.state.event_log)[-self.prompt_recent_events_limit :]
            ],
        }

    def _llm_agent_batch_ids(self) -> list[str]:
        if not self.llm_decider.enabled:
            return []

        ordered = sorted(self.state.agents.keys())
        if not ordered:
            return []
        count = min(self.llm_decider.max_agents_per_tick, len(ordered))
        start_idx = (self.state.tick - 1) % len(ordered)
        return [ordered[(start_idx + idx) % len(ordered)] for idx in range(count)]

    def _llm_decisions_for_tick(self) -> dict[str, AgentDecision]:
        agent_ids = self._llm_agent_batch_ids()
        if not agent_ids:
            return {}

        contexts = [self._agent_context_for_prompt(self.state.agents[agent_id]) for agent_id in agent_ids]
        return self.llm_decider.decide(
            tick=self.state.tick,
            world_summary=self._world_summary_for_prompt(),
            agents_context=contexts,
            expected_agent_ids=agent_ids,
        )

    def _build_agents(self) -> dict[str, AgentState]:
        base = [
            ("a1", "Скебоб", "эмпатичный, любопытный", "#f4a261", -10),
            ("a2", "Бобекс", "загадочный, решительный", "#2a9d8f", 5),
            ("a3", "Скебоб Дьявол", "злой, импульсивный", "#e76f51", -20),
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
            text="Загрузка мира завершена. Агентный мозг v0 активен.",
            tags=["system", "world"],
        )
        self._append_event(
            source_type="world",
            source_id=None,
            text="Каждый тик: reflect -> goal -> act. Поведение предсказуемо и наблюдаемо.",
            tags=["system", "world"],
        )

    def _remember_event(self, event_id: str) -> None:
        for agent in self.state.agents.values():
            agent.memory_short.append(event_id)
            if len(agent.memory_short) > self.memory_short_limit:
                agent.memory_short.pop(0)

    def _append_event(
        self,
        source_type: str,
        source_id: str | None,
        text: str,
        tags: list[str],
        target_id: str | None = None,
    ) -> dict:
        self.state.next_event_id += 1
        event = {
            "id": f"e{self.state.next_event_id}",
            "ts": _utc_iso(),
            "source_type": source_type,
            "source_id": source_id,
            "text": text,
            "tags": tags,
        }
        if target_id is not None:
            event["target_id"] = target_id

        self.state.event_log.append(event)
        self._remember_event(event["id"])
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
        target.inbox.append(
            {
                "source_type": source_type,
                "source_id": source_id,
                "source_name": source_name,
                "text": text,
                "tags": list(tags),
                "received_tick": self.state.tick,
                "penalized": False,
            }
        )

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
            return ActionIntent(kind="idle")

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

            return ActionIntent(kind="move", target_id=target_id, destination=destination)

        if decision.act == "say":
            if agent.say_cooldown > 0:
                return None
            text = (decision.say_text or "").strip()
            if not text:
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
            )

        if decision.act == "message":
            if agent.message_cooldown > 0:
                return None
            target_id = decision.target_id
            text = (decision.say_text or "").strip()
            if (
                not target_id
                or target_id not in self.state.agents
                or target_id == agent.id
                or not text
            ):
                return None
            return ActionIntent(
                kind="message",
                text=text,
                tags=["agent_message", "dialogue", "llm"],
                target_id=target_id,
            )

        return None

    def _apply_llm_deltas(self, agent: AgentState, decision: AgentDecision) -> bool:
        if not decision.deltas:
            return False

        # LLM may only suggest tiny shifts; server clips harder before applying.
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

    def _should_consume_inbox(self, context, intent: ActionIntent) -> bool:
        if not context.direct_messages:
            return False
        if intent.kind not in {"say", "message"}:
            return False

        latest_message = context.direct_messages[-1]
        source_id = latest_message.get("source_id")
        if intent.kind == "message" and source_id and intent.target_id and intent.target_id != source_id:
            return False
        return True

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
    ) -> tuple[list[dict], bool]:
        apply_ignored_inbox_penalty(
            relations=self.state.relations,
            inbox=agent.inbox,
            owner_agent_id=agent.id,
            now_tick=self.state.tick,
        )

        recent_events = list(self.state.event_log)[-self.recent_window :]
        all_agent_ids = sorted(self.state.agents.keys())
        traits = parse_traits(agent.traits)
        context = reflect(
            agent=agent,
            tick=self.state.tick,
            recent_events=recent_events,
            relations=self.state.relations,
            all_agent_ids=all_agent_ids,
        )

        if context.mood_shift != 0:
            agent.mood = _clamp_int(agent.mood + context.mood_shift, -100, 100)

        fallback_goal = choose_goal(agent=agent, context=context, traits=traits)
        fallback_target_id = (
            context.target_id if fallback_goal in {GOAL_RESPOND, GOAL_HELP} else context.best_friend_id
        )
        fallback_intent = self._goal_to_intent(agent, fallback_goal, context, traits)

        goal = fallback_goal
        intent = fallback_intent
        agent.current_plan = goal_to_plan(fallback_goal, target_name=self._agent_name(fallback_target_id))
        agent.target_id = fallback_target_id
        used_llm = False

        if llm_decision is not None:
            llm_intent = self._intent_from_llm_decision(agent, llm_decision)
            if llm_intent is not None:
                used_llm = True
                goal = llm_decision.goal.strip() or fallback_goal
                intent = llm_intent
                agent.current_plan = goal
                agent.target_id = llm_intent.target_id

        event = self._execute_intent(agent, intent)

        # Message is considered processed only after successful response action.
        if event is not None and agent.inbox and (goal == GOAL_RESPOND or self._should_consume_inbox(context, intent)):
            processed = agent.inbox.pop(-1)
            agent.last_topic = processed.get("text", "")[:60]
            agent.last_interaction_tick = self.state.tick

        llm_relations_changed = used_llm and llm_decision is not None and self._apply_llm_deltas(agent, llm_decision)
        return ([event] if event is not None else []), llm_relations_changed

    def _goal_to_intent(self, agent: AgentState, goal: str, context, traits) -> ActionIntent:
        intent = act(
            agent=agent,
            goal=goal,
            context=context,
            traits=traits,
            tick=self.state.tick,
        )
        # Panic goal prefers moving to safe center when no explicit destination.
        if goal == GOAL_PANIC and intent.kind == "move" and intent.destination is None:
            intent.destination = SAFE_POINT
        return intent

    def _immediate_world_reactions(self, text: str, sentiment: int, count: int) -> list[dict]:
        reactions: list[dict] = []
        ordered_agents = self._ordered_agents()
        if not ordered_agents:
            return reactions

        seed = sum(ord(ch) for ch in text) + self.state.tick
        for idx in range(count):
            speaker = ordered_agents[(seed + idx) % len(ordered_agents)]
            friend_id = pick_best_friend(self.state.relations, speaker.id, sorted(self.state.agents.keys()))
            selector = seed + idx

            if sentiment < 0 and parse_traits(speaker.traits).courage < 55:
                text_line = templates.render("panic", selector)
                tags = ["dialogue", "world", "conflict"]
                text_kind = "panic"
            else:
                text_line = templates.render("support", selector)
                tags = ["dialogue", "world", "help"]
                text_kind = "support"

            text_line = self._dedupe_dialogue_text(
                agent=speaker,
                text=text_line,
                tags=tags,
                target_id=friend_id,
                text_kind=text_kind,
            )

            event = self._append_event(
                source_type="agent",
                source_id=speaker.id,
                text=text_line,
                tags=tags,
                target_id=friend_id,
            )
            speaker.last_action = "say"
            speaker.last_say = text_line
            speaker.say_cooldown = 2
            speaker.last_interaction_tick = self.state.tick
            update_relations_from_event(self.state.relations, event)
            reactions.append(event)
        return reactions

    def step(self) -> TickResult:
        self.state.tick += 1
        self._decrement_cooldowns()

        llm_decisions = self._llm_decisions_for_tick()
        events: list[dict] = []
        llm_relations_changed = False
        for agent in self._ordered_agents():
            self._apply_passive_mood(agent)
            agent_events, agent_relations_changed = self._run_agent_brain(
                agent=agent,
                llm_decision=llm_decisions.get(agent.id),
            )
            events.extend(agent_events)
            llm_relations_changed = llm_relations_changed or agent_relations_changed

        relations_changed = llm_relations_changed or any(event.get("target_id") is not None for event in events)
        relations_changed = relations_changed or (self.state.tick % self.relations_interval_ticks == 0)
        if relations_changed:
            self._step_relations()

        return TickResult(events=events, relations_changed=relations_changed)

    def agents_state_payload(self) -> list[dict]:
        return [agent.to_state_payload() for agent in self._ordered_agents()]

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
        }

    def events_payload(self, limit: int = 200, agent_id: str | None = None) -> list[dict]:
        items = list(self.state.event_log)
        if agent_id:
            items = [event for event in items if event.get("source_id") == agent_id]
        return items[-max(1, min(limit, 500)) :]

    def update_speed(self, speed: float) -> float:
        self.state.speed = _clamp_float(speed, 0.1, 5.0)
        return self.state.speed

    def add_world_event(self, text: str, importance: float | None = None) -> tuple[dict, list[dict]]:
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
            if sentiment < 0 and parse_traits(agent.traits).courage < 55:
                agent.current_plan = goal_to_plan(GOAL_PANIC)
            elif sentiment < 0:
                agent.current_plan = goal_to_plan(GOAL_HELP)
            elif sentiment > 0:
                agent.current_plan = goal_to_plan(GOAL_SOCIAL)
            else:
                agent.current_plan = goal_to_plan(GOAL_EXPLORE)
            agent.last_topic = text[:60]

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
        target.current_plan = goal_to_plan(GOAL_RESPOND, target_name="user")
        target.target_id = None
        target.last_topic = text[:60]

        selector = self.state.tick + self.state.next_event_id
        if sentiment < 0 and parse_traits(target.traits).aggression > 60:
            reply_text = templates.render("conflict", selector)
            reply_tags = ["dialogue", "reply", "user_message", "conflict"]
            reply_kind = "conflict"
        else:
            reply_text = templates.render("respond_user", selector)
            reply_tags = ["dialogue", "reply", "user_message"]
            reply_kind = "respond_user"
        reply_text = self._dedupe_dialogue_text(
            agent=target,
            text=reply_text,
            tags=reply_tags,
            text_kind=reply_kind,
        )

        reply = self._append_event(
            source_type="agent",
            source_id=target.id,
            text=reply_text,
            tags=reply_tags,
        )

        target.last_action = "say"
        target.last_say = reply_text
        target.say_cooldown = 2
        target.last_interaction_tick = self.state.tick
        if target.inbox:
            target.inbox.pop(-1)
        return event, reply

    def agent_details(self, agent_id: str) -> dict | None:
        agent = self.state.agents.get(agent_id)
        if not agent:
            return None
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
            "memory_short": list(agent.memory_short[-10:]),
            "key_memories": [
                {"text": "короткая память событий активна.", "score": 0.48},
                {"text": f"last topic: {agent.last_topic or 'n/a'}", "score": 0.35},
            ],
            "recent_events": self.events_payload(limit=10, agent_id=agent.id),
        }
