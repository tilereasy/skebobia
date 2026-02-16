from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Deque

from app.agents.agent import AgentState, Vec3, plan_for_mood


def _utc_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _clamp_float(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _text_hash(text: str) -> int:
    return sum(ord(char) for char in text.lower())


def _sentiment_score(text: str) -> int:
    positive_tokens = {
        "help",
        "gift",
        "peace",
        "good",
        "safe",
        "celebrate",
        "спасибо",
        "помоги",
        "подар",
        "мир",
        "добро",
        "рад",
        "поддерж",
    }
    negative_tokens = {
        "attack",
        "conflict",
        "fight",
        "fire",
        "danger",
        "hate",
        "war",
        "угр",
        "пожар",
        "конфликт",
        "драка",
        "ненав",
        "зло",
        "опас",
    }

    normalized = text.lower()
    pos_hits = sum(token in normalized for token in positive_tokens)
    neg_hits = sum(token in normalized for token in negative_tokens)
    score = pos_hits - neg_hits
    if score == 0:
        score = 1 if _text_hash(normalized) % 2 == 0 else -1
    return _clamp_int(score, -2, 2)


@dataclass
class TickResult:
    event: dict | None
    relations_changed: bool


@dataclass
class WorldState:
    agents: dict[str, AgentState]
    relations: dict[tuple[str, str], int]
    event_log: Deque[dict]
    speed: float = 1.0
    tick: int = 0
    next_event_id: int = 0
    pending_reactions: Deque[dict] = field(default_factory=deque)
    forced_plan_until_tick: dict[str, int] = field(default_factory=dict)


class StubWorld:
    def __init__(self, relations_interval_ticks: int = 5, history_limit: int = 300):
        agents = self._build_agents()
        relations = self._build_relations(list(agents.keys()))
        self.relations_interval_ticks = max(1, relations_interval_ticks)
        self.state = WorldState(
            agents=agents,
            relations=relations,
            event_log=deque(maxlen=history_limit),
        )
        self._seed_initial_events()

    @property
    def speed(self) -> float:
        return self.state.speed

    @property
    def world_state(self) -> WorldState:
        return self.state

    def _ordered_agents(self) -> list[AgentState]:
        return [self.state.agents[agent_id] for agent_id in sorted(self.state.agents.keys())]

    def _build_agents(self) -> dict[str, AgentState]:
        base = [
            ("a1", "Скебоб", "эмпатичный, любопытный", "#f4a261", -15),
            ("a2", "Бобекс", "загадочный, безумный", "#2a9d8f", 5),
            ("a3", "Скебоб Дьявол", "злой, импульсивный", "#e76f51", -5),
            ("a4", "Скебобиха", "кокетливая, решительная", "#457b9d", 20),
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
                current_plan="Инициализация симуляции",
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
                base = 35 - abs(src_idx - dst_idx) * 18 + (dst_idx - src_idx) * 4
                relations[(src, dst)] = _clamp_int(base, -100, 100)
        return relations

    def _seed_initial_events(self) -> None:
        self._append_event(
            source_type="world",
            source_id=None,
            text="Загрузка мира завершена. Детерминированная симуляция активна.",
            tags=["system"],
        )
        self._append_event(
            source_type="world",
            source_id=None,
            text="Агенты синхронизированы. Любое действие пользователя влияет на состояние мира.",
            tags=["system"],
        )

    def _append_event(self, source_type: str, source_id: str | None, text: str, tags: list[str]) -> dict:
        self.state.next_event_id += 1
        event = {
            "id": f"e{self.state.next_event_id}",
            "ts": _utc_iso(),
            "source_type": source_type,
            "source_id": source_id,
            "text": text,
            "tags": tags,
        }
        self.state.event_log.append(event)
        return event

    def _set_forced_plan(self, agent_id: str, text: str, ttl_ticks: int) -> None:
        if agent_id not in self.state.agents:
            return
        self.state.agents[agent_id].current_plan = text
        self.state.forced_plan_until_tick[agent_id] = self.state.tick + max(1, ttl_ticks)

    def _apply_world_influence(self, text: str, importance: float | None) -> None:
        score = _sentiment_score(text)
        intensity = max(4, int(round((importance if importance is not None else 0.5) * 20)))
        mood_delta = score * max(3, intensity // 3)
        relation_delta = score * max(2, intensity // 4)

        for idx, agent in enumerate(self._ordered_agents()):
            spread = idx if mood_delta < 0 else -idx
            agent.mood = _clamp_int(agent.mood + mood_delta + spread, -100, 100)
            self._set_forced_plan(agent.id, f"реагирует на событие: {text[:42]}", ttl_ticks=3)

        for edge_key, edge_value in list(self.state.relations.items()):
            self.state.relations[edge_key] = _clamp_int(edge_value + relation_delta, -100, 100)

        self.state.pending_reactions.append(
            {
                "kind": "world_event",
                "text": text,
                "score": score,
            }
        )

    def _apply_message_influence(self, agent_id: str, text: str) -> None:
        score = _sentiment_score(text)
        focus_delta = score * 12
        crowd_delta = score * 2
        relation_delta = score * 7

        target = self.state.agents[agent_id]
        target.mood = _clamp_int(target.mood + focus_delta, -100, 100)
        target.last_action = "listen"
        self._set_forced_plan(target.id, f"обрабатывает сообщение: {text[:42]}", ttl_ticks=3)

        for agent in self._ordered_agents():
            if agent.id == agent_id:
                continue
            agent.mood = _clamp_int(agent.mood + crowd_delta, -100, 100)
            forward = (agent_id, agent.id)
            backward = (agent.id, agent_id)
            self.state.relations[forward] = _clamp_int(self.state.relations[forward] + relation_delta, -100, 100)
            self.state.relations[backward] = _clamp_int(
                self.state.relations[backward] + relation_delta // 2,
                -100,
                100,
            )

        self.state.pending_reactions.append(
            {
                "kind": "message",
                "agent_id": agent_id,
                "text": text,
                "score": score,
            }
        )

    def _step_agents(self) -> None:
        dx_pattern = (-0.4, -0.2, 0.0, 0.2, 0.4, 0.0)
        dz_pattern = (0.3, 0.1, -0.1, -0.3, 0.0, 0.2)
        mood_pattern = (-1, 0, 1, 0, 0)

        for idx, agent in enumerate(self._ordered_agents()):
            phase = self.state.tick + idx * 2
            dx = dx_pattern[phase % len(dx_pattern)] * max(0.4, self.state.speed)
            dz = dz_pattern[(phase + idx) % len(dz_pattern)] * max(0.4, self.state.speed)

            agent.pos.x = _clamp_float(agent.pos.x + dx, -20.0, 20.0)
            agent.pos.z = _clamp_float(agent.pos.z + dz, -20.0, 20.0)
            agent.look_at = Vec3(dx, 0.0, dz if abs(dz) > 1e-6 else 1.0)
            agent.mood = _clamp_int(agent.mood + mood_pattern[(phase + idx) % len(mood_pattern)], -100, 100)

            forced_until = self.state.forced_plan_until_tick.get(agent.id, -1)
            if self.state.tick >= forced_until:
                agent.current_plan = plan_for_mood(agent.mood_label, selector=self.state.tick + idx)

            if agent.last_action in {"say", "listen"}:
                agent.last_action = "walk"
            elif agent.last_action != "walk":
                agent.last_action = "walk"

    def _reaction_event(self) -> dict | None:
        if not self.state.pending_reactions:
            return None

        reaction = self.state.pending_reactions.popleft()
        ordered_agents = self._ordered_agents()
        if not ordered_agents:
            return None

        if reaction["kind"] == "message":
            speaker = self.state.agents[reaction["agent_id"]]
            tone = "принял" if reaction["score"] > 0 else "напрягся"
            text = f'{speaker.name}: "{tone}: {reaction["text"][:60]}"'
            tags = ["dialogue", "reaction", "message"]
        else:
            speaker = ordered_agents[self.state.tick % len(ordered_agents)]
            tone = "поддерживаю" if reaction["score"] > 0 else "тревожно"
            text = f'{speaker.name}: "{tone}: {reaction["text"][:60]}"'
            tags = ["dialogue", "reaction", "world_event"]

        speaker.last_action = "say"
        speaker.last_say = text
        return self._append_event(source_type="agent", source_id=speaker.id, text=text, tags=tags)

    def _routine_dialogue_event(self) -> dict | None:
        if self.state.tick % 3 != 0:
            return None

        ordered_agents = self._ordered_agents()
        if len(ordered_agents) < 2:
            return None

        speaker = ordered_agents[self.state.tick % len(ordered_agents)]
        target = ordered_agents[(self.state.tick + 1) % len(ordered_agents)]
        text = f'{speaker.name}: "План: {speaker.current_plan}. {target.name}, держим курс."'
        speaker.last_action = "say"
        speaker.last_say = text

        pair = (speaker.id, target.id)
        alignment = 1 if (speaker.mood + target.mood) >= 0 else -1
        self.state.relations[pair] = _clamp_int(self.state.relations[pair] + alignment, -100, 100)

        return self._append_event(
            source_type="agent",
            source_id=speaker.id,
            text=text,
            tags=["dialogue"],
        )

    def _step_relations(self) -> None:
        for src_id, src_agent in self.state.agents.items():
            for dst_id, dst_agent in self.state.agents.items():
                if src_id == dst_id:
                    continue
                key = (src_id, dst_id)
                current = self.state.relations[key]
                mood_alignment = 1 if (src_agent.mood + dst_agent.mood) >= 0 else -1
                plan_alignment = 1 if src_agent.current_plan[:12] == dst_agent.current_plan[:12] else 0
                self.state.relations[key] = _clamp_int(current + mood_alignment + plan_alignment, -100, 100)

    def step(self) -> TickResult:
        self.state.tick += 1
        self._step_agents()

        event = self._reaction_event()
        if event is None:
            event = self._routine_dialogue_event()

        relations_changed = self.state.tick % self.relations_interval_ticks == 0
        if relations_changed:
            self._step_relations()

        return TickResult(event=event, relations_changed=relations_changed)

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

    def add_world_event(self, text: str, importance: float | None = None) -> dict:
        tags = ["world_event"]
        if importance is not None and importance >= 0.7:
            tags.append("important")

        event = self._append_event(source_type="world", source_id=None, text=text, tags=tags)
        self._apply_world_influence(text=text, importance=importance)
        return event

    def add_agent_message(self, agent_id: str, text: str) -> dict:
        if agent_id not in self.state.agents:
            raise KeyError(agent_id)

        agent = self.state.agents[agent_id]
        message = f"User -> {agent.name}: {text}"
        event = self._append_event(source_type="world", source_id=None, text=message, tags=["message"])
        self._apply_message_influence(agent_id=agent_id, text=text)
        return event

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
            "key_memories": [
                {"text": "день 0 в памяти.", "score": 0.42},
                {"text": "мир детерминирован и управляем.", "score": 0.37},
            ],
            "recent_events": self.events_payload(limit=10, agent_id=agent.id),
        }
