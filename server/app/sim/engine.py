from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from random import Random

from app.agents.agent import AgentState, Vec3, plan_for_mood


def _utc_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


@dataclass
class TickResult:
    event: dict | None
    relations_changed: bool


class StubWorld:
    def __init__(self, seed: int = 42, relations_interval_ticks: int = 5, history_limit: int = 300):
        self.rng = Random(seed)
        self.tick = 0
        self.speed = 1.0
        self.relations_interval_ticks = max(1, relations_interval_ticks)
        self.events = deque(maxlen=history_limit)
        self.event_seq = 0
        self.agents = self._build_agents()
        self.relations = self._build_relations()
        self._seed_initial_events()

    def _build_agents(self) -> dict[str, AgentState]:
        base = [
            ("a1", "Скебоб", "эмпатичный, любопытный", "#f4a261"),
            ("a2", "Бобекс", "загадочный, безумный", "#2a9d8f"),
            ("a3", "Скебоб Дьявол", "злой, конченный", "#e76f51"),
            ("a4", "Скебобиха", "кокетливая, решительная", "#457b9d"),
        ]
        agents: dict[str, AgentState] = {}
        for idx, (agent_id, name, traits, avatar) in enumerate(base):
            mood = self.rng.randint(-25, 35)
            agents[agent_id] = AgentState(
                id=agent_id,
                name=name,
                traits=traits,
                mood=mood,
                avatar=avatar,
                pos=Vec3(x=-8 + idx * 5, z=-3 + idx),
                current_plan="Начинаем день 0.",
            )
            agents[agent_id].current_plan = plan_for_mood(agents[agent_id].mood_label)
        return agents

    def _build_relations(self) -> dict[tuple[str, str], int]:
        rel: dict[tuple[str, str], int] = {}
        ids = list(self.agents.keys())
        for src in ids:
            for dst in ids:
                if src == dst:
                    continue
                rel[(src, dst)] = self.rng.randint(-25, 45)
        return rel

    def _seed_initial_events(self) -> None:
        self._append_event(
            source_type="world",
            source_id=None,
            text="Загрузка мира завершена. Начало симуляции день 0.",
            tags=["system"],
        )
        self._append_event(
            source_type="world",
            source_id=None,
            text="Агенты синхронизированы и готовы к реал-тайм трансляции.",
            tags=["system"],
        )

    def _append_event(self, source_type: str, source_id: str | None, text: str, tags: list[str]) -> dict:
        self.event_seq += 1
        event = {
            "id": f"e{self.event_seq}",
            "ts": _utc_iso(),
            "source_type": source_type,
            "source_id": source_id,
            "text": text,
            "tags": tags,
        }
        self.events.append(event)
        return event

    def _dialogue_event(self) -> dict:
        speaker = self.rng.choice(list(self.agents.values()))
        target = self.rng.choice([a for a in self.agents.values() if a.id != speaker.id])
        lines = [
            f"{speaker.name}: \"Здаровва, {target.name}. Че каво?\"",
            f"{speaker.name}: \"Эщкере.\"",
            f"{speaker.name}: \"ого я крутой.\"",
            f"{speaker.name}: \"вечер в хату?\"",
        ]
        text = self.rng.choice(lines)
        speaker.last_action = "say"
        speaker.last_say = text

        key = (speaker.id, target.id)
        self.relations[key] = _clamp(self.relations[key] + self.rng.randint(-4, 6), -100, 100)
        return self._append_event(
            source_type="agent",
            source_id=speaker.id,
            text=text,
            tags=["dialogue"],
        )

    def _step_agents(self) -> None:
        for agent in self.agents.values():
            agent.pos.x = max(-20.0, min(20.0, agent.pos.x + self.rng.uniform(-1.1, 1.1) * self.speed))
            agent.pos.z = max(-20.0, min(20.0, agent.pos.z + self.rng.uniform(-1.1, 1.1) * self.speed))
            agent.look_at = Vec3(self.rng.uniform(-1, 1), 0.0, self.rng.uniform(-1, 1))
            agent.mood = _clamp(agent.mood + self.rng.randint(-5, 5), -100, 100)
            if self.rng.random() < 0.5:
                agent.current_plan = plan_for_mood(agent.mood_label)
            if agent.last_action != "say":
                agent.last_action = "walk"

    def _step_relations(self) -> None:
        for key, value in self.relations.items():
            delta = self.rng.randint(-2, 2)
            self.relations[key] = _clamp(value + delta, -100, 100)

    def step(self) -> TickResult:
        self.tick += 1
        self._step_agents()

        event = None
        if self.rng.random() < 0.35:
            event = self._dialogue_event()

        relations_changed = self.tick % self.relations_interval_ticks == 0
        if relations_changed:
            self._step_relations()
        return TickResult(event=event, relations_changed=relations_changed)

    def agents_state_payload(self) -> list[dict]:
        return [agent.to_state_payload() for agent in self.agents.values()]

    def relations_payload(self) -> dict:
        return {
            "nodes": [{"id": agent.id, "name": agent.name} for agent in self.agents.values()],
            "edges": [
                {"from": src, "to": dst, "value": value}
                for (src, dst), value in sorted(self.relations.items())
            ],
        }

    def agents_list_payload(self) -> list[dict]:
        return [agent.to_agent_summary() for agent in self.agents.values()]

    def state_payload(self) -> dict:
        return {
            "tick": self.tick,
            "speed": self.speed,
            "agents": self.agents_state_payload(),
            "relations": self.relations_payload(),
            "events": list(self.events)[-200:],
        }

    def events_payload(self, limit: int = 200, agent_id: str | None = None) -> list[dict]:
        items = list(self.events)
        if agent_id:
            items = [event for event in items if event.get("source_id") == agent_id]
        return items[-max(1, min(limit, 500)) :]

    def update_speed(self, speed: float) -> float:
        self.speed = max(0.1, min(speed, 5.0))
        return self.speed

    def add_world_event(self, text: str, importance: float | None = None) -> dict:
        tags = ["world_event"]
        if importance is not None and importance >= 0.7:
            tags.append("important")
        return self._append_event(source_type="world", source_id=None, text=text, tags=tags)

    def add_agent_message(self, agent_id: str, text: str) -> dict:
        if agent_id not in self.agents:
            raise KeyError(agent_id)
        agent = self.agents[agent_id]
        message = f"User -> {agent.name}: {text}"
        agent.last_action = "listen"
        return self._append_event(source_type="world", source_id=None, text=message, tags=["message"])

    def agent_details(self, agent_id: str) -> dict | None:
        agent = self.agents.get(agent_id)
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
                {"text": "меня смотрят.", "score": 0.37},
            ],
            "recent_events": self.events_payload(limit=10, agent_id=agent.id),
        }
