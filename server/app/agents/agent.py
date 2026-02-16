from __future__ import annotations

from dataclasses import dataclass, field
from random import choice


@dataclass
class Vec3:
    x: float
    y: float = 0.0
    z: float = 0.0

    def to_dict(self) -> dict:
        return {"x": round(self.x, 2), "y": round(self.y, 2), "z": round(self.z, 2)}


def mood_label(mood: int) -> str:
    if mood <= -60:
        return "angry"
    if mood <= -20:
        return "sad"
    if mood < 20:
        return "neutral"
    if mood < 60:
        return "happy"
    return "excited"


def plan_for_mood(label: str) -> str:
    plans = {
        "angry": ["пойду охлажу свое трахание", "я уже не скебоб я зверь нахер пойду спать"],
        "sad": ["пойду поною кому нибудь", "гуляю как дед инсайд"],
        "neutral": ["гуляю", "ищу подарки"],
        "happy": ["хочет по доброму поговорить", "приветвтовать соседей"],
        "excited": ["рассказать всем хорошие новости", "двигается быстро"],
    }
    return choice(plans[label])


@dataclass
class AgentState:
    id: str
    name: str
    traits: str
    mood: int
    avatar: str
    pos: Vec3
    look_at: Vec3 = field(default_factory=lambda: Vec3(0.0, 0.0, 1.0))
    current_plan: str = "наблюдать мир"
    last_action: str = "idle"
    last_say: str = ""

    @property
    def mood_label(self) -> str:
        return mood_label(self.mood)

    def to_state_payload(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "mood": self.mood,
            "mood_label": self.mood_label,
            "current_plan": self.current_plan,
            "pos": self.pos.to_dict(),
            "look_at": self.look_at.to_dict(),
            "last_action": self.last_action,
            "last_say": self.last_say,
        }

    def to_agent_summary(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "avatar": self.avatar,
            "mood": self.mood,
            "mood_label": self.mood_label,
            "current_plan": self.current_plan,
        }
