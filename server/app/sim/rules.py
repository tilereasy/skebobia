from __future__ import annotations

from dataclasses import dataclass, field

from app.agents.agent import AgentState
from app.sim import templates
from app.sim.movement import SAFE_POINT, Vec3, pick_wander_target


NEGATIVE_KEYWORDS = {"пожар", "авар", "опас", "угр", "конфликт", "драка", "атака", "ненав"}
POSITIVE_KEYWORDS = {"еда", "награ", "спасибо", "класс", "help", "peace", "добро", "поддерж"}


@dataclass
class TraitProfile:
    aggression: int = 45
    sociability: int = 50
    curiosity: int = 50
    courage: int = 50


@dataclass
class ReflectContext:
    direct_messages: list[dict]
    world_events: list[dict]
    danger_events: list[dict]
    mood_shift: int
    topic: str
    target_id: str | None
    best_friend_id: str | None
    emit_memory_event: bool
    recent_events: list[dict] = field(default_factory=list)


@dataclass
class ActionIntent:
    kind: str
    text: str = ""
    tags: list[str] = field(default_factory=list)
    target_id: str | None = None
    destination: Vec3 | None = None
    text_kind: str | None = None


GOAL_RESPOND = "RespondToMessage"
GOAL_HELP = "HelpOther"
GOAL_COMFORT = "SeekComfort"
GOAL_SOCIAL = "Socialize"
GOAL_EXPLORE = "Explore"
GOAL_PANIC = "Panic"


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _selector_seed(agent: AgentState, tick: int, context: ReflectContext, traits: TraitProfile) -> int:
    topic_source = (agent.last_topic or context.topic)[:24]
    topic_weight = sum(ord(ch) for ch in topic_source)
    return (
        tick * 97
        + sum(ord(ch) for ch in agent.id) * 17
        + agent.mood * 13
        + len(context.direct_messages) * 29
        + len(context.danger_events) * 31
        + len(context.world_events) * 19
        + traits.curiosity * 7
        + topic_weight
    )


def text_sentiment(text: str) -> int:
    normalized = text.lower()
    pos = sum(token in normalized for token in POSITIVE_KEYWORDS)
    neg = sum(token in normalized for token in NEGATIVE_KEYWORDS)
    score = pos - neg
    if score == 0:
        score = 1 if (sum(ord(ch) for ch in normalized) % 2 == 0) else -1
    return _clamp(score, -2, 2)


def parse_traits(traits: str) -> TraitProfile:
    normalized = traits.lower()
    profile = TraitProfile()

    if "злой" in normalized or "агрессив" in normalized:
        profile.aggression += 25
        profile.courage += 10
    if "эмпат" in normalized or "support" in normalized:
        profile.sociability += 20
        profile.aggression -= 10
    if "любопыт" in normalized or "curious" in normalized:
        profile.curiosity += 20
    if "безум" in normalized:
        profile.courage += 15
        profile.aggression += 10
    if "решитель" in normalized:
        profile.courage += 15
        profile.sociability += 5

    profile.aggression = _clamp(profile.aggression, 0, 100)
    profile.sociability = _clamp(profile.sociability, 0, 100)
    profile.curiosity = _clamp(profile.curiosity, 0, 100)
    profile.courage = _clamp(profile.courage, 0, 100)
    return profile


def pick_best_friend(
    relations: dict[tuple[str, str], int],
    agent_id: str,
    all_agent_ids: list[str],
) -> str | None:
    best_id: str | None = None
    best_score = -10_000
    for other_id in all_agent_ids:
        if other_id == agent_id:
            continue
        score = relations.get((agent_id, other_id), -100)
        if score > best_score:
            best_score = score
            best_id = other_id
    return best_id


def _pick_topic(messages: list[dict], world_events: list[dict]) -> str:
    source = messages[-1]["text"] if messages else (world_events[-1]["text"] if world_events else "")
    source = source.strip()
    if not source:
        return "последние события"
    return source[:40]


def reflect(
    agent: AgentState,
    tick: int,
    recent_events: list[dict],
    relations: dict[tuple[str, str], int],
    all_agent_ids: list[str],
) -> ReflectContext:
    direct_messages = list(agent.inbox)
    world_events = [event for event in recent_events if "world" in event.get("tags", [])][-6:]
    danger_events = [
        event for event in world_events if text_sentiment(event.get("text", "")) < 0 or "conflict" in event.get("tags", [])
    ][-2:]

    mood_shift = 0
    for message in direct_messages:
        score = text_sentiment(message.get("text", ""))
        mood_shift += 5 if score > 0 else -5

    for event in danger_events:
        _ = event
        mood_shift -= 4

    if tick - agent.last_interaction_tick >= 4 and not direct_messages:
        mood_shift -= 2

    mood_shift = _clamp(mood_shift, -20, 20)
    topic = _pick_topic(direct_messages, world_events)
    target_id = direct_messages[-1].get("source_id") if direct_messages else None
    best_friend_id = pick_best_friend(relations, agent.id, all_agent_ids)
    emit_memory_event = tick % 6 == (sum(ord(ch) for ch in agent.id) % 6)

    return ReflectContext(
        direct_messages=direct_messages,
        world_events=world_events,
        danger_events=danger_events,
        mood_shift=mood_shift,
        topic=topic,
        target_id=target_id,
        best_friend_id=best_friend_id,
        emit_memory_event=emit_memory_event,
        recent_events=recent_events,
    )


def choose_goal(agent: AgentState, context: ReflectContext, traits: TraitProfile) -> str:
    if context.direct_messages:
        return GOAL_RESPOND
    if context.danger_events:
        if traits.courage >= 55:
            return GOAL_HELP
        return GOAL_PANIC
    if agent.mood < -40:
        return GOAL_COMFORT
    if agent.mood > 40 and traits.sociability >= 45:
        return GOAL_SOCIAL
    if traits.curiosity >= 55:
        return GOAL_EXPLORE
    return GOAL_EXPLORE


def goal_to_plan(goal: str, target_name: str | None = None) -> str:
    if goal == GOAL_RESPOND:
        return "Respond to user" if not target_name else f"Respond to {target_name}"
    if goal == GOAL_HELP:
        return f"Help {target_name}" if target_name else "Help nearby agent"
    if goal == GOAL_COMFORT:
        return "Calm down and find someone friendly"
    if goal == GOAL_SOCIAL:
        return f"Check on {target_name}" if target_name else "Share a positive update"
    if goal == GOAL_PANIC:
        return "Move to a safer place"
    return "Walk around and observe"


def act(
    agent: AgentState,
    goal: str,
    context: ReflectContext,
    traits: TraitProfile,
    tick: int,
) -> ActionIntent:
    selector = _selector_seed(agent=agent, tick=tick, context=context, traits=traits)

    if goal == GOAL_RESPOND and context.direct_messages:
        latest = context.direct_messages[-1]
        source_id = latest.get("source_id")
        source_type = latest.get("source_type")
        source_name = latest.get("source_name") or source_id or "друг"
        topic = latest.get("text", "")[:45]
        agent.last_topic = topic
        if source_type == "agent" and source_id and agent.message_cooldown == 0:
            text = templates.render("respond_agent", selector, target_name=source_name, topic=topic)
            return ActionIntent(
                kind="message",
                text=text,
                tags=["agent_message", "dialogue", "reply"],
                target_id=source_id,
                text_kind="respond_agent",
            )
        text = templates.render("respond_user", selector)
        return ActionIntent(kind="say", text=text, tags=["dialogue", "user_message"], text_kind="respond_user")

    if goal == GOAL_HELP:
        if context.target_id:
            return ActionIntent(kind="move", target_id=context.target_id)
        if agent.say_cooldown == 0:
            text = templates.render("support", selector)
            return ActionIntent(kind="say", text=text, tags=["dialogue", "help"], text_kind="support")
        return ActionIntent(kind="move", destination=SAFE_POINT)

    if goal == GOAL_COMFORT:
        if context.best_friend_id:
            return ActionIntent(kind="move", target_id=context.best_friend_id)
        if agent.say_cooldown == 0:
            text = templates.render("panic", selector)
            return ActionIntent(kind="say", text=text, tags=["dialogue", "memory"], text_kind="panic")
        return ActionIntent(kind="idle")

    if goal == GOAL_SOCIAL:
        if agent.say_cooldown == 0:
            text = templates.render("support", selector)
            return ActionIntent(kind="say", text=text, tags=["dialogue", "help"], text_kind="support")
        if context.best_friend_id:
            return ActionIntent(kind="move", target_id=context.best_friend_id)
        return ActionIntent(kind="move", destination=pick_wander_target(agent.id, tick))

    if goal == GOAL_PANIC:
        if agent.say_cooldown == 0 and traits.aggression < 65:
            text = templates.render("panic", selector)
            return ActionIntent(kind="say", text=text, tags=["dialogue", "conflict"], text_kind="panic")
        return ActionIntent(kind="move", destination=SAFE_POINT)

    if context.emit_memory_event and agent.say_cooldown == 0:
        text = templates.render("memory", selector, name=agent.name, topic=context.topic)
        return ActionIntent(kind="say", text=text, tags=["memory"], text_kind="memory")

    if agent.say_cooldown == 0 and tick % 5 == 0 and traits.sociability > 45:
        text = templates.render("explore", selector)
        return ActionIntent(kind="say", text=text, tags=["dialogue"], text_kind="explore")

    return ActionIntent(kind="move", destination=pick_wander_target(agent.id, tick))
