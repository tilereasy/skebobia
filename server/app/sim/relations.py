from __future__ import annotations


def clamp(value: int, low: int = -100, high: int = 100) -> int:
    return max(low, min(high, value))


def _delta_by_tags(tags: list[str]) -> int:
    tags_set = set(tags)
    if "conflict" in tags_set:
        return -5
    if "help" in tags_set:
        return 3
    if "agent_message" in tags_set or "dialogue" in tags_set:
        return 1
    return 0


def update_relations_from_event(relations: dict[tuple[str, str], int], event: dict) -> None:
    source_id = event.get("source_id")
    target_id = event.get("target_id")
    if not source_id or not target_id or source_id == target_id:
        return
    key_forward = (source_id, target_id)
    key_backward = (target_id, source_id)
    if key_forward not in relations or key_backward not in relations:
        return

    delta = _delta_by_tags(event.get("tags", []))
    if delta == 0:
        return

    relations[key_forward] = clamp(relations[key_forward] + delta)
    relations[key_backward] = clamp(relations[key_backward] + (delta - 1 if delta > 0 else delta))


def apply_ignored_inbox_penalty(
    relations: dict[tuple[str, str], int],
    inbox: list[dict],
    owner_agent_id: str,
    now_tick: int,
    ignore_after_ticks: int = 5,
) -> None:
    for message in inbox:
        source_id = message.get("source_id")
        if not source_id or source_id == owner_agent_id:
            continue
        if message.get("penalized"):
            continue
        received_tick = int(message.get("received_tick", now_tick))
        if now_tick - received_tick < ignore_after_ticks:
            continue

        key = (source_id, owner_agent_id)
        if key in relations:
            relations[key] = clamp(relations[key] - 1)
        message["penalized"] = True
