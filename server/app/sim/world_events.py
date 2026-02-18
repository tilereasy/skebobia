from __future__ import annotations

import logging
import os
import re
from collections import Counter, deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from app.llm.client import LLMClient


def _clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _clamp_float(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, low: int, high: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return _clamp_int(default, low, high)
    try:
        value = int(raw.strip())
    except ValueError:
        value = default
    return _clamp_int(value, low, high)


def _env_int_range(name: str, default_low: int, default_high: int, low: int, high: int) -> tuple[int, int]:
    raw = os.getenv(name, f"{default_low}..{default_high}").strip()
    if ".." in raw:
        left, right = raw.split("..", 1)
        try:
            lo = int(left.strip())
            hi = int(right.strip())
        except ValueError:
            lo, hi = default_low, default_high
    else:
        try:
            value = int(raw)
        except ValueError:
            value = default_low
        lo = value
        hi = value

    lo = _clamp_int(lo, low, high)
    hi = _clamp_int(hi, low, high)
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _normalize_text(text: str) -> str:
    lowered = str(text).lower()
    cleaned = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in lowered)
    return " ".join(cleaned.split())


def _tokenize(text: str) -> list[str]:
    normalized = _normalize_text(text)
    return [token for token in normalized.split() if len(token) >= 3]


@dataclass
class GeneratedWorldEvent:
    text: str
    anchor: str
    severity: str
    importance: float
    tags: list[str]


class WorldEventGenerator:
    # near_train пока намеренно пропущен: добавим после финализации координат.
    DEFAULT_ANCHORS = (
        "tracks",
        "path",
        "vending_machines",
        "bench",
        "stone_circle",
        "signpost",
        "bushes_right",
    )

    _SEVERITY_VALUES = {"micro", "normal", "danger"}
    _META_BLACKLIST = (
        "симуляц",
        "коммуникац",
        "план",
        "ии",
        "искусственный интеллект",
        "ai",
        "assistant",
        "model",
        "prompt",
        "tool",
    )
    _DIRECTIVE_PATTERNS = (
        re.compile(r"\bвы\s+должны\b", re.IGNORECASE),
        re.compile(r"\bвам\s+нужно\b", re.IGNORECASE),
        re.compile(r"\bобязаны\b", re.IGNORECASE),
        re.compile(r"\byou\s+must\b", re.IGNORECASE),
        re.compile(r"\bagents?\s+must\b", re.IGNORECASE),
    )
    _STOPWORDS = {
        "это",
        "этот",
        "эта",
        "или",
        "как",
        "вот",
        "где",
        "там",
        "тут",
        "чтобы",
        "когда",
        "около",
        "между",
        "после",
        "снова",
        "очень",
        "просто",
        "который",
        "которая",
        "которые",
        "будет",
        "были",
        "быть",
        "есть",
        "его",
        "ее",
        "них",
        "для",
        "with",
        "from",
        "this",
        "that",
        "have",
        "been",
        "were",
        "your",
        "about",
    }

    def __init__(
        self,
        *,
        client: LLMClient,
        enabled: bool,
        anchors: tuple[str, ...],
        recent_history_limit: int,
        min_interval_range: tuple[int, int],
        max_per_50_range: tuple[int, int],
        danger_cooldown_range: tuple[int, int],
        silence_window_ticks: int,
        repeat_ratio_threshold: float,
    ):
        self.client = client
        self.enabled_by_config = enabled
        self.anchors = anchors
        self.anchor_set = set(anchors)
        self.recent_world_events: deque[dict[str, Any]] = deque(maxlen=max(10, recent_history_limit))
        self.recent_emit_ticks: deque[int] = deque(maxlen=400)
        self.recent_danger_ticks: deque[int] = deque(maxlen=200)
        self.seen_event_ids: set[str] = set()
        self.seen_event_order: deque[str] = deque(maxlen=2048)

        self.min_interval_range = min_interval_range
        self.max_per_50_range = max_per_50_range
        self.danger_cooldown_range = danger_cooldown_range
        self.silence_window_ticks = max(4, silence_window_ticks)
        self.repeat_ratio_threshold = _clamp_float(repeat_ratio_threshold, 0.1, 0.95)

        self.last_emit_tick = -10_000
        self.last_should_emit_reason = ""
        self.last_reject_reason = ""
        self.rerolls = 0
        self.blocked_budget = 0
        self.blocked_guardrails = 0
        self.blocked_repeats = 0
        self.validation_failures = 0

        self.logger = logging.getLogger("app.sim.world_events")

    @classmethod
    def from_env(cls, client: LLMClient | None = None) -> "WorldEventGenerator":
        llm_client = client or LLMClient.from_env()
        enabled = _env_bool("WORLD_EVENTS_ENABLED", True)
        min_interval_range = _env_int_range("WORLD_EVENTS_MIN_INTERVAL_TICKS", 8, 12, 1, 240)
        max_per_50_range = _env_int_range("WORLD_EVENTS_MAX_PER_50_TICKS", 6, 10, 1, 50)
        danger_cooldown_range = _env_int_range("WORLD_EVENTS_DANGER_COOLDOWN_TICKS", 40, 80, 5, 500)
        history_limit = _env_int("WORLD_EVENTS_RECENT_HISTORY", 30, 10, 150)
        silence_window_ticks = _env_int("WORLD_EVENTS_SILENCE_WINDOW_TICKS", 12, 4, 80)
        try:
            repeat_ratio_threshold = float(os.getenv("WORLD_EVENTS_REPEAT_RATIO_THRESHOLD", "0.45"))
        except ValueError:
            repeat_ratio_threshold = 0.45

        return cls(
            client=llm_client,
            enabled=enabled,
            anchors=tuple(cls.DEFAULT_ANCHORS),
            recent_history_limit=history_limit,
            min_interval_range=min_interval_range,
            max_per_50_range=max_per_50_range,
            danger_cooldown_range=danger_cooldown_range,
            silence_window_ticks=silence_window_ticks,
            repeat_ratio_threshold=repeat_ratio_threshold,
        )

    @property
    def enabled(self) -> bool:
        return self.enabled_by_config and self.client.enabled

    def _value_from_range(self, interval_range: tuple[int, int], tick: int, salt: int) -> int:
        low, high = interval_range
        if low >= high:
            return low
        span = high - low + 1
        return low + abs((tick * 131 + salt * 17) % span)

    def _min_interval_ticks(self, tick: int) -> int:
        return self._value_from_range(self.min_interval_range, tick, salt=3)

    def _max_per_50_ticks(self, tick: int) -> int:
        return self._value_from_range(self.max_per_50_range, tick, salt=11)

    def _danger_cooldown_ticks(self, tick: int) -> int:
        return self._value_from_range(self.danger_cooldown_range, tick, salt=29)

    def _is_world_event(self, event: Mapping[str, Any]) -> bool:
        if str(event.get("source_type", "")).lower() != "world":
            return False
        tags = {str(tag).lower() for tag in event.get("tags", [])}
        if "user_message" in tags:
            return False
        if "system" in tags:
            return False
        if "llm_trace" in tags:
            return False
        return True

    def _is_dialogue_event(self, event: Mapping[str, Any]) -> bool:
        tags = {str(tag).lower() for tag in event.get("tags", [])}
        if tags & {"dialogue", "agent_message", "reply"}:
            return True
        return str(event.get("source_type", "")).lower() == "agent"

    def _anchor_from_event(self, event: Mapping[str, Any]) -> str | None:
        direct = event.get("anchor")
        if isinstance(direct, str) and direct in self.anchor_set:
            return direct
        for tag in event.get("tags", []):
            if not isinstance(tag, str):
                continue
            if tag.startswith("anchor:"):
                anchor = tag.split(":", 1)[1].strip()
                if anchor in self.anchor_set:
                    return anchor
        return None

    def _severity_from_event(self, event: Mapping[str, Any]) -> str:
        direct = event.get("severity")
        if isinstance(direct, str) and direct in self._SEVERITY_VALUES:
            return direct
        tags = {str(tag).lower() for tag in event.get("tags", [])}
        if "danger" in tags:
            return "danger"
        if "normal" in tags:
            return "normal"
        return "micro"

    def _keywords(self, text: str, limit: int = 8) -> list[str]:
        tokens = [token for token in _tokenize(text) if token not in self._STOPWORDS]
        if not tokens:
            return []
        counts = Counter(tokens)
        ranked = [token for token, _count in counts.most_common(limit)]
        return ranked

    def sync_from_event_log(self, events: list[dict[str, Any]]) -> None:
        for event in events:
            if not self._is_world_event(event):
                continue
            self.observe_world_event(event)

    def observe_world_event(self, event: Mapping[str, Any]) -> None:
        if not self._is_world_event(event):
            return
        event_id = str(event.get("id", "")).strip()
        if event_id and event_id in self.seen_event_ids:
            return

        tick = int(event.get("tick", 0))
        tags = [str(tag) for tag in event.get("tags", []) if isinstance(tag, str)]
        text = str(event.get("text", "")).strip()
        if not text:
            return

        anchor = self._anchor_from_event(event)
        severity = self._severity_from_event(event)
        importance_raw = event.get("importance")
        importance = 0.5
        if isinstance(importance_raw, (int, float)):
            importance = _clamp_float(float(importance_raw), 0.0, 1.0)

        record = {
            "id": event.get("id"),
            "tick": tick,
            "text": text,
            "tags": tags,
            "anchor": anchor,
            "severity": severity,
            "importance": importance,
            "keywords": self._keywords(text, limit=10),
        }
        self.recent_world_events.append(record)

        tags_set = {tag.lower() for tag in tags}
        self.recent_emit_ticks.append(tick)
        self.last_emit_tick = max(self.last_emit_tick, tick)
        if severity == "danger" or "danger" in tags_set:
            self.recent_danger_ticks.append(tick)
        if event_id:
            if len(self.seen_event_order) >= self.seen_event_order.maxlen:
                stale_id = self.seen_event_order.popleft()
                self.seen_event_ids.discard(stale_id)
            self.seen_event_order.append(event_id)
            self.seen_event_ids.add(event_id)

    def _has_important_event_this_tick(self, tick: int, events: list[dict[str, Any]]) -> bool:
        for event in reversed(events):
            event_tick = int(event.get("tick", -1))
            if event_tick < tick:
                break
            if event_tick != tick:
                continue
            tags = {str(tag).lower() for tag in event.get("tags", [])}
            if "important" in tags or "danger" in tags:
                return True
            importance = event.get("importance")
            if isinstance(importance, (int, float)) and float(importance) >= 0.7:
                return True
        return False

    def _repeat_ratio(self, texts: list[str]) -> float:
        normalized = [_normalize_text(text) for text in texts if str(text).strip()]
        if len(normalized) < 2:
            return 0.0
        counts = Counter(normalized)
        repeated = sum(count - 1 for count in counts.values() if count > 1)
        return repeated / max(1, len(normalized))

    def _is_silence_trigger(self, tick: int, events: list[dict[str, Any]], agents_count: int) -> bool:
        window = max(self.silence_window_ticks, self._min_interval_ticks(tick))
        recent_dialogue: list[str] = []
        for event in reversed(events):
            event_tick = int(event.get("tick", tick))
            if tick - event_tick > window:
                break
            if not self._is_dialogue_event(event):
                continue
            text = str(event.get("text", "")).strip()
            if text:
                recent_dialogue.append(text)

        low_dialogue_threshold = max(2, agents_count - 1)
        if len(recent_dialogue) <= low_dialogue_threshold:
            self.last_should_emit_reason = "silence_low_dialogue"
            return True

        repeat_ratio = self._repeat_ratio(recent_dialogue)
        if repeat_ratio >= self.repeat_ratio_threshold:
            self.last_should_emit_reason = f"silence_repeats_{repeat_ratio:.2f}"
            return True

        return False

    def _is_world_quiet_trigger(self, tick: int) -> bool:
        if not self.recent_world_events:
            self.last_should_emit_reason = "world_quiet_empty"
            return True

        last_tick = max(int(item.get("tick", -10_000)) for item in self.recent_world_events)
        needed_gap = max(self._min_interval_ticks(tick) * 2, 14)
        if tick - last_tick >= needed_gap:
            self.last_should_emit_reason = f"world_quiet_gap_{tick - last_tick}"
            return True
        return False

    def _prune_ticks(self, ticks: deque[int], current_tick: int, window: int) -> None:
        while ticks and (current_tick - ticks[0]) > window:
            ticks.popleft()

    def should_emit(
        self,
        *,
        tick: int,
        events: list[dict[str, Any]],
        agents: list[dict[str, Any]],
        reply_queue_pending: int,
    ) -> bool:
        self.last_should_emit_reason = ""
        self.last_reject_reason = ""

        if not self.enabled:
            self.last_reject_reason = "disabled"
            return False
        if not agents:
            self.last_reject_reason = "no_agents"
            return False

        self.sync_from_event_log(events)

        if self._has_important_event_this_tick(tick, events):
            self.blocked_guardrails += 1
            self.last_reject_reason = "important_event_same_tick"
            return False

        pending_guardrail = max(4, len(agents) + 2)
        if reply_queue_pending >= pending_guardrail:
            self.blocked_guardrails += 1
            self.last_reject_reason = f"reply_queue_pending_{reply_queue_pending}"
            return False

        min_interval = self._min_interval_ticks(tick)
        if (tick - self.last_emit_tick) < min_interval:
            self.blocked_budget += 1
            self.last_reject_reason = f"min_interval_{min_interval}"
            return False

        self._prune_ticks(self.recent_emit_ticks, tick, window=50)
        max_per_50 = self._max_per_50_ticks(tick)
        if len(self.recent_emit_ticks) >= max_per_50:
            self.blocked_budget += 1
            self.last_reject_reason = f"max_per_50_{max_per_50}"
            return False

        is_silence = self._is_silence_trigger(tick, events=events, agents_count=len(agents))
        is_world_quiet = self._is_world_quiet_trigger(tick)
        if not is_silence and not is_world_quiet:
            self.last_reject_reason = "no_trigger"
            return False

        if not self.last_should_emit_reason:
            self.last_should_emit_reason = "trigger"
        return True

    def _recent_world_payload(self, limit: int = 10) -> list[dict[str, Any]]:
        items = list(self.recent_world_events)[-limit:]
        return [
            {
                "id": item.get("id"),
                "tick": item.get("tick"),
                "text": str(item.get("text", ""))[:220],
                "anchor": item.get("anchor"),
                "severity": item.get("severity"),
                "tags": list(item.get("tags", []))[:6],
            }
            for item in items
        ]

    def _forbidden_topics(self) -> list[str]:
        tokens: list[str] = []
        for item in list(self.recent_world_events)[-10:]:
            tokens.extend(item.get("keywords", []))
        ranked = [token for token, _count in Counter(tokens).most_common(24)]
        return ranked[:24]

    def _agent_snapshot(self, agents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        snapshot: list[dict[str, Any]] = []
        for agent in agents[:8]:
            item = {
                "id": agent.get("id"),
                "name": agent.get("name"),
                "traits": agent.get("traits"),
                "mood": agent.get("mood"),
                "mood_label": agent.get("mood_label"),
                "pos": agent.get("pos", {}),
            }
            snapshot.append(item)
        return snapshot

    def _system_prompt(self) -> str:
        return (
            "Ты генератор микро-событий мира.\n"
            "Событие должно быть наблюдаемым, локальным и привязанным к anchor.\n"
            "Без объяснений, без советов, без мета.\n"
            "Верни строго один JSON-объект, без markdown и без лишнего текста.\n"
            "Используй только anchors из USER_CONTEXT_JSON.anchors.\n"
            "text: 140-220 символов, конкретное наблюдаемое событие.\n"
            "Событие должно создавать напряжение, неожиданность или повод для разных реакций агентов.\n"
            "Избегай сухого описания погоды или нейтрального фона без действия.\n"
            "Добавляй конкретный наблюдаемый крючок: резкое действие, странный объект, тревожную деталь, или сбой в окружении.\n"
            "Пиши как живую сцену в парке, а не как отчёт наблюдателя.\n"
            "Не пиши прямых приказов агентам и не используй мета-лексикон.\n"
            "tags должны включать world, micro и anchor:<anchor>."
        )

    def _json_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["text", "anchor", "severity", "importance", "tags"],
            "properties": {
                "text": {"type": "string", "minLength": 140, "maxLength": 220},
                "anchor": {"type": "string", "enum": list(self.anchors)},
                "severity": {"type": "string", "enum": ["micro", "normal", "danger"]},
                "importance": {"type": "number", "minimum": 0.1, "maximum": 0.9},
                "tags": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1, "maxLength": 40},
                    "minItems": 2,
                    "maxItems": 8,
                },
            },
        }

    def _base_user_payload(
        self,
        *,
        tick: int,
        agents: list[dict[str, Any]],
        reply_queue_pending: int,
    ) -> dict[str, Any]:
        return {
            "tick": tick,
            "anchors": list(self.anchors),
            "recent_world_events": self._recent_world_payload(limit=10),
            "world_state": {
                "agents_count": len(agents),
                "agents": self._agent_snapshot(agents),
                "reply_queue_pending": reply_queue_pending,
            },
            "anti_repeat": {
                "forbidden_topics": self._forbidden_topics(),
            },
        }

    def _extract_candidate_payload(self, payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
        if {"text", "anchor", "severity", "importance"}.issubset(set(payload.keys())):
            return payload

        for key in ("event", "data", "result", "payload", "response"):
            nested = payload.get(key)
            if isinstance(nested, Mapping):
                if {"text", "anchor", "severity", "importance"}.issubset(set(nested.keys())):
                    return nested
            if isinstance(nested, list) and nested and isinstance(nested[0], Mapping):
                first = nested[0]
                if {"text", "anchor", "severity", "importance"}.issubset(set(first.keys())):
                    return first
        return None

    def _contains_forbidden_meta(self, text: str) -> bool:
        lowered = text.lower()
        return any(token in lowered for token in self._META_BLACKLIST)

    def _contains_directive(self, text: str) -> bool:
        return any(pattern.search(text) for pattern in self._DIRECTIVE_PATTERNS)

    def _sanitize_tags(self, tags: list[str], anchor: str, severity: str) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for raw_tag in tags:
            tag = str(raw_tag).strip().lower()
            if not tag:
                continue
            if tag in seen:
                continue
            seen.add(tag)
            normalized.append(tag)
            if len(normalized) >= 8:
                break

        required = ["world", "micro", severity, f"anchor:{anchor}"]
        for tag in required:
            if tag not in seen:
                normalized.append(tag)
                seen.add(tag)
        return normalized[:8]

    def _normalize_event_candidate(self, raw_payload: Mapping[str, Any]) -> GeneratedWorldEvent | None:
        payload = self._extract_candidate_payload(raw_payload)
        if payload is None:
            self.validation_failures += 1
            self.last_reject_reason = "schema_extract_failed"
            return None

        text = str(payload.get("text", "")).strip()
        if not text:
            self.validation_failures += 1
            self.last_reject_reason = "empty_text"
            return None

        if not (140 <= len(text) <= 220):
            self.validation_failures += 1
            self.last_reject_reason = f"text_len_{len(text)}"
            return None

        if self._contains_forbidden_meta(text):
            self.validation_failures += 1
            self.last_reject_reason = "meta_blacklist"
            return None

        if self._contains_directive(text):
            self.validation_failures += 1
            self.last_reject_reason = "direct_command"
            return None

        anchor = str(payload.get("anchor", "")).strip()
        if anchor not in self.anchor_set:
            self.validation_failures += 1
            self.last_reject_reason = "invalid_anchor"
            return None

        severity = str(payload.get("severity", "")).strip().lower()
        if severity not in self._SEVERITY_VALUES:
            self.validation_failures += 1
            self.last_reject_reason = "invalid_severity"
            return None

        importance_raw = payload.get("importance")
        if not isinstance(importance_raw, (int, float)):
            self.validation_failures += 1
            self.last_reject_reason = "invalid_importance"
            return None
        importance = _clamp_float(float(importance_raw), 0.1, 0.9)

        tags_raw = payload.get("tags")
        tags = [str(item) for item in tags_raw] if isinstance(tags_raw, list) else []
        tags = self._sanitize_tags(tags, anchor=anchor, severity=severity)

        return GeneratedWorldEvent(
            text=text,
            anchor=anchor,
            severity=severity,
            importance=importance,
            tags=tags,
        )

    def _is_repetitive(self, candidate: GeneratedWorldEvent) -> bool:
        candidate_norm = _normalize_text(candidate.text)
        if not candidate_norm:
            return True
        candidate_tokens = set(_tokenize(candidate.text))
        for recent in self.recent_world_events:
            recent_text = str(recent.get("text", ""))
            if not recent_text:
                continue
            recent_norm = _normalize_text(recent_text)
            if candidate_norm == recent_norm:
                return True

            recent_tokens = set(_tokenize(recent_text))
            if len(candidate_tokens) < 4 or len(recent_tokens) < 4:
                continue
            union = len(candidate_tokens | recent_tokens)
            if union == 0:
                continue
            jaccard = len(candidate_tokens & recent_tokens) / union
            if jaccard >= 0.8:
                return True

            if recent.get("anchor") == candidate.anchor:
                overlap = len(candidate_tokens & recent_tokens)
                if overlap >= 5 and len(candidate_tokens) >= 6:
                    return True
        return False

    def _shift_anchor(self, candidate: GeneratedWorldEvent, tick: int) -> GeneratedWorldEvent:
        if len(self.anchors) <= 1:
            return candidate
        idx = self.anchors.index(candidate.anchor)
        next_idx = (idx + 1 + (tick % (len(self.anchors) - 1))) % len(self.anchors)
        if next_idx == idx:
            next_idx = (idx + 1) % len(self.anchors)
        shifted_anchor = self.anchors[next_idx]
        shifted_tags = self._sanitize_tags(candidate.tags, anchor=shifted_anchor, severity=candidate.severity)
        return GeneratedWorldEvent(
            text=candidate.text,
            anchor=shifted_anchor,
            severity=candidate.severity,
            importance=candidate.importance,
            tags=shifted_tags,
        )

    def _apply_danger_cooldown(self, candidate: GeneratedWorldEvent, tick: int) -> GeneratedWorldEvent:
        if candidate.severity != "danger":
            return candidate

        self._prune_ticks(self.recent_danger_ticks, tick, window=500)
        cooldown = self._danger_cooldown_ticks(tick)
        if any((tick - prev_tick) < cooldown for prev_tick in self.recent_danger_ticks):
            lowered_tags = [tag for tag in candidate.tags if tag != "danger"]
            lowered_tags = self._sanitize_tags(lowered_tags, anchor=candidate.anchor, severity="normal")
            return GeneratedWorldEvent(
                text=candidate.text,
                anchor=candidate.anchor,
                severity="normal",
                importance=min(candidate.importance, 0.7),
                tags=lowered_tags,
            )
        return candidate

    def _request_event(self, *, user_payload: dict[str, Any], minimum_output_tokens: int = 220) -> GeneratedWorldEvent | None:
        response_obj = self.client.request_json_object(
            system_prompt=self._system_prompt(),
            user_payload=user_payload,
            temperature=0.35,
            json_schema=self._json_schema(),
            minimum_output_tokens=minimum_output_tokens,
        )
        if not response_obj:
            self.last_reject_reason = "llm_empty"
            return None
        return self._normalize_event_candidate(response_obj)

    def generate(
        self,
        *,
        tick: int,
        events: list[dict[str, Any]],
        agents: list[dict[str, Any]],
        reply_queue_pending: int,
    ) -> GeneratedWorldEvent | None:
        if not self.enabled:
            self.last_reject_reason = "disabled"
            return None

        self.sync_from_event_log(events)

        payload = self._base_user_payload(
            tick=tick,
            agents=agents,
            reply_queue_pending=reply_queue_pending,
        )
        candidate = self._request_event(user_payload=payload)
        if candidate is None:
            return None

        if self._is_repetitive(candidate):
            self.rerolls += 1
            reroll_payload = dict(payload)
            reroll_payload["anti_repeat"] = {
                **payload.get("anti_repeat", {}),
                "reroll": True,
                "note": "Не повторяй темы и формулировки recent_world_events. Возьми другой anchor.",
            }
            rerolled = self._request_event(user_payload=reroll_payload, minimum_output_tokens=260)
            if rerolled is not None:
                candidate = rerolled

            if self._is_repetitive(candidate):
                candidate = self._shift_anchor(candidate, tick=tick)
                if self._is_repetitive(candidate):
                    self.blocked_repeats += 1
                    self.last_reject_reason = "repeat_after_reroll"
                    return None

        candidate = self._apply_danger_cooldown(candidate, tick=tick)
        return candidate

    def metrics(self, *, tick: int, events: list[dict[str, Any]]) -> dict[str, Any]:
        self.sync_from_event_log(events)

        world_micro_last_100 = [
            event
            for event in events
            if int(event.get("tick", -10_000)) >= tick - 99
            and "world" in {str(tag).lower() for tag in event.get("tags", [])}
            and "micro" in {str(tag).lower() for tag in event.get("tags", [])}
        ]

        world_ids_last_100 = {
            str(event.get("id"))
            for event in events
            if int(event.get("tick", -10_000)) >= tick - 99
            and self._is_world_event(event)
        }

        agent_responses_last_100 = [
            event
            for event in events
            if int(event.get("tick", -10_000)) >= tick - 99
            and str(event.get("source_type", "")).lower() == "agent"
            and ({str(tag).lower() for tag in event.get("tags", [])} & {"dialogue", "reply", "agent_message"})
        ]

        with_world_evidence = 0
        for event in agent_responses_last_100:
            evidence_ids = event.get("evidence_ids", [])
            if not isinstance(evidence_ids, list):
                continue
            if any(str(eid) in world_ids_last_100 for eid in evidence_ids):
                with_world_evidence += 1

        evidence_ratio = 0.0
        if agent_responses_last_100:
            evidence_ratio = with_world_evidence / len(agent_responses_last_100)

        repeat_recent = self._dialogue_repeat_ratio(events, start_tick=tick - 49, end_tick=tick)
        repeat_prev = self._dialogue_repeat_ratio(events, start_tick=tick - 99, end_tick=tick - 50)

        return {
            "enabled": self.enabled,
            "config_enabled": self.enabled_by_config,
            "client_enabled": self.client.enabled,
            "anchors": list(self.anchors),
            "last_should_emit_reason": self.last_should_emit_reason,
            "last_reject_reason": self.last_reject_reason,
            "budget": {
                "min_interval_ticks": self._min_interval_ticks(tick),
                "max_per_50_ticks": self._max_per_50_ticks(tick),
                "danger_cooldown_ticks": self._danger_cooldown_ticks(tick),
                "blocked_budget": self.blocked_budget,
                "blocked_guardrails": self.blocked_guardrails,
                "blocked_repeats": self.blocked_repeats,
                "rerolls": self.rerolls,
                "validation_failures": self.validation_failures,
            },
            "metrics": {
                "world_events_per_100_ticks": len(world_micro_last_100),
                "agent_world_evidence_ratio_100_ticks": round(evidence_ratio, 4),
                "dialogue_repeat_ratio_recent_50": round(repeat_recent, 4),
                "dialogue_repeat_ratio_prev_50": round(repeat_prev, 4),
            },
        }

    def _dialogue_repeat_ratio(self, events: list[dict[str, Any]], start_tick: int, end_tick: int) -> float:
        if end_tick < start_tick:
            return 0.0

        texts: list[str] = []
        for event in events:
            tick = int(event.get("tick", -10_000))
            if tick < start_tick or tick > end_tick:
                continue
            if str(event.get("source_type", "")).lower() != "agent":
                continue
            tags = {str(tag).lower() for tag in event.get("tags", [])}
            if not (tags & {"dialogue", "reply", "agent_message"}):
                continue
            text = str(event.get("text", "")).strip()
            if not text:
                continue
            texts.append(_normalize_text(text))

        if len(texts) < 2:
            return 0.0
        counts = Counter(texts)
        repeated = sum(count - 1 for count in counts.values() if count > 1)
        return repeated / len(texts)
