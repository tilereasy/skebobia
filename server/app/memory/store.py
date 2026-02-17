from __future__ import annotations

import os
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Deque


_STOPWORDS = {
    "and",
    "the",
    "for",
    "with",
    "this",
    "that",
    "from",
    "have",
    "was",
    "were",
    "как",
    "что",
    "это",
    "так",
    "или",
    "для",
    "его",
    "она",
    "they",
    "them",
}


def _is_enabled(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _utc_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _tokenize(text: str) -> frozenset[str]:
    lowered = text.lower()
    cleaned = "".join(ch if ch.isalnum() else " " for ch in lowered)
    words = [word for word in cleaned.split() if len(word) >= 3 and word not in _STOPWORDS]
    return frozenset(words)


def _clamp_float(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(slots=True)
class MemoryEntry:
    id: str
    agent_id: str
    text: str
    tags: tuple[str, ...]
    ts: str
    tick: int
    importance: float
    event_id: str | None = None
    source_id: str | None = None
    target_id: str | None = None
    tokens: frozenset[str] = field(default_factory=frozenset)


class EpisodicMemoryStore:
    def __init__(
        self,
        *,
        enabled: bool = True,
        per_agent_limit: int = 400,
    ) -> None:
        self.enabled = enabled
        self.per_agent_limit = max(50, per_agent_limit)
        self._entries: dict[str, Deque[MemoryEntry]] = defaultdict(
            lambda: deque(maxlen=self.per_agent_limit)
        )
        self._next_id = 0

    @classmethod
    def from_env(cls) -> "EpisodicMemoryStore":
        enabled = _is_enabled(os.getenv("MEMORY_ENABLED"), default=True)
        try:
            per_agent_limit = int(os.getenv("MEMORY_EPISODES_PER_AGENT", "400"))
        except ValueError:
            per_agent_limit = 400
        per_agent_limit = max(50, min(per_agent_limit, 5000))
        return cls(enabled=enabled, per_agent_limit=per_agent_limit)

    def remember(
        self,
        *,
        agent_id: str,
        text: str,
        tags: list[str] | tuple[str, ...],
        tick: int,
        event_id: str | None = None,
        source_id: str | None = None,
        target_id: str | None = None,
        importance: float = 0.5,
    ) -> str | None:
        if not self.enabled:
            return None

        normalized_text = " ".join(text.strip().split())
        if not normalized_text:
            return None

        self._next_id += 1
        entry = MemoryEntry(
            id=f"m{self._next_id}",
            agent_id=agent_id,
            text=normalized_text[:300],
            tags=tuple(str(tag) for tag in tags)[:8],
            ts=_utc_iso(),
            tick=tick,
            importance=_clamp_float(float(importance), 0.0, 1.0),
            event_id=event_id,
            source_id=source_id,
            target_id=target_id,
            tokens=_tokenize(normalized_text),
        )
        self._entries[agent_id].append(entry)
        return entry.id

    def recall(self, *, agent_id: str, query: str, limit: int = 3) -> list[dict]:
        if not self.enabled:
            return []
        if limit <= 0:
            return []

        items = list(self._entries.get(agent_id, ()))
        if not items:
            return []

        query_tokens = _tokenize(query)
        items_count = len(items)
        scored: list[tuple[float, MemoryEntry]] = []

        for index, entry in enumerate(items):
            recency = (index + 1) / items_count
            overlap = 0.0
            if query_tokens and entry.tokens:
                overlap = len(query_tokens & entry.tokens) / max(len(query_tokens), len(entry.tokens))
            score = (entry.importance * 0.5) + (overlap * 0.35) + (recency * 0.15)
            scored.append((score, entry))

        scored.sort(key=lambda item: (item[0], item[1].tick), reverse=True)

        result: list[dict] = []
        for score, entry in scored[: max(1, min(limit, 20))]:
            result.append(
                {
                    "id": entry.id,
                    "event_id": entry.event_id,
                    "text": entry.text,
                    "tags": list(entry.tags),
                    "tick": entry.tick,
                    "ts": entry.ts,
                    "score": round(score, 4),
                }
            )
        return result

    def latest(self, *, agent_id: str, limit: int = 5) -> list[dict]:
        if not self.enabled:
            return []
        items = list(self._entries.get(agent_id, ()))
        if not items:
            return []

        result: list[dict] = []
        for entry in items[-max(1, min(limit, 20)) :]:
            result.append(
                {
                    "id": entry.id,
                    "event_id": entry.event_id,
                    "text": entry.text,
                    "tags": list(entry.tags),
                    "tick": entry.tick,
                    "ts": entry.ts,
                    "score": round(entry.importance, 4),
                }
            )
        return result

    def stats(self) -> dict[str, int | bool]:
        total_entries = sum(len(entries) for entries in self._entries.values())
        return {
            "enabled": self.enabled,
            "agents": len(self._entries),
            "entries": total_entries,
            "per_agent_limit": self.per_agent_limit,
        }
