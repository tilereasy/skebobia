from __future__ import annotations

import hashlib
import logging
import math
import os
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Deque

from openai import OpenAI


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

LOGGER = logging.getLogger("app.memory.store")


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


def _vector_norm(vector: tuple[float, ...]) -> float:
    return math.sqrt(sum(component * component for component in vector))


def _cosine_similarity(
    left_vector: tuple[float, ...],
    left_norm: float,
    right_vector: tuple[float, ...],
    right_norm: float,
) -> float:
    if left_norm <= 1e-9 or right_norm <= 1e-9:
        return 0.0
    if len(left_vector) != len(right_vector):
        return 0.0
    dot_product = sum(l * r for l, r in zip(left_vector, right_vector))
    return dot_product / (left_norm * right_norm)


def _extract_chat_content(response: Any) -> str | None:
    choices = getattr(response, "choices", None)
    if not choices:
        return None
    first = choices[0]
    message = getattr(first, "message", None)
    if message is None:
        return None
    content = getattr(message, "content", None)
    if isinstance(content, str):
        content = content.strip()
        return content or None
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
        if chunks:
            return " ".join(chunks)
    return None


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
    embedding: tuple[float, ...] = field(default_factory=tuple)
    embedding_norm: float = 0.0
    vector_backend: str = "none"


class EpisodicMemoryStore:
    def __init__(
        self,
        *,
        enabled: bool = True,
        per_agent_limit: int = 400,
        vector_enabled: bool = True,
        embedding_model: str = "text-embedding-3-small",
        embedding_base_url: str = "https://api.openai.com/v1",
        embedding_api_key: str | None = None,
        embedding_timeout_sec: float = 20.0,
        embedding_hash_fallback: bool = True,
        hash_embedding_dim: int = 256,
        summary_enabled: bool = True,
        summary_batch_size: int = 10,
        summary_model: str = "",
        summary_base_url: str = "https://api.openai.com/v1",
        summary_api_key: str | None = None,
        summary_timeout_sec: float = 20.0,
        summary_max_output_chars: int = 320,
    ) -> None:
        self.enabled = enabled
        self.per_agent_limit = max(20, per_agent_limit)

        self.vector_enabled = vector_enabled
        self.embedding_model = embedding_model.strip()
        self.embedding_base_url = embedding_base_url.rstrip("/")
        self.embedding_api_key = embedding_api_key
        self.embedding_timeout_sec = max(2.0, min(float(embedding_timeout_sec), 120.0))
        self.embedding_hash_fallback = embedding_hash_fallback
        self.hash_embedding_dim = max(32, min(hash_embedding_dim, 4096))

        self.summary_enabled = summary_enabled
        self.summary_batch_size = max(3, min(summary_batch_size, 40))
        self.summary_model = summary_model.strip()
        self.summary_base_url = summary_base_url.rstrip("/")
        self.summary_api_key = summary_api_key
        self.summary_timeout_sec = max(2.0, min(float(summary_timeout_sec), 120.0))
        self.summary_max_output_chars = max(80, min(summary_max_output_chars, 1200))

        self._entries: dict[str, Deque[MemoryEntry]] = defaultdict(deque)
        self._next_id = 0
        self._summary_entries_created = 0
        self._embedding_failures = 0
        self._summary_failures = 0

        self._embedding_client: OpenAI | None = None
        self._summary_client: OpenAI | None = None

    @classmethod
    def from_env(cls) -> "EpisodicMemoryStore":
        enabled = _is_enabled(os.getenv("MEMORY_ENABLED"), default=True)
        try:
            per_agent_limit = int(os.getenv("MEMORY_EPISODES_PER_AGENT", "400"))
        except ValueError:
            per_agent_limit = 400
        per_agent_limit = max(20, min(per_agent_limit, 5000))

        vector_enabled = _is_enabled(os.getenv("MEMORY_VECTOR_ENABLED"), default=True)
        embedding_model = os.getenv("MEMORY_EMBEDDING_MODEL", "text-embedding-3-small").strip()
        embedding_base_url = os.getenv(
            "MEMORY_EMBEDDING_BASE_URL",
            os.getenv("LLM_DECIDER_BASE_URL", "https://api.openai.com/v1"),
        ).strip()
        embedding_api_key = (
            os.getenv("MEMORY_EMBEDDING_API_KEY")
            or os.getenv("LLM_DECIDER_API_KEY")
            or ""
        ).strip() or None
        try:
            embedding_timeout_sec = float(os.getenv("MEMORY_EMBEDDING_TIMEOUT_SEC", "20"))
        except ValueError:
            embedding_timeout_sec = 20.0
        embedding_hash_fallback = _is_enabled(os.getenv("MEMORY_EMBEDDING_HASH_FALLBACK"), default=True)
        try:
            hash_embedding_dim = int(os.getenv("MEMORY_HASH_EMBEDDING_DIM", "256"))
        except ValueError:
            hash_embedding_dim = 256

        summary_enabled = _is_enabled(os.getenv("MEMORY_SUMMARY_ENABLED"), default=True)
        try:
            summary_batch_size = int(os.getenv("MEMORY_SUMMARY_BATCH_SIZE", "10"))
        except ValueError:
            summary_batch_size = 10
        summary_model = os.getenv("MEMORY_SUMMARY_MODEL", os.getenv("LLM_DECIDER_MODEL", "")).strip()
        summary_base_url = os.getenv(
            "MEMORY_SUMMARY_BASE_URL",
            os.getenv("LLM_DECIDER_BASE_URL", "https://api.openai.com/v1"),
        ).strip()
        summary_api_key = (
            os.getenv("MEMORY_SUMMARY_API_KEY")
            or os.getenv("LLM_DECIDER_API_KEY")
            or ""
        ).strip() or None
        try:
            summary_timeout_sec = float(os.getenv("MEMORY_SUMMARY_TIMEOUT_SEC", "20"))
        except ValueError:
            summary_timeout_sec = 20.0
        try:
            summary_max_output_chars = int(os.getenv("MEMORY_SUMMARY_MAX_OUTPUT_CHARS", "320"))
        except ValueError:
            summary_max_output_chars = 320

        return cls(
            enabled=enabled,
            per_agent_limit=per_agent_limit,
            vector_enabled=vector_enabled,
            embedding_model=embedding_model,
            embedding_base_url=embedding_base_url,
            embedding_api_key=embedding_api_key,
            embedding_timeout_sec=embedding_timeout_sec,
            embedding_hash_fallback=embedding_hash_fallback,
            hash_embedding_dim=hash_embedding_dim,
            summary_enabled=summary_enabled,
            summary_batch_size=summary_batch_size,
            summary_model=summary_model,
            summary_base_url=summary_base_url,
            summary_api_key=summary_api_key,
            summary_timeout_sec=summary_timeout_sec,
            summary_max_output_chars=summary_max_output_chars,
        )

    def _next_memory_id(self) -> str:
        self._next_id += 1
        return f"m{self._next_id}"

    def _get_embedding_client(self) -> OpenAI | None:
        if not self.embedding_api_key or not self.embedding_model:
            return None
        if self._embedding_client is None:
            self._embedding_client = OpenAI(
                api_key=self.embedding_api_key,
                base_url=self.embedding_base_url,
                timeout=self.embedding_timeout_sec,
                max_retries=1,
            )
        return self._embedding_client

    def _get_summary_client(self) -> OpenAI | None:
        if not self.summary_api_key or not self.summary_model:
            return None
        if self._summary_client is None:
            self._summary_client = OpenAI(
                api_key=self.summary_api_key,
                base_url=self.summary_base_url,
                timeout=self.summary_timeout_sec,
                max_retries=1,
            )
        return self._summary_client

    def _hash_embedding(self, text: str) -> tuple[tuple[float, ...], float]:
        tokens = sorted(_tokenize(text))
        if not tokens:
            compact = "".join(ch for ch in text.lower() if ch.isalnum())
            tokens = [compact[i : i + 3] for i in range(0, max(1, len(compact) - 2), 3)] or [compact or "empty"]

        vector = [0.0] * self.hash_embedding_dim
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.hash_embedding_dim
            sign = -1.0 if (digest[4] & 1) else 1.0
            weight = 1.0 + (digest[5] / 255.0) * 0.25
            vector[idx] += sign * weight
        packed = tuple(vector)
        norm = _vector_norm(packed)
        return packed, norm

    def _embed_text(self, text: str) -> tuple[tuple[float, ...], float, str]:
        if not self.vector_enabled:
            return tuple(), 0.0, "none"

        normalized = " ".join(text.strip().split())[:2000]
        if not normalized:
            return tuple(), 0.0, "none"

        client = self._get_embedding_client()
        if client is not None:
            try:
                response = client.embeddings.create(
                    model=self.embedding_model,
                    input=normalized,
                )
                data = getattr(response, "data", None)
                if data:
                    vector = tuple(float(item) for item in data[0].embedding)
                    norm = _vector_norm(vector)
                    if norm > 1e-9:
                        return vector, norm, "openai"
            except Exception as exc:
                self._embedding_failures += 1
                LOGGER.warning("Memory embedding request failed type=%s detail=%r", type(exc).__name__, exc)

        if self.embedding_hash_fallback:
            vector, norm = self._hash_embedding(normalized)
            if norm > 1e-9:
                return vector, norm, "hash"

        return tuple(), 0.0, "none"

    def _summarize_with_llm(self, entries: list[MemoryEntry]) -> str | None:
        if not self.summary_enabled:
            return None
        client = self._get_summary_client()
        if client is None:
            return None

        lines = []
        for item in entries:
            lines.append(f"- [{item.tick}] {item.text[:180]}")
        user_text = "\n".join(lines)

        try:
            response = client.chat.completions.create(
                model=self.summary_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Суммаризируй пачку эпизодической памяти агента в 1-2 коротких предложения. "
                            "Сохрани факты, участников и эмоциональный тон. Ответ только текстом."
                        ),
                    },
                    {"role": "user", "content": user_text},
                ],
                temperature=0.2,
                max_tokens=220,
            )
            content = _extract_chat_content(response)
            if not content:
                return None
            compact = " ".join(content.split())
            if not compact:
                return None
            return compact[: self.summary_max_output_chars]
        except Exception as exc:
            self._summary_failures += 1
            LOGGER.warning("Memory summary request failed type=%s detail=%r", type(exc).__name__, exc)
            return None

    def _summarize_fallback(self, entries: list[MemoryEntry]) -> str:
        parts: list[str] = []
        seen: set[str] = set()
        for item in entries:
            snippet = " ".join(item.text.strip().split())
            if not snippet:
                continue
            snippet = snippet[:120]
            if snippet in seen:
                continue
            seen.add(snippet)
            parts.append(snippet)
            if len(parts) >= 4:
                break
        if not parts:
            return "Сводка: существенных деталей не найдено."
        return ("Сводка: " + "; ".join(parts))[: self.summary_max_output_chars]

    def _summarize_entries(self, entries: list[MemoryEntry]) -> str:
        llm_summary = self._summarize_with_llm(entries)
        if llm_summary:
            return llm_summary
        return self._summarize_fallback(entries)

    def _create_entry(
        self,
        *,
        agent_id: str,
        text: str,
        tags: list[str] | tuple[str, ...],
        tick: int,
        event_id: str | None,
        source_id: str | None,
        target_id: str | None,
        importance: float,
    ) -> MemoryEntry | None:
        normalized_text = " ".join(text.strip().split())
        if not normalized_text:
            return None
        vector, vector_norm, vector_backend = self._embed_text(normalized_text)
        return MemoryEntry(
            id=self._next_memory_id(),
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
            embedding=vector,
            embedding_norm=vector_norm,
            vector_backend=vector_backend,
        )

    def _summarize_oldest_chunk(self, agent_id: str) -> bool:
        if not self.summary_enabled:
            return False
        entries = self._entries[agent_id]
        if len(entries) < 2:
            return False

        chunk_size = min(self.summary_batch_size, len(entries) - 1)
        if chunk_size < 2:
            return False

        chunk = [entries.popleft() for _ in range(chunk_size)]
        summary_text = self._summarize_entries(chunk)
        summary_importance = _clamp_float(
            max(0.55, sum(item.importance for item in chunk) / len(chunk)),
            0.0,
            1.0,
        )
        summary_tick = max(item.tick for item in chunk)
        summary_entry = self._create_entry(
            agent_id=agent_id,
            text=summary_text,
            tags=["summary", "memory"],
            tick=summary_tick,
            event_id=None,
            source_id=None,
            target_id=None,
            importance=summary_importance,
        )
        if summary_entry is None:
            return False

        entries.appendleft(summary_entry)
        self._summary_entries_created += 1
        return True

    def _enforce_limits(self, agent_id: str) -> None:
        entries = self._entries[agent_id]
        while len(entries) > self.per_agent_limit:
            if self._summarize_oldest_chunk(agent_id):
                continue
            entries.popleft()

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

        entry = self._create_entry(
            agent_id=agent_id,
            text=text,
            tags=tags,
            tick=tick,
            event_id=event_id,
            source_id=source_id,
            target_id=target_id,
            importance=importance,
        )
        if entry is None:
            return None

        self._entries[agent_id].append(entry)
        self._enforce_limits(agent_id)
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
        query_vector: tuple[float, ...] = tuple()
        query_vector_norm = 0.0
        if query.strip():
            query_vector, query_vector_norm, _ = self._embed_text(query)

        items_count = len(items)
        scored: list[tuple[float, MemoryEntry]] = []

        for index, entry in enumerate(items):
            recency = (index + 1) / items_count
            semantic = 0.0
            if query_vector and entry.embedding:
                cosine = _cosine_similarity(
                    query_vector,
                    query_vector_norm,
                    entry.embedding,
                    entry.embedding_norm,
                )
                semantic = _clamp_float((cosine + 1.0) / 2.0, 0.0, 1.0)
            elif query_tokens and entry.tokens:
                semantic = len(query_tokens & entry.tokens) / max(len(query_tokens), len(entry.tokens))

            summary_bonus = 0.03 if "summary" in entry.tags else 0.0
            score = (entry.importance * 0.35) + (semantic * 0.47) + (recency * 0.15) + summary_bonus
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
                    "vector_backend": entry.vector_backend,
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
                    "vector_backend": entry.vector_backend,
                }
            )
        return result

    def stats(self) -> dict[str, Any]:
        total_entries = sum(len(entries) for entries in self._entries.values())
        summary_entries = sum(
            1
            for entries in self._entries.values()
            for entry in entries
            if "summary" in entry.tags
        )
        return {
            "enabled": self.enabled,
            "agents": len(self._entries),
            "entries": total_entries,
            "summary_entries": summary_entries,
            "summary_entries_created": self._summary_entries_created,
            "per_agent_limit": self.per_agent_limit,
            "vector_enabled": self.vector_enabled,
            "embedding_model": self.embedding_model if self.embedding_model else "none",
            "embedding_hash_fallback": self.embedding_hash_fallback,
            "embedding_failures": self._embedding_failures,
            "summary_enabled": self.summary_enabled,
            "summary_model": self.summary_model if self.summary_model else "none",
            "summary_failures": self._summary_failures,
        }
