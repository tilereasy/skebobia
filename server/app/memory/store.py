from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Deque
from urllib.parse import urlparse, urlunparse

from openai import OpenAI

try:
    import psycopg
    from psycopg.rows import dict_row
except Exception:  # pragma: no cover - optional dependency in local dev
    psycopg = None
    dict_row = None


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
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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


def _safe_identifier(raw: str, fallback: str) -> str:
    name = (raw or "").strip()
    if not name:
        return fallback
    return name if _IDENTIFIER_RE.match(name) else fallback


def _quote_identifier(identifier: str) -> str:
    return f'"{identifier}"'


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
        backend: str = "memory",
        database_url: str | None = None,
        table_name: str = "agent_memories",
        vector_dim: int = 1536,
        pg_create_index: bool = True,
        pg_ivfflat_lists: int = 100,
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
        self.backend = backend.strip().lower() if backend else "memory"
        if self.backend not in {"memory", "pgvector"}:
            self.backend = "memory"
        self.database_url = (database_url or "").strip() or None
        self.table_name = _safe_identifier(table_name, "agent_memories")
        self.vector_dim = max(32, min(int(vector_dim), 4096))
        self.pg_create_index = pg_create_index
        self.pg_ivfflat_lists = max(8, min(int(pg_ivfflat_lists), 4000))

        self.vector_enabled = vector_enabled
        self.embedding_model = embedding_model.strip()
        self.embedding_base_url = embedding_base_url.rstrip("/")
        self.embedding_api_key = embedding_api_key
        self.embedding_timeout_sec = max(2.0, min(float(embedding_timeout_sec), 120.0))
        self.embedding_hash_fallback = embedding_hash_fallback
        self.hash_embedding_dim = max(32, min(hash_embedding_dim, 4096))
        if self.backend == "pgvector" and self.embedding_hash_fallback and self.hash_embedding_dim != self.vector_dim:
            LOGGER.warning(
                "MEMORY_HASH_EMBEDDING_DIM=%s mismatches MEMORY_VECTOR_DIM=%s, coercing to vector dim for pgvector",
                self.hash_embedding_dim,
                self.vector_dim,
            )
            self.hash_embedding_dim = self.vector_dim

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
        self._pg_conn: Any | None = None
        self._pg_enabled = False
        self._pg_errors = 0
        self._table_ident = _quote_identifier(self.table_name)

        self._init_pgvector_backend()

    @classmethod
    def from_env(cls) -> "EpisodicMemoryStore":
        enabled = _is_enabled(os.getenv("MEMORY_ENABLED"), default=True)
        try:
            per_agent_limit = int(os.getenv("MEMORY_EPISODES_PER_AGENT", "400"))
        except ValueError:
            per_agent_limit = 400
        per_agent_limit = max(20, min(per_agent_limit, 5000))

        backend = os.getenv("MEMORY_BACKEND", "auto").strip().lower()
        database_url = (
            os.getenv("MEMORY_DATABASE_URL")
            or os.getenv("VECTOR_DB_URL")
            or os.getenv("DATABASE_URL")
            or ""
        ).strip() or None
        if backend == "auto":
            backend = "pgvector" if database_url else "memory"

        try:
            vector_dim = int(os.getenv("MEMORY_VECTOR_DIM", "1536"))
        except ValueError:
            vector_dim = 1536
        table_name = os.getenv("MEMORY_TABLE_NAME", "agent_memories").strip() or "agent_memories"
        pg_create_index = _is_enabled(os.getenv("MEMORY_PGVECTOR_CREATE_INDEX"), default=True)
        try:
            pg_ivfflat_lists = int(os.getenv("MEMORY_PGVECTOR_IVFFLAT_LISTS", "100"))
        except ValueError:
            pg_ivfflat_lists = 100

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
        hash_dim_default = str(vector_dim if backend == "pgvector" else 256)
        try:
            hash_embedding_dim = int(os.getenv("MEMORY_HASH_EMBEDDING_DIM", hash_dim_default))
        except ValueError:
            hash_embedding_dim = vector_dim if backend == "pgvector" else 256

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
            backend=backend,
            database_url=database_url,
            table_name=table_name,
            vector_dim=vector_dim,
            pg_create_index=pg_create_index,
            pg_ivfflat_lists=pg_ivfflat_lists,
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
        stamp = int(datetime.now(UTC).timestamp() * 1000)
        return f"m{stamp:x}{self._next_id:04x}"

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

    def _init_pgvector_backend(self) -> None:
        if self.backend != "pgvector":
            return
        if psycopg is None:
            LOGGER.warning("MEMORY_BACKEND=pgvector requested, but psycopg is not installed. Falling back to memory backend.")
            self.backend = "memory"
            return
        if not self.database_url:
            LOGGER.warning("MEMORY_BACKEND=pgvector requested, but MEMORY_DATABASE_URL/VECTOR_DB_URL is empty.")
            self.backend = "memory"
            return

        candidates = [self.database_url]
        try:
            parsed = urlparse(self.database_url)
            if parsed.hostname == "postgres":
                if "@" in parsed.netloc:
                    creds, host_part = parsed.netloc.rsplit("@", 1)
                    host_part = host_part.replace("postgres", "127.0.0.1", 1)
                    netloc = f"{creds}@{host_part}"
                else:
                    netloc = parsed.netloc.replace("postgres", "127.0.0.1", 1)
                fallback_url = urlunparse(parsed._replace(netloc=netloc))
                if fallback_url not in candidates:
                    candidates.append(fallback_url)
        except Exception:
            pass

        last_error: Exception | None = None
        for dsn in candidates:
            try:
                self._pg_conn = psycopg.connect(
                    dsn,
                    autocommit=True,
                    row_factory=dict_row,
                )
                self._ensure_pg_schema()
                self._pg_enabled = True
                if dsn != self.database_url:
                    LOGGER.warning("Connected pgvector via fallback DSN host for local runtime.")
                return
            except Exception as exc:
                last_error = exc
                self._pg_conn = None

        self._pg_errors += 1
        LOGGER.warning(
            "Failed to init pgvector backend type=%s detail=%r",
            type(last_error).__name__ if last_error else "UnknownError",
            last_error,
        )
        self.backend = "memory"
        self._pg_enabled = False

    def _ensure_pg_schema(self) -> None:
        if not self._pg_conn:
            return

        with self._pg_conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = %s
                """,
                (self.table_name,),
            )
            existing_columns = list(cur.fetchall())

        if existing_columns:
            legacy_required = {
                "kind",
                "metadata",
                "archived",
                "embedding_status",
                "embedding_attempts",
                "created_at",
                "updated_at",
            }
            legacy_hits = {
                str(row.get("column_name"))
                for row in existing_columns
                if str(row.get("column_name")) in legacy_required
                and str(row.get("is_nullable")) == "NO"
                and row.get("column_default") is None
            }
            if legacy_hits:
                fallback_table = _safe_identifier(f"{self.table_name}_v2", "agent_memories_v2")
                LOGGER.warning(
                    "Table %s has legacy non-null columns %s; switching pgvector backend table to %s.",
                    self.table_name,
                    sorted(legacy_hits),
                    fallback_table,
                )
                self.table_name = fallback_table
                self._table_ident = _quote_identifier(self.table_name)

        agent_tick_idx = _quote_identifier(f"{self.table_name}_agent_tick_idx")
        id_unique_idx = _quote_identifier(f"{self.table_name}_id_uidx")
        vector_idx = _quote_identifier(f"{self.table_name}_embedding_ivfflat_idx")

        with self._pg_conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table_ident} (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    tags JSONB NOT NULL DEFAULT '[]'::jsonb,
                    ts TEXT NOT NULL,
                    tick INTEGER NOT NULL,
                    importance DOUBLE PRECISION NOT NULL,
                    event_id TEXT NULL,
                    source_id TEXT NULL,
                    target_id TEXT NULL,
                    tokens TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
                    embedding vector({self.vector_dim}) NULL,
                    embedding_norm DOUBLE PRECISION NOT NULL DEFAULT 0,
                    vector_backend TEXT NOT NULL DEFAULT 'none'
                )
                """
            )
            required_columns = [
                ("id", "TEXT"),
                ("agent_id", "TEXT"),
                ("text", "TEXT"),
                ("tags", "JSONB NOT NULL DEFAULT '[]'::jsonb"),
                ("ts", "TEXT"),
                ("tick", "INTEGER NOT NULL DEFAULT 0"),
                ("importance", "DOUBLE PRECISION NOT NULL DEFAULT 0"),
                ("event_id", "TEXT NULL"),
                ("source_id", "TEXT NULL"),
                ("target_id", "TEXT NULL"),
                ("tokens", "TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[]"),
                ("embedding", f"vector({self.vector_dim}) NULL"),
                ("embedding_norm", "DOUBLE PRECISION NOT NULL DEFAULT 0"),
                ("vector_backend", "TEXT NOT NULL DEFAULT 'none'"),
            ]
            for column_name, ddl in required_columns:
                cur.execute(
                    f"ALTER TABLE {self._table_ident} ADD COLUMN IF NOT EXISTS {_quote_identifier(column_name)} {ddl}"
                )

            cur.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS {id_unique_idx} ON {self._table_ident} (id)")
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {agent_tick_idx} ON {self._table_ident} (agent_id, tick DESC, ts DESC)"
            )
            if self.pg_create_index:
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {vector_idx}
                    ON {self._table_ident}
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = {self.pg_ivfflat_lists})
                    """
                )

    @staticmethod
    def _vector_literal(vector: tuple[float, ...]) -> str:
        return "[" + ",".join(f"{component:.10f}" for component in vector) + "]"

    def _persist_entry_pg(self, entry: MemoryEntry) -> None:
        if not self._pg_enabled or not self._pg_conn:
            return

        has_vector = (
            self.vector_enabled
            and entry.embedding_norm > 1e-9
            and len(entry.embedding) == self.vector_dim
        )
        vector_sql = "%s::vector" if has_vector else "NULL"
        query = (
            f"""
            INSERT INTO {self._table_ident} (
                id, agent_id, text, tags, ts, tick, importance, event_id, source_id, target_id,
                tokens, embedding, embedding_norm, vector_backend
            ) VALUES (
                %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s,
                %s, {vector_sql}, %s, %s
            )
            ON CONFLICT (id) DO UPDATE SET
                text = EXCLUDED.text,
                tags = EXCLUDED.tags,
                ts = EXCLUDED.ts,
                tick = EXCLUDED.tick,
                importance = EXCLUDED.importance,
                event_id = EXCLUDED.event_id,
                source_id = EXCLUDED.source_id,
                target_id = EXCLUDED.target_id,
                tokens = EXCLUDED.tokens,
                embedding = EXCLUDED.embedding,
                embedding_norm = EXCLUDED.embedding_norm,
                vector_backend = EXCLUDED.vector_backend
            """
        )

        params: list[Any] = [
            entry.id,
            entry.agent_id,
            entry.text,
            json.dumps(list(entry.tags), ensure_ascii=False),
            entry.ts,
            entry.tick,
            float(entry.importance),
            entry.event_id,
            entry.source_id,
            entry.target_id,
            list(entry.tokens),
        ]
        if has_vector:
            params.append(self._vector_literal(entry.embedding))
        params.extend(
            [
                float(entry.embedding_norm if has_vector else 0.0),
                entry.vector_backend if has_vector else "none",
            ]
        )

        try:
            with self._pg_conn.cursor() as cur:
                cur.execute(query, params)
        except Exception as exc:
            self._pg_errors += 1
            LOGGER.warning("Failed to persist memory entry in pgvector type=%s detail=%r", type(exc).__name__, exc)

    def _trim_agent_pg(self, agent_id: str) -> None:
        if not self._pg_enabled or not self._pg_conn:
            return
        try:
            with self._pg_conn.cursor() as cur:
                cur.execute(
                    f"""
                    DELETE FROM {self._table_ident}
                    WHERE agent_id = %s
                      AND id IN (
                        SELECT id
                        FROM {self._table_ident}
                        WHERE agent_id = %s
                        ORDER BY tick DESC, ts DESC, id DESC
                        OFFSET %s
                      )
                    """,
                    (agent_id, agent_id, self.per_agent_limit),
                )
        except Exception as exc:
            self._pg_errors += 1
            LOGGER.warning("Failed to trim pgvector memory rows type=%s detail=%r", type(exc).__name__, exc)

    def _recall_pg(
        self,
        *,
        agent_id: str,
        query: str,
        query_tokens: frozenset[str],
        query_vector: tuple[float, ...],
        limit: int,
    ) -> list[dict]:
        if not self._pg_enabled or not self._pg_conn:
            return []
        if limit <= 0:
            return []

        vector_usable = query_vector and len(query_vector) == self.vector_dim
        candidate_limit = max(limit * 8, 40)
        rows: list[dict[str, Any]]

        try:
            with self._pg_conn.cursor() as cur:
                if vector_usable:
                    vector_literal = self._vector_literal(query_vector)
                    cur.execute(
                        f"""
                        SELECT
                            id,
                            event_id,
                            text,
                            tags,
                            tick,
                            ts,
                            importance,
                            vector_backend,
                            tokens,
                            CASE
                                WHEN embedding IS NULL THEN NULL
                                ELSE 1 - (embedding <=> %s::vector)
                            END AS semantic_sim
                        FROM {self._table_ident}
                        WHERE agent_id = %s
                        ORDER BY
                            CASE WHEN embedding IS NULL THEN 1 ELSE 0 END,
                            embedding <=> %s::vector,
                            tick DESC
                        LIMIT %s
                        """,
                        (vector_literal, agent_id, vector_literal, candidate_limit),
                    )
                else:
                    cur.execute(
                        f"""
                        SELECT
                            id,
                            event_id,
                            text,
                            tags,
                            tick,
                            ts,
                            importance,
                            vector_backend,
                            tokens,
                            NULL::DOUBLE PRECISION AS semantic_sim
                        FROM {self._table_ident}
                        WHERE agent_id = %s
                        ORDER BY tick DESC, ts DESC
                        LIMIT %s
                        """,
                        (agent_id, candidate_limit),
                    )
                rows = list(cur.fetchall())
        except Exception as exc:
            self._pg_errors += 1
            LOGGER.warning("Failed pgvector recall query type=%s detail=%r", type(exc).__name__, exc)
            return []

        if not rows:
            return []

        max_tick = max(int(row.get("tick") or 0) for row in rows)
        min_tick = min(int(row.get("tick") or 0) for row in rows)
        tick_span = max(1, max_tick - min_tick)

        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            row_tags = row.get("tags") or []
            if isinstance(row_tags, str):
                try:
                    row_tags = json.loads(row_tags)
                except Exception:
                    row_tags = []
            if not isinstance(row_tags, list):
                row_tags = [str(row_tags)]

            row_tokens_raw = row.get("tokens") or []
            if isinstance(row_tokens_raw, str):
                row_tokens = _tokenize(row_tokens_raw)
            else:
                row_tokens = frozenset(str(token) for token in row_tokens_raw if str(token).strip())

            semantic = 0.0
            semantic_sim = row.get("semantic_sim")
            if semantic_sim is not None:
                semantic = _clamp_float((float(semantic_sim) + 1.0) / 2.0, 0.0, 1.0)
            elif query_tokens and row_tokens:
                semantic = len(query_tokens & row_tokens) / max(len(query_tokens), len(row_tokens))

            tick = int(row.get("tick") or 0)
            recency = _clamp_float((tick - min_tick) / tick_span, 0.0, 1.0)
            importance = _clamp_float(float(row.get("importance") or 0.0), 0.0, 1.0)
            summary_bonus = 0.03 if "summary" in {str(tag) for tag in row_tags} else 0.0
            score = (importance * 0.35) + (semantic * 0.47) + (recency * 0.15) + summary_bonus

            scored.append(
                (
                    score,
                    {
                        "id": str(row.get("id") or ""),
                        "event_id": row.get("event_id"),
                        "text": str(row.get("text") or ""),
                        "tags": [str(tag) for tag in row_tags],
                        "tick": tick,
                        "ts": str(row.get("ts") or ""),
                        "vector_backend": str(row.get("vector_backend") or "none"),
                    },
                )
            )

        scored.sort(key=lambda item: (item[0], int(item[1]["tick"])), reverse=True)
        return [
            {
                **item,
                "score": round(score, 4),
            }
            for score, item in scored[: max(1, min(limit, 20))]
        ]

    def _latest_pg(self, *, agent_id: str, limit: int) -> list[dict]:
        if not self._pg_enabled or not self._pg_conn:
            return []
        capped_limit = max(1, min(limit, 20))
        try:
            with self._pg_conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, event_id, text, tags, tick, ts, importance, vector_backend
                    FROM {self._table_ident}
                    WHERE agent_id = %s
                    ORDER BY tick DESC, ts DESC
                    LIMIT %s
                    """,
                    (agent_id, capped_limit),
                )
                rows = list(cur.fetchall())
        except Exception as exc:
            self._pg_errors += 1
            LOGGER.warning("Failed latest query from pgvector type=%s detail=%r", type(exc).__name__, exc)
            return []

        rows.reverse()
        result: list[dict] = []
        for row in rows:
            row_tags = row.get("tags") or []
            if isinstance(row_tags, str):
                try:
                    row_tags = json.loads(row_tags)
                except Exception:
                    row_tags = []
            if not isinstance(row_tags, list):
                row_tags = [str(row_tags)]
            result.append(
                {
                    "id": str(row.get("id") or ""),
                    "event_id": row.get("event_id"),
                    "text": str(row.get("text") or ""),
                    "tags": [str(tag) for tag in row_tags],
                    "tick": int(row.get("tick") or 0),
                    "ts": str(row.get("ts") or ""),
                    "score": round(_clamp_float(float(row.get("importance") or 0.0), 0.0, 1.0), 4),
                    "vector_backend": str(row.get("vector_backend") or "none"),
                }
            )
        return result

    def _stats_pg(self) -> dict[str, int]:
        if not self._pg_enabled or not self._pg_conn:
            return {"entries": 0, "agents": 0, "summary_entries": 0}
        try:
            with self._pg_conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT
                        COUNT(*) AS entries,
                        COUNT(DISTINCT agent_id) AS agents,
                        COUNT(*) FILTER (WHERE tags ? 'summary') AS summary_entries
                    FROM {self._table_ident}
                    """
                )
                row = cur.fetchone()
                if not row:
                    return {"entries": 0, "agents": 0, "summary_entries": 0}
                return {
                    "entries": int(row.get("entries") or 0),
                    "agents": int(row.get("agents") or 0),
                    "summary_entries": int(row.get("summary_entries") or 0),
                }
        except Exception as exc:
            self._pg_errors += 1
            LOGGER.warning("Failed pgvector stats query type=%s detail=%r", type(exc).__name__, exc)
            return {"entries": 0, "agents": 0, "summary_entries": 0}

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
                        if self.backend == "pgvector" and len(vector) != self.vector_dim:
                            LOGGER.warning(
                                "Embedding vector dim=%s mismatches MEMORY_VECTOR_DIM=%s; storing without pgvector embedding.",
                                len(vector),
                                self.vector_dim,
                            )
                            return tuple(), 0.0, "none"
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
        self._persist_entry_pg(summary_entry)
        self._trim_agent_pg(agent_id)
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
        self._persist_entry_pg(entry)
        self._trim_agent_pg(agent_id)
        self._enforce_limits(agent_id)
        return entry.id

    def recall(self, *, agent_id: str, query: str, limit: int = 3) -> list[dict]:
        if not self.enabled:
            return []
        if limit <= 0:
            return []

        query_tokens = _tokenize(query)
        query_vector: tuple[float, ...] = tuple()
        query_vector_norm = 0.0
        if query.strip():
            query_vector, query_vector_norm, _ = self._embed_text(query)

        pg_result = self._recall_pg(
            agent_id=agent_id,
            query=query,
            query_tokens=query_tokens,
            query_vector=query_vector,
            limit=limit,
        )
        if pg_result:
            return pg_result

        items = list(self._entries.get(agent_id, ()))
        if not items:
            return []

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
        pg_result = self._latest_pg(agent_id=agent_id, limit=limit)
        if pg_result:
            return pg_result
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
        pg_stats = self._stats_pg()
        local_entries = sum(len(entries) for entries in self._entries.values())
        local_summary_entries = sum(
            1 for entries in self._entries.values() for entry in entries if "summary" in entry.tags
        )
        total_entries = pg_stats["entries"] if self._pg_enabled else local_entries
        summary_entries = pg_stats["summary_entries"] if self._pg_enabled else local_summary_entries
        agents_count = pg_stats["agents"] if self._pg_enabled else len(self._entries)
        return {
            "enabled": self.enabled,
            "backend": "pgvector" if self._pg_enabled else "memory",
            "agents": agents_count,
            "entries": total_entries,
            "summary_entries": summary_entries,
            "summary_entries_created": self._summary_entries_created,
            "per_agent_limit": self.per_agent_limit,
            "vector_enabled": self.vector_enabled,
            "vector_dim": self.vector_dim,
            "embedding_model": self.embedding_model if self.embedding_model else "none",
            "embedding_hash_fallback": self.embedding_hash_fallback,
            "embedding_failures": self._embedding_failures,
            "summary_enabled": self.summary_enabled,
            "summary_model": self.summary_model if self.summary_model else "none",
            "summary_failures": self._summary_failures,
            "pg_enabled": self._pg_enabled,
            "pg_table": self.table_name,
            "pg_errors": self._pg_errors,
        }
