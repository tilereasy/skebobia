# Server (FastAPI) — “психика + симуляция + память + realtime API”

#### **Цель**
Сервер реализует симуляцию “виртуального мира”, в котором несколько автономных AI-агентов:
- имеют личность (traits), настроение (mood), отношения (relations),
- ведут события и диалоги,
- сохраняют эпизодическую память (episodic memory store),
- транслируют жизнь в реальном времени на клиент (WebSocket),
- принимают вмешательства пользователя (события, сообщения, скорость времени).

#### **MVP-функционал (обязательный)**
*Агенты и симуляция:*

- Поддержка N агентов (по умолчанию 3–5), каждый агент имеет:
id, name, avatar (изображение(url) или просто цвет), traits (строка/JSON),

- mood (число/категория), mood_label/emoji,

- текущую цель/план (коротко), последний action.


*“Тик” симуляции: сервер раз в T секунд делает шаг мира (tick), где часть агентов:*

- читает входящие события/сообщения,
- кладет реплики в очередь ответов (priority queue),

- выбирает действие (LLM-first, rule-fallback только при ошибках/таймаутах),
- обрабатывает только ограниченное число reply-задач за тик (для стабильной латентности),
- добавляет ограниченное число proactive-агентов за тик, чтобы диалог не замирал при пустой очереди,

- генерирует событие в ленту,

- обновляет mood и relations.

*Память:*

Эпизодическая память хранится как записи вида:

- {agent_id, ts, text, tags, importance, event_id, source_id, target_id}

- На каждый тик в prompt передаются top-k воспоминаний, отобранных по релевантности и свежести.

*Эмоциональный интеллект:*

- У каждого агента есть mood (например шкала -100..+100 или категории: angry/sad/neutral/happy/excited).

Mood меняется при:

- получении события (world event),

- личном взаимодействии (сообщение, конфликт),

- воспоминании важного эпизода.

- Mood влияет на стиль ответа (тон/лексика) — хотя бы простыми правилами в system prompt.

*Мультиагентность и отношения:*

Агенты могут отправлять сообщения друг другу через сервер.

Отношения — матрица/таблица вида:

- relation(from_id, to_id, value -100..+100)

Значение меняется по итогам взаимодействий/событий.

*Realtime-доставка:*

WebSocket поток для:

- ленты событий (event feed),

- обновлений состояния агентов (mood/plan),

- обновлений графа отношений.

*Дополнительный функционал(**если успеем**):*

- “Инспектор агента”: эндпоинт, который отдаёт “ключевые воспоминания” и “текущий план”.

- История агента: список последних событий с фильтрацией по agent_id.

#### **Архитектура**
Модули (рекомендуемая структура)

app/main.py — FastAPI, роуты, ws

app/sim/engine.py — tick-loop, scheduler, скорость времени

app/agents/agent.py — состояние агента и сериализация

app/llm/client.py — провайдер LLM (OpenAI/Gemini/YandexGPT) + таймауты/ретраи

app/memory/store.py — in-memory episodic memory + retrieval

app/db/models.py — сущности (Agent, Event, Relation, Memory)

app/api/ — REST endpoints

app/ws/ — WebSocket endpoints + broadcaster

#### **Хранилище**

- runtime state мира хранится в памяти процесса сервера
- эпизодическая память агентов хранится в in-memory store
- PostgreSQL/pgvector в compose остаётся опциональным заделом под дальнейшую персистентность

#### **API**

##### *REST*
GET /api/health
200 OK

GET /api/agents
Список агентов: id, name, avatar, mood, mood_label, current_plan

GET /api/state
Получить стартовый снапшот

GET /api/agents/{id}
Профиль агента: traits, mood, current_plan, key_memories(top-k), recent_events

GET /api/relations
Граф отношений: nodes=[agents], edges=[{from,to,value}]

GET /api/events?limit=200&agent_id=...
Лента событий (последние N)

POST /api/control/event
Пользователь добавляет событие в мир
Body: { text: string, importance?: number }
Response: { event_id }

POST /api/control/message
Пользователь отправляет сообщение агенту
Body: { agent_id: string, text: string }
Response: { accepted: true }

POST /api/control/speed
Скорость времени
Body: { speed: number } (например 0.5..5)
Response: { speed }

##### *WebSocket*
WS /ws/stream
Сервер шлёт сообщения типов:
{type:"event", payload:{...}}
{type:"agents_state", payload:[...]}
{type:"relations", payload:{nodes,edges}}
Логика “reflect → goal → act” (минимум)

На каждом tick агент:
Reflect: что произошло с последнего тика? что я чувствую? что важно?
Recall: найти top-k воспоминаний по текущей ситуации/сообщениям
Goal: сформулировать цель на ближайший шаг (1 строка)
Act: выбрать действие:
- сказать что-то в чат (agent->agent),
- отреагировать на world event,
- создать событие.
- перейти в позицию pos(x, z).
Всё это можно реализовать одной LLM-генерацией с жёстким JSON-выводом (чтобы не развалилось).

сервер хранит pos для каждого агента (x,z)
раз в тик слегка меняет pos (или меняет при действиях)
Unity просто интерполирует

#### Конфигурация (env)

При запуске сервера автоматически читается `.env` из корня репозитория
(`skebobia/.env`). Уже заданные переменные окружения не перезаписываются.

TICK_INTERVAL_SEC=2
RELATIONS_INTERVAL_TICKS=5

# LLM-decider (LLM-first, с мягким salvage partial-ответов)
LLM_DECIDER_ENABLED=0
LLM_DECIDER_BASE_URL=https://bothub.chat/api/v2/openai/v1
LLM_DECIDER_MODEL=
LLM_DECIDER_API_KEY=
LLM_DECIDER_TIMEOUT_SEC=30
LLM_DECIDER_TEMPERATURE=0.2
LLM_DECIDER_MAX_OUTPUT_TOKENS=1200
LLM_DECIDER_MAX_RETRIES=0
LLM_DECIDER_MAX_AGENTS_PER_TICK=16
LLM_DECIDER_DEBUG=0
LLM_DECIDER_STRICT_JSON_SCHEMA=0
LLM_DECIDER_STRICT_SCHEMA_VALIDATION=0
LLM_DECIDER_BACKFILL_RETRIES=2
LLM_TARGET_RESPONSE_RATIO=0.9
LLM_RESPONSE_RATIO_WINDOW=240
LLM_PROMPT_RECENT_EVENTS=6
LLM_PROMPT_INBOX=4
LLM_PROMPT_MEMORIES=5
REPLY_QUEUE_MAX_REPLIES_PER_TICK=2
REPLY_QUEUE_MAX_WAIT_TICKS=10
REPLY_QUEUE_MAX_SKIPS=2
REPLY_QUEUE_MAX_SIZE=512
LLM_PROACTIVE_AGENTS_PER_TICK=1
STARTUP_WORLD_EVENT_ENABLED=1
STARTUP_WORLD_EVENT_IMPORTANCE=0.85

# Episodic memory
MEMORY_ENABLED=1
MEMORY_EPISODES_PER_AGENT=400

#### Запуск

Поднимаем через Docker Compose вместе с клиентом.

## Минимальный контракт для клиентов (важно)
Чтобы Unity и Dashboard рисовали одно и то же, сервер гарантирует:

### agents_state payload (минимум)
[
  {
    "id": "a1",
    "name": "Alice",
    "mood": 20,
    "mood_label": "happy",
    "current_plan": "Find SkeBob and apologize",
    "pos": {"x": 12.3, "y": 0.0, "z": -4.1},
    "look_at": {"x": 0.0, "y": 0.0, "z": 1.0},
    "last_action": "say",
    "last_say": "Прасти меня.",
    "tick": 42
  }
]

### event payload (минимум)
{
  "id": "e123",
  "ts": "2026-02-16T12:34:56Z",
  "source_type": "agent|world",
  "source_id": "a1|null",
  "text": "Alice said: ...",
  "tags": ["dialogue"],
  "tick": 52
}

### relations payload (минимум)
{
  "nodes": [{"id":"a1","name":"Alice"}, ...],
  "edges": [{"from":"a1","to":"a2","value":-15}, ...]
}
