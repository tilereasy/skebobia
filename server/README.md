# Server (FastAPI) — “психика + симуляция + память + realtime API”

#### **Цель**
Сервер реализует симуляцию “виртуального мира”, в котором несколько автономных AI-агентов:
- имеют личность (traits), настроение (mood), отношения (relations),
- ведут события и диалоги,
- сохраняют долговременную эпизодическую память (vector store),
- транслируют жизнь в реальном времени на клиент (WebSocket),
- принимают вмешательства пользователя (события, сообщения, скорость времени).

#### **MVP-функционал (обязательный)**
*Агенты и симуляция:*

- Поддержка N агентов (по умолчанию 3–5), каждый агент имеет:
id, name, avatar_url (или цвет), traits (строка/JSON),

- mood (число/категория), mood_label/emoji,

- текущую цель/план (коротко), последний action.


*“Тик” симуляции: сервер раз в T секунд делает шаг мира (tick), где часть агентов:*

- читает входящие события/сообщения,

- выбирает действие (через LLM или правило-фоллбек),

- генерирует событие в ленту,

- обновляет mood и relations.

*Долговременная память:*

Эпизодическая память хранится как записи вида:

- {agent_id, ts, text, tags, embedding, importance(optional), summary_ref(optional)}

- Векторная БД используется для семантического поиска воспоминаний.

При переполнении “контекста” (или при достижении лимита записей) запускается суммаризация:
- старые записи сворачиваются в summary (кратко),

- summary тоже сохраняется и может быть найден семантически.

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

app/agents/agent.py — логика агента (reflect → goal → act)

app/llm/client.py — провайдер LLM (OpenAI/Gemini/YandexGPT) + таймауты/ретраи

app/memory/store.py — vector store + суммаризация

app/db/models.py — сущности (Agent, Event, Relation, Memory)

app/api/ — REST endpoints

app/ws/ — WebSocket endpoints + broadcaster

#### **Хранилище**

Postgres для событий/агентов/отношений

Векторная БД для memory (pgvector)

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

LLM_PROVIDER=openai|gemini|yandex|bothub|openrouter

LLM_API_KEY=...

VECTOR_DB_URL=...

DATABASE_URL=...

TICK_INTERVAL_SEC=2

DEFAULT_SPEED=1.0

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
    "last_say": "Прасти меня."
  }
]

### event payload (минимум)
{
  "id": "e123",
  "ts": "2026-02-16T12:34:56Z",
  "source_type": "agent|world",
  "source_id": "a1|null",
  "text": "Alice said: ...",
  "tags": ["dialogue"]
}

### relations payload (минимум)
{
  "nodes": [{"id":"a1","name":"Alice"}, ...],
  "edges": [{"from":"a1","to":"a2","value":-15}, ...]
}
