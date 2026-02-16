# Skebobia Contract

## 1) Scope
Документ фиксирует единый технический контракт между компонентами:
- `gateway` (Nginx)
- `server` (FastAPI + simulation + WS)
- `dashboard` (Web client)
- `scene` (Unity WebGL client)

Цель: все клиенты читают и отображают один и тот же источник истины от `server` через единые HTTP/WS интерфейсы за `gateway`.

## 2) Routing Contract (gateway)
Единая точка входа под одним доменом:
- `/` -> `dashboard`
- `/scene/` -> `scene` (Unity WebGL static)
- `/api/` -> `server` REST API
- `/ws/` -> `server` WebSocket

Обязательства gateway:
- проксировать WebSocket с корректными `Upgrade/Connection` заголовками
- не ломать поток `/ws/stream` кешированием
- сохранять корректную работу относительных путей Unity под `/scene/` (как директория)

## 3) Base URLs (local)
- Dashboard: `http://localhost/`
- Scene: `http://localhost/scene/`
- Health: `http://localhost/api/health`
- WS stream: `ws://localhost/ws/stream`

## 4) Server Contract

### 4.1 REST API (MVP)
- `GET /api/health` -> `200 OK`
- `GET /api/agents` -> список агентов (`id`, `name`, `avatar`, `mood`, `mood_label`, `current_plan`)
- `GET /api/state` -> стартовый снапшот мира (`server_time`, `agents`, `relations`, `events`)
- `GET /api/agents/{id}` -> профиль агента (`traits`, `mood`, `current_plan`, `key_memories`, `recent_events`)
- `GET /api/relations` -> граф отношений (`nodes`, `edges`)
- `GET /api/events?limit=...&agent_id=...` -> лента событий
- `POST /api/control/event` body: `{ "text": "string", "importance": number? }` -> `{ "event_id": "..." }`
- `POST /api/control/message` body: `{ "agent_id": "string", "text": "string" }` -> `{ "accepted": true }`
- `POST /api/control/speed` body: `{ "speed": number }` -> `{ "speed": number }`

### 4.2 WebSocket
- Endpoint: `WS /ws/stream`
- Типы исходящих сообщений:
  - `event`
  - `agents_state`
  - `relations`

Минимальный формат сообщений:
```json
{ "type": "event", "payload": { "...": "..." } }
{ "type": "agents_state", "payload": [ { "...": "..." } ] }
{ "type": "relations", "payload": { "nodes": [], "edges": [] } }
```

### 4.3 Payload guarantees (минимум)
`agents_state.payload[]`:
```json
{
  "id": "a1",
  "name": "Alice",
  "mood": 20,
  "mood_label": "happy",
  "current_plan": "Find SkeBob and apologize",
  "pos": { "x": 12.3, "y": 0.0, "z": -4.1 },
  "look_at": { "x": 0.0, "y": 0.0, "z": 1.0 },
  "last_action": "say",
  "last_say": "Прасти меня.",
  "tick" : 42
}
```

`event.payload`:
```json
{
  "id": "e123",
  "ts": "2026-02-16T12:34:56Z",
  "source_type": "agent|world",
  "source_id": "a1|null",
  "text": "Alice said: ...",
  "tags": ["dialogue"],
  "tick": 52
}
```

`relations.payload`:
```json
{
  "nodes": [{ "id": "a1", "name": "Alice" }],
  "edges": [{ "from": "a1", "to": "a2", "value": -15 }]
}
```

## 5) Client Contract

### 5.1 Dashboard
Обязан:
- подключаться к `ws://<host>/ws/stream`
- показывать realtime event feed (200-300 последних событий)
- отображать карточки агентов (`name`, `mood_label`, `current_plan`)
- отображать граф отношений (`value` в диапазоне `-100..+100`)
- отправлять control-команды:
  - `POST /api/control/event`
  - `POST /api/control/message`
  - `POST /api/control/speed`
- поддерживать инспектор агента через `GET /api/agents/{id}`

Режим данных:
- старт: `GET /api/state`
- realtime: `WS /ws/stream`
- fallback (допустим): polling `/api/state` каждые 1-2 сек

### 5.2 Scene (Unity WebGL)
Обязан:
- быть доступной по `/scene/`
- подключаться к `ws://<host>/ws/stream`
- обрабатывать:
  - `agents_state`: обновление позиций, mood, plan
  - `event`: отображение speech bubble для диалогов (например по `tags`/типу dialogue)
  - `relations`: может игнорировать
- создавать N визуальных агентов при старте из `GET /api/state` или первого `agents_state`
- для каждого агента показывать минимум:
  - имя
  - mood-индикатор
  - позицию `pos.x`, `pos.z`
- плавно интерполировать перемещение между обновлениями

## 6) Simulation/Data Rules (server-side MVP)
- сервер является источником истины по состоянию мира, позиции, mood и отношениям
- симуляция выполняется тиком (`TICK_INTERVAL_SEC`, базово 2 сек)
- агентный state минимум: `id`, `name`, `traits`, `mood`, `mood_label`, `current_plan`, `last_action`, `pos`
- отношения: направленные значения `value` в диапазоне `-100..+100`
- память (эпизодическая): хранение + семантический поиск + суммаризация при росте контекста

## 7) Configuration Contract (env)
Минимально поддерживаемые переменные:
- `LLM_PROVIDER=openai|gemini|yandex|bothub|openrouter`
- `LLM_API_KEY=...`
- `VECTOR_DB_URL=...`
- `DATABASE_URL=...`
- `TICK_INTERVAL_SEC=2`
- `DEFAULT_SPEED=1.0`

## 8) End-to-End DoD
Система считается готовой, если одновременно выполняется:
- открывается `http://localhost/` (dashboard)
- открывается `http://localhost/scene/` (Unity scene)
- `GET /api/health` возвращает `200`
- `WS /ws/stream` стабильно доставляет `event`, `agents_state`, `relations`
- Dashboard управляет симуляцией через control endpoints
- Scene показывает движение агентов, настроение и периодические реплики
