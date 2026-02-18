# Skebobia Contract

Документ фиксирует актуальный контракт между `gateway`, `server`, `dashboard`, `scene`.

## 1. Скоуп
- Источник истины по состоянию мира: только `server`.
- Клиенты (`dashboard`, `scene`) читают состояние сервера и отправляют управление через HTTP.
- Realtime-доставка — через единый WebSocket поток `WS /ws/stream`.

## 2. Сетевой контракт

### 2.1 Маршруты gateway (`gateway/nginx.conf`)
- `/` -> `dashboard:80`
- `/scene/` -> `scene:80`
- `/api/` -> `server:8000`
- `/ws/` -> `server:8000`

Gateway обязан:
- проксировать WebSocket с `Upgrade/Connection`;
- не кешировать `/ws/*`;
- держать `/scene/` как директорию (редирект `/scene` -> `/scene/`).

### 2.2 Прямой доступ к server
В `docker-compose.yml` открыт `8000:8000`.
Это нужно, потому что оба клиента по умолчанию обращаются к `:8000` напрямую.

## 3. Server Contract (`server/app/main.py`)

### 3.1 REST API
- `GET /api/health` -> `{ "status": "ok" }`
- `GET /api/state` -> полный снапшот:
  - `tick`, `speed`, `agents`, `relations`, `events`
  - `llm_stats`, `world_event_stats`, `memory_stats`
  - `runtime` (добавляется в `main.py`: `last_tick_ms`, `avg_tick_ms`)
- `GET /api/agents` -> краткие карточки агентов (`id`, `name`, `avatar`, `mood`, `mood_label`, `current_plan`)
- `GET /api/agents/{agent_id}` -> расширенный профиль агента (память, recent events, relations snapshot, cooldowns)
- `GET /api/relations` -> `{ nodes, edges }`
- `GET /api/events?limit=1..500&agent_id=...` -> лента событий

Control endpoints:
- `POST /api/control/event`
  - body: `{ "text": "...", "importance": 0..1? }`
  - resp: `{ "event_id": "e...", "reaction_event_ids": ["e...", ...] }`
- `POST /api/control/message`
  - body: `{ "agent_id": "a1", "text": "..." }`
  - resp: `{ "accepted": true, "event_id": "...", "reply_event_id": "...|null", "reply_pending": bool }`
- `POST /api/control/speed`
  - body: `{ "speed": 0.1..5.0 }`
  - resp: `{ "speed": number }`
- `POST /api/control/agent/add`
  - body: `{ "id"?, "name", "traits"?, "avatar"?, "mood"?, "pos_x"?, "pos_z"? }`
  - resp: `{ "accepted": true, "agent": {...summary...} }`
- `POST /api/control/agent/remove`
  - body: `{ "agent_id": "aN" }`
  - resp: `{ "accepted": true, "agent_id": "aN" }`

### 3.2 WebSocket
- Endpoint: `WS /ws/stream`
- При подключении сервер сразу отправляет:
  - один `agents_state`
  - один `relations`
  - до 10 последних `event`
- Далее потоковые сообщения:
  - `{ "type": "agents_state", "payload": [...] }`
  - `{ "type": "event", "payload": {...} }`
  - `{ "type": "relations", "payload": {...} }`

### 3.3 Минимальные гарантии payload
`agents_state.payload[i]` содержит минимум:
- `id`, `name`, `avatar`, `mood`, `mood_label`, `current_plan`
- `pos`, `look_at`, `last_action`, `last_say`, `target_id`, `tick`

`event.payload` содержит минимум:
- `id`, `ts`, `source_type`, `source_id`, `text`, `tags`, `tick`
- опционально: `target_id`, `thread_id`, `expects_reply`, `can_reply`, `importance`, `anchor`, `severity`, `evidence_ids`

`relations.payload`:
- `nodes: [{id, name}]`
- `edges: [{from, to, value}]`, `value` в диапазоне `-100..100`

## 4. Dashboard Contract (`dashboard/src/App.jsx`)
Dashboard обязан:
- на старте читать `GET /api/state`;
- держать подключение к `WS /ws/stream`;
- обновлять события/агентов/отношения по сообщениям WS;
- отправлять control-команды:
  - `/api/control/event`
  - `/api/control/message`
  - `/api/control/speed`
  - `/api/control/agent/add`
  - `/api/control/agent/remove`
- читать инспектор через `GET /api/agents/{id}`.

Фоллбек:
- периодический refresh статистики `GET /api/state` раз в ~3 сек.

## 5. Scene Contract (`scene/unity/Assets`)
Scene обязана:
- загрузить старт через `GET /api/state` (`StateLoader`);
- подключиться к `WS /ws/stream` (`WsClient`);
- обрабатывать:
  - `agents_state` -> позиция, имя, настроение, реплики/план;
  - `event` -> speech bubble для агентских диалогов;
  - `relations` можно игнорировать.

Текущее значение по умолчанию:
- `NetConfig.originOverride = "http://localhost:8000"`.

## 6. Конфигурационный контракт
Сервер читает `.env` из корня репозитория (`server/app/main.py`), если переменные не заданы в окружении.

Ключевые группы:
- тики и скорость: `TICK_INTERVAL_SEC`, `RELATIONS_*`
- LLM: `LLM_DECIDER_*`, `LLM_FIRST_HARD_MODE`, `LLM_FORCE_USER_REPLY_VIA_LLM`
- world events: `WORLD_EVENTS_*`, `STARTUP_WORLD_EVENT_*`
- память: `MEMORY_*`
- PostgreSQL: `POSTGRES_*`, `DATABASE_URL`/`MEMORY_DATABASE_URL`

Базовый шаблон переменных: `.env.example`.

## 7. Definition of Done
Система считается рабочей, если одновременно:
- открываются `http://localhost/` и `http://localhost/scene/`;
- `GET /api/health` отвечает `200` (через gateway и/или напрямую на `:8000`);
- `WS /ws/stream` стабильно шлет `event`, `agents_state`, `relations`;
- control-endpoints реально меняют состояние симуляции;
- scene визуально обновляет агентов и показывает диалоговые bubble.
