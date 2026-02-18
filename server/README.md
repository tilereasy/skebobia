# Server (FastAPI)

Сервер симуляции: тики мира, агентные решения, отношения, память и realtime-поток.

## Основные модули
- `app/main.py` — FastAPI, HTTP/WS endpoints, lifecycle, tick loop.
- `app/sim/engine.py` — `StubWorld`, логика тика, события, control-операции.
- `app/sim/llm_decider.py` + `app/llm/client.py` — LLM-decider и клиент OpenAI-compatible API.
- `app/memory/store.py` — эпизодическая память (`memory`/`pgvector`) + recall/summarization.
- `app/db/models.py` — pydantic-модели входов control API.

## API
### GET
- `/api/health`
- `/api/state`
- `/api/agents`
- `/api/agents/{agent_id}`
- `/api/relations`
- `/api/events?limit=&agent_id=`

### POST
- `/api/control/event`
- `/api/control/message`
- `/api/control/speed`
- `/api/control/agent/add`
- `/api/control/agent/remove`

### WS
- `/ws/stream`
  - типы сообщений: `event`, `agents_state`, `relations`

## Поведение на тик
- шаг симуляции;
- выбор агентных действий (LLM-first + fallback);
- обновление mood/relations/position;
- запись событий и памяти;
- публикация обновлений в WS.

## Конфигурация
Сервер автоматически читает `.env` из корня репозитория (если переменные не заданы извне).

Ключевые группы:
- темп симуляции: `TICK_INTERVAL_SEC`, `RELATIONS_*`
- LLM: `LLM_DECIDER_*`, `LLM_FIRST_HARD_MODE`, `LLM_FORCE_USER_REPLY_VIA_LLM`
- world events: `WORLD_EVENTS_*`, `STARTUP_WORLD_EVENT_*`
- память: `MEMORY_*`

Актуальный шаблон: `../.env.example`.

## Локальный запуск
```bash
cd server
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Docker
Используется `python:3.12-slim`, старт через:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
