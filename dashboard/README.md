# Dashboard

React/Vite-клиент для мониторинга и управления симуляцией.

## Что умеет
- realtime-лента событий;
- граф отношений между агентами;
- карточки агентов и инспектор (`GET /api/agents/{id}`);
- управление:
  - `POST /api/control/event`
  - `POST /api/control/message`
  - `POST /api/control/speed`
  - `POST /api/control/agent/add`
  - `POST /api/control/agent/remove`

## Источники данных
- старт: `GET /api/state`
- realtime: `WS /ws/stream`
- доп. refresh статистики: `GET /api/state` каждые ~3 сек

## Базовые URL по умолчанию
В `src/App.jsx`:
- `API_BASE = http(s)://<host>:8000`
- `WS_URL = ws(s)://<host>:8000/ws/stream`

Это соответствует текущему `docker-compose`, где открыт `8000:8000`.

## Переопределение URL (если нужен gateway `/api` и `/ws`)
Задайте переменные при сборке:
- `VITE_API_BASE=http://localhost`
- `VITE_WS_URL=ws://localhost/ws/stream`

## Локальный запуск
```bash
cd dashboard
npm install
npm run dev
```

## Docker
Dashboard собирается мультистейджем в `dashboard/Dockerfile` и обслуживается `nginx`.
