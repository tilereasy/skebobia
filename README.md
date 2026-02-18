# Skebobia

Мультиагентная симуляция с FastAPI-сервером, Web-dashboard и Unity WebGL-сценой.

## Разработка в рамках хакатона «Кибер-Рывок»
Этот репозиторий создан в рамках хакатона «Кибер-Рывок».

Команда: t34M4 n3 R4k0V

Участники:

* [Listy-V](https://github.com/Listy-V)
* [Kwerdu](https://github.com/Kwerd-u)
* [Zama47](https://github.com/Zama47)
* [tilereasy](https://github.com/tilereasy)
* [alexeydarkness] (https://github.com/alexeydarkness)


**Проект разрабатывался в ограниченные сроки как proof‑of‑concept**.

## Что внутри
- `server/` — FastAPI + тик-движок симуляции + WebSocket + память (`memory`/`pgvector`).
- `dashboard/` — React/Vite-пульт: лента событий, граф отношений, инспектор агента, control-формы.
- `scene/` — Unity-проект (`scene/unity`) и готовая WebGL-сборка (`scene/webgl`).
- `gateway/` — Nginx-шлюз с маршрутами `/`, `/scene/`, `/api/`, `/ws/`.
- `postgres` (в `docker-compose`) — `pgvector/pg16` для долговременной памяти.

## Как работает обмен данными
- Источник истины: `server`.
- Клиенты получают старт через `GET /api/state` и realtime через `WS /ws/stream`.
- На каждый тик сервер шлет `agents_state`; `event` и `relations` шлются по факту изменений.

Важно про адреса по умолчанию:
- Dashboard в коде использует `http://<host>:8000` и `ws://<host>:8000/ws/stream` (если не заданы `VITE_*`).
- Unity `NetConfig` по умолчанию использует `originOverride = http://localhost:8000`.
- Поэтому в compose оставлен проброс `8000:8000`.

## Запуск (Docker Compose)
1. Подготовьте переменные:
```bash
cp .env.example .env
```
2. При необходимости заполните LLM-ключи в `.env`.
3. Запустите стек:
```bash
docker compose up --build
```

## URL после запуска
- Dashboard: `http://localhost/`
- Unity Scene: `http://localhost/scene/`
- API (через gateway): `http://localhost/api/health`
- API (напрямую в server): `http://localhost:8000/api/health`
- WebSocket (через gateway): `ws://localhost/ws/stream`
- WebSocket (напрямую в server): `ws://localhost:8000/ws/stream`

## Остановка
```bash
docker compose down
```

Полный контракт API/WS и обязательства компонентов: `CONTRACT.md`.
