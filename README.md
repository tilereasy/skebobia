# Skebobia (Unity scene + Web dashboard)

## Что это
Система симуляции “живого мира” с несколькими AI-агентами (личность, эмоции, отношения, память).
Есть два клиента:
- **/scene/** — визуализация мира (Unity WebGL): агенты ходят, общаются, показывают эмоции.
- **/** — дашборд (Web): лента событий, граф отношений, инспектор агента, управление событиями/скоростью.

Все работает через один домен (Nginx gateway): единый API + единый WebSocket.

## URL после запуска
- Dashboard: http://localhost/
- Unity scene: http://localhost/scene/
- API: http://localhost/api/health
- WebSocket: ws://localhost/ws/stream

## Быстрый старт
1) Скопировать env:
   cp .env.example .env
2) Запуск:
   docker compose up --build

## Компоненты
- server/ — FastAPI симуляция + LLM + память + realtime WS
- dashboard/ — веб-пульт (граф/лента/инспектор/контролы)
- scene/ — Unity WebGL визуализация (позиции, реплики, эмоции)
- gateway/ — nginx reverse proxy (один домен, разные пути)

## Протокол данных (общий для клиентов)
Сервер шлёт в WebSocket (/ws/stream) сообщения:
- type="event" — новое событие в мире
- type="agents_state" — состояние агентов (позиция, mood, текущий план)
- type="relations" — граф отношений

Клиенты отправляют управление через REST:
- POST /api/control/event
- POST /api/control/message
- POST /api/control/speed
