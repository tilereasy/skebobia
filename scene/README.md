# Scene (Unity WebGL)

Визуализация симуляции агентов в браузере.

## Структура
- `scene/unity/` — исходники Unity-проекта.
- `scene/webgl/` — готовая WebGL-сборка, которую раздает `nginx`.

## Runtime-схема
- `StateLoader` грузит стартовое состояние через `GET /api/state`.
- `WsClient` держит подключение к `WS /ws/stream`.
- `StreamRouter` обрабатывает:
  - `agents_state` -> обновление/создание агентов;
  - `event` -> speech bubble для агентских реплик;
  - `relations` игнорируется.
- `AgentRegistry` отвечает за визуальные объекты агентов, цвет/иконки эмоций, подписи и bubble.

## Сетевые настройки
По умолчанию `NetConfig` использует:
- `originOverride = http://localhost:8000`
- `state = /api/state`
- `stream = /ws/stream`

То есть scene ходит напрямую в `server:8000`.

## Docker
`scene/Dockerfile` копирует `scene/webgl/` в `nginx`.

## Что проверить после запуска
- `http://localhost/scene/` открывается;
- агенты появляются и двигаются;
- периодически появляются реплики (speech bubble).
