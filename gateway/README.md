# Gateway (Nginx) единая точка входа

## Цель
Свести все сервисы под один домен:
- `/` → dashboard
- `/scene/` → unity webgl static
- `/api/` → server (FastAPI)
- `/ws/` → server websocket

## Требования
- проксирование WebSocket (Upgrade/Connection headers)
- отключить кеширование для ws
- корректные пути (Unity любит относительные пути — важно, чтобы /scene/ был “директорией”)

## Definition of Done
- открывается `/` и `/scene/`
- `/api/health` отвечает 200
- WS работает через `/ws/stream`
