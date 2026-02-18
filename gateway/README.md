# Gateway (Nginx)

`gateway` — единая входная точка в compose-стеке.

## Маршруты
- `/` -> `dashboard:80`
- `/scene/` -> `scene:80`
- `/api/` -> `server:8000`
- `/ws/` -> `server:8000`

## Поведение
- Для `/ws/*` включен корректный WS proxy (`Upgrade`/`Connection`).
- Для `/ws/*` отключено кеширование и буферизация.
- `/scene` перенаправляется в `/scene/`, чтобы Unity корректно грузила относительные пути.

## Проверка
- `http://localhost/` открывает dashboard.
- `http://localhost/scene/` открывает Unity scene.
- `http://localhost/api/health` возвращает `200`.
- `ws://localhost/ws/stream` принимает сообщения.
