from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from random import uniform

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.db.models import ControlEventIn, ControlMessageIn, ControlSpeedIn
from app.sim.engine import StubWorld

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WsHub:
    """WebSocket hub для управления подключениями и рассылки сообщений"""
    
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def add(self, ws: WebSocket) -> None:
        """Добавить WebSocket соединение"""
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)
        logger.info(f"WebSocket connected. Total clients: {len(self._clients)}")

    async def remove(self, ws: WebSocket) -> None:
        """Удалить WebSocket соединение"""
        async with self._lock:
            self._clients.discard(ws)
        logger.info(f"WebSocket disconnected. Total clients: {len(self._clients)}")

    async def send(self, ws: WebSocket, message: dict, timeout: float = 5.0) -> None:
        """Отправить сообщение конкретному клиенту с таймаутом"""
        try:
            await asyncio.wait_for(
                ws.send_text(json.dumps(message, ensure_ascii=False)),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning("WebSocket send timeout")
            raise RuntimeError("WebSocket send timeout")

    async def broadcast(self, message: dict) -> None:
        """Отправить сообщение всем подключенным клиентам"""
        async with self._lock:
            clients = list(self._clients)
        
        if not clients:
            return

        serialized = json.dumps(message, ensure_ascii=False)
        stale: list[WebSocket] = []
        
        for ws in clients:
            try:
                await ws.send_text(serialized)
            except RuntimeError as e:
                logger.warning(f"RuntimeError during broadcast: {e}")
                stale.append(ws)
            except Exception as e:
                logger.error(f"Error during broadcast: {e}")
                stale.append(ws)
        
        # Удалить неактивные соединения
        if stale:
            async with self._lock:
                for ws in stale:
                    self._clients.discard(ws)
            logger.info(f"Removed {len(stale)} stale connections")

    async def close_all(self) -> None:
        """Закрыть все WebSocket соединения (для graceful shutdown)"""
        async with self._lock:
            clients = list(self._clients)
        
        for ws in clients:
            try:
                await ws.close(code=1001, reason="Server shutdown")
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")


# Конфигурация из переменных окружения
TICK_MIN_SEC = float(os.getenv("TICK_MIN_SEC", "0.5"))
TICK_MAX_SEC = float(os.getenv("TICK_MAX_SEC", "2.0"))
RELATIONS_INTERVAL_TICKS = int(os.getenv("RELATIONS_INTERVAL_TICKS", "5"))

# Инициализация приложения
app = FastAPI(title="Skebobia Day 0 Stub Server", version="0.1.0")

# CORS middleware - КРИТИЧЕСКИ ВАЖНО для работы POST запросов из браузера
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене заменить на конкретные домены
    allow_credentials=True,
    allow_methods=["*"],  # Разрешить все HTTP методы (GET, POST, OPTIONS и т.д.)
    allow_headers=["*"],  # Разрешить все заголовки
)

world = StubWorld(relations_interval_ticks=RELATIONS_INTERVAL_TICKS)
hub = WsHub()


async def tick_loop() -> None:
    """Основной цикл симуляции с обработкой ошибок"""
    logger.info("Tick loop started")
    
    while True:
        try:
            interval = uniform(max(0.1, TICK_MIN_SEC), max(TICK_MIN_SEC, TICK_MAX_SEC))
            await asyncio.sleep(interval / max(world.speed, 0.1))

            result = world.step()
            await hub.broadcast({
                "type": "agents_state",
                "payload": world.agents_state_payload()
            })

            if result.event is not None:
                await hub.broadcast({"type": "event", "payload": result.event})
            
            if result.relations_changed:
                await hub.broadcast({
                    "type": "relations",
                    "payload": world.relations_payload()
                })
                
        except asyncio.CancelledError:
            logger.info("Tick loop cancelled")
            break
        except Exception as e:
            logger.error(f"Error in tick_loop: {e}", exc_info=True)
            await asyncio.sleep(1)  # Пауза перед следующей попыткой


@app.on_event("startup")
async def startup() -> None:
    """Запуск фоновых задач при старте приложения"""
    logger.info("Starting application...")
    app.state.tick_task = asyncio.create_task(tick_loop())
    logger.info("Application started successfully")


@app.on_event("shutdown")
async def shutdown() -> None:
    """Graceful shutdown - закрыть все соединения и задачи"""
    logger.info("Shutting down application...")
    
    # Закрыть все WebSocket соединения
    await hub.close_all()
    
    # Отменить tick task
    task = getattr(app.state, "tick_task", None)
    if task:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    
    logger.info("Application shutdown complete")


@app.get("/api/health")
async def health() -> dict:
    """Проверка здоровья сервера"""
    return {"status": "ok", "version": "0.1.0"}


@app.get("/api/state")
async def state() -> dict:
    """Получить полное состояние симуляции"""
    return world.state_payload()


@app.get("/api/agents")
async def agents() -> list[dict]:
    """Получить список всех агентов"""
    return world.agents_list_payload()


@app.get("/api/agents/{agent_id}")
async def agent(agent_id: str) -> dict:
    """Получить детальную информацию об агенте"""
    details = world.agent_details(agent_id)
    if not details:
        raise HTTPException(status_code=404, detail="agent not found")
    return details


@app.get("/api/relations")
async def relations() -> dict:
    """Получить граф отношений между агентами"""
    return world.relations_payload()


@app.get("/api/events")
async def events(
    limit: int = Query(default=200, ge=1, le=500),
    agent_id: str | None = Query(default=None),
) -> list[dict]:
    """Получить список событий с фильтрацией"""
    return world.events_payload(limit=limit, agent_id=agent_id)


@app.post("/api/control/event")
async def control_event(payload: ControlEventIn) -> dict:
    """Добавить событие от мира"""
    logger.info(f"World event received: {payload.text[:50]}...")
    
    event = world.add_world_event(payload.text, payload.importance)
    await hub.broadcast({"type": "event", "payload": event})
    
    logger.info(f"World event broadcasted with id: {event['id']}")
    return {"event_id": event["id"]}


@app.post("/api/control/message")
async def control_message(payload: ControlMessageIn) -> dict:
    """Отправить сообщение конкретному агенту"""
    logger.info(f"Message to agent {payload.agent_id}: {payload.text[:50]}...")
    
    try:
        event = world.add_agent_message(payload.agent_id, payload.text)
    except KeyError:
        logger.warning(f"Agent not found: {payload.agent_id}")
        raise HTTPException(status_code=404, detail="agent not found") from None
    
    await hub.broadcast({"type": "event", "payload": event})
    logger.info(f"Message broadcasted to agent {payload.agent_id}")
    return {"accepted": True}


@app.post("/api/control/speed")
async def control_speed(payload: ControlSpeedIn) -> dict:
    """Изменить скорость симуляции"""
    old_speed = world.speed
    speed = world.update_speed(payload.speed)
    logger.info(f"Speed changed: {old_speed:.1f}x -> {speed:.1f}x")
    return {"speed": speed}


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket) -> None:
    """WebSocket endpoint для real-time обновлений"""
    await hub.add(ws)
    
    try:
        # Отправить начальное состояние
        await hub.send(ws, {
            "type": "agents_state",
            "payload": world.agents_state_payload()
        })
        await hub.send(ws, {
            "type": "relations",
            "payload": world.relations_payload()
        })
        
        # Отправить последние 10 событий
        for event in world.events_payload(limit=10):
            await hub.send(ws, {"type": "event", "payload": event})

        # Держать соединение открытым и периодически проверять его
        while True:
            try:
                # Ждем сообщения от клиента с таймаутом для проверки соединения
                await asyncio.wait_for(ws.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Отправить ping для проверки соединения
                await ws.send_text(json.dumps({"type": "ping"}))
                
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        await hub.remove(ws)