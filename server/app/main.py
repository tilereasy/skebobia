from __future__ import annotations

import asyncio
import contextlib
import json
import os

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.db.models import ControlEventIn, ControlMessageIn, ControlSpeedIn
from app.sim.engine import StubWorld


class WsHub:
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def add(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)

    async def remove(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def send(self, ws: WebSocket, message: dict) -> None:
        await ws.send_text(json.dumps(message, ensure_ascii=False))

    async def broadcast(self, message: dict) -> None:
        async with self._lock:
            clients = list(self._clients)
        if not clients:
            return

        serialized = json.dumps(message, ensure_ascii=False)
        stale: list[WebSocket] = []
        for ws in clients:
            try:
                await ws.send_text(serialized)
            except RuntimeError:
                stale.append(ws)
            except Exception:
                stale.append(ws)
        if stale:
            async with self._lock:
                for ws in stale:
                    self._clients.discard(ws)


TICK_INTERVAL_SEC = float(os.getenv("TICK_INTERVAL_SEC", "1.0"))
RELATIONS_INTERVAL_TICKS = int(os.getenv("RELATIONS_INTERVAL_TICKS", "5"))

app = FastAPI(title="Skebobia Day 0 Stub Server", version="0.1.0")
world = StubWorld(relations_interval_ticks=RELATIONS_INTERVAL_TICKS)
hub = WsHub()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def tick_loop() -> None:
    while True:
        await asyncio.sleep(max(0.1, TICK_INTERVAL_SEC) / max(world.speed, 0.1))

        result = world.step()
        await hub.broadcast({"type": "agents_state", "payload": world.agents_state_payload()})

        if result.event is not None:
            await hub.broadcast({"type": "event", "payload": result.event})
        if result.relations_changed:
            await hub.broadcast({"type": "relations", "payload": world.relations_payload()})


@app.on_event("startup")
async def startup() -> None:
    app.state.tick_task = asyncio.create_task(tick_loop())


@app.on_event("shutdown")
async def shutdown() -> None:
    task = getattr(app.state, "tick_task", None)
    if task:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/api/state")
async def state() -> dict:
    return world.state_payload()


@app.get("/api/agents")
async def agents() -> list[dict]:
    return world.agents_list_payload()


@app.get("/api/agents/{agent_id}")
async def agent(agent_id: str) -> dict:
    details = world.agent_details(agent_id)
    if not details:
        raise HTTPException(status_code=404, detail="agent not found")
    return details


@app.get("/api/relations")
async def relations() -> dict:
    return world.relations_payload()


@app.get("/api/events")
async def events(
    limit: int = Query(default=200, ge=1, le=500),
    agent_id: str | None = Query(default=None),
) -> list[dict]:
    return world.events_payload(limit=limit, agent_id=agent_id)


@app.post("/api/control/event")
async def control_event(payload: ControlEventIn) -> dict:
    event = world.add_world_event(payload.text, payload.importance)
    await hub.broadcast({"type": "event", "payload": event})
    await hub.broadcast({"type": "agents_state", "payload": world.agents_state_payload()})
    await hub.broadcast({"type": "relations", "payload": world.relations_payload()})
    return {"event_id": event["id"]}


@app.post("/api/control/message")
async def control_message(payload: ControlMessageIn) -> dict:
    try:
        event = world.add_agent_message(payload.agent_id, payload.text)
    except KeyError:
        raise HTTPException(status_code=404, detail="agent not found") from None
    await hub.broadcast({"type": "event", "payload": event})
    await hub.broadcast({"type": "agents_state", "payload": world.agents_state_payload()})
    await hub.broadcast({"type": "relations", "payload": world.relations_payload()})
    return {"accepted": True}


@app.post("/api/control/speed")
async def control_speed(payload: ControlSpeedIn) -> dict:
    speed = world.update_speed(payload.speed)
    return {"speed": speed}


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket) -> None:
    await hub.add(ws)
    try:
        await hub.send(ws, {"type": "agents_state", "payload": world.agents_state_payload()})
        await hub.send(ws, {"type": "relations", "payload": world.relations_payload()})
        for event in world.events_payload(limit=10):
            await hub.send(ws, {"type": "event", "payload": event})

        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await hub.remove(ws)
