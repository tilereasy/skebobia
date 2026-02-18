from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.db.models import (
    ControlAgentAddIn,
    ControlAgentRemoveIn,
    ControlEventIn,
    ControlMessageIn,
    ControlSpeedIn,
)
from app.sim.engine import StubWorld


def _load_env_from_repo_root() -> None:
    # server/app/main.py -> repo root is 2 levels up from "server"
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue

        value = value.strip()
        if value and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


_load_env_from_repo_root()


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

app = FastAPI(title="Skebobia LLM-first Server", version="0.2.0")
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

        started_at = time.perf_counter()
        before_event_id = world.world_state.next_event_id
        result = world.step()
        await hub.broadcast({"type": "agents_state", "payload": world.agents_state_payload()})

        for event in world.events_since(before_event_id):
            await hub.broadcast({"type": "event", "payload": event})
        if result.relations_changed:
            await hub.broadcast({"type": "relations", "payload": world.relations_payload()})

        tick_ms = (time.perf_counter() - started_at) * 1000.0
        avg = getattr(app.state, "avg_tick_ms", 0.0)
        if avg <= 0.0:
            app.state.avg_tick_ms = tick_ms
        else:
            app.state.avg_tick_ms = (avg * 0.88) + (tick_ms * 0.12)
        app.state.last_tick_ms = tick_ms


@app.on_event("startup")
async def startup() -> None:
    app.state.last_tick_ms = 0.0
    app.state.avg_tick_ms = 0.0
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
    payload = world.state_payload()
    payload["runtime"] = {
        "last_tick_ms": round(float(getattr(app.state, "last_tick_ms", 0.0)), 3),
        "avg_tick_ms": round(float(getattr(app.state, "avg_tick_ms", 0.0)), 3),
    }
    return payload


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
    before_event_id = world.world_state.next_event_id
    event, reactions = world.add_world_event(payload.text, payload.importance)
    for emitted_event in world.events_since(before_event_id):
        await hub.broadcast({"type": "event", "payload": emitted_event})
    await hub.broadcast({"type": "agents_state", "payload": world.agents_state_payload()})
    await hub.broadcast({"type": "relations", "payload": world.relations_payload()})
    return {"event_id": event["id"], "reaction_event_ids": [reaction["id"] for reaction in reactions]}


@app.post("/api/control/message")
async def control_message(payload: ControlMessageIn) -> dict:
    before_event_id = world.world_state.next_event_id
    try:
        event, reply = world.add_agent_message(payload.agent_id, payload.text)
    except KeyError:
        raise HTTPException(status_code=404, detail="agent not found") from None
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    for emitted_event in world.events_since(before_event_id):
        await hub.broadcast({"type": "event", "payload": emitted_event})
    await hub.broadcast({"type": "agents_state", "payload": world.agents_state_payload()})
    await hub.broadcast({"type": "relations", "payload": world.relations_payload()})
    return {
        "accepted": True,
        "event_id": event["id"],
        "reply_event_id": reply["id"] if reply is not None else None,
        "reply_pending": reply is None,
    }


@app.post("/api/control/speed")
async def control_speed(payload: ControlSpeedIn) -> dict:
    speed = world.update_speed(payload.speed)
    return {"speed": speed}


@app.post("/api/control/agent/add")
async def control_agent_add(payload: ControlAgentAddIn) -> dict:
    before_event_id = world.world_state.next_event_id
    try:
        added = world.add_agent(
            agent_id=payload.id,
            name=payload.name,
            traits=payload.traits,
            mood=payload.mood,
            avatar=payload.avatar,
            pos_x=payload.pos_x,
            pos_z=payload.pos_z,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    for emitted_event in world.events_since(before_event_id):
        await hub.broadcast({"type": "event", "payload": emitted_event})
    await hub.broadcast({"type": "agents_state", "payload": world.agents_state_payload()})
    await hub.broadcast({"type": "relations", "payload": world.relations_payload()})
    return {"accepted": True, "agent": added.to_agent_summary()}


@app.post("/api/control/agent/remove")
async def control_agent_remove(payload: ControlAgentRemoveIn) -> dict:
    before_event_id = world.world_state.next_event_id
    try:
        removed = world.remove_agent(payload.agent_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except KeyError:
        raise HTTPException(status_code=404, detail="agent not found") from None

    for emitted_event in world.events_since(before_event_id):
        await hub.broadcast({"type": "event", "payload": emitted_event})
    await hub.broadcast({"type": "agents_state", "payload": world.agents_state_payload()})
    await hub.broadcast({"type": "relations", "payload": world.relations_payload()})
    return {"accepted": True, "agent_id": removed.id}


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
