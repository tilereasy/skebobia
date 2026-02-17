from __future__ import annotations

import math

from app.agents.agent import Vec3


MAP_MIN = -20.0
MAP_MAX = 20.0
SAFE_POINT = Vec3(0.0, 0.0, 0.0)


def clamp_position(pos: Vec3) -> Vec3:
    return Vec3(
        x=max(MAP_MIN, min(MAP_MAX, pos.x)),
        y=0.0,
        z=max(MAP_MIN, min(MAP_MAX, pos.z)),
    )


def distance_2d(a: Vec3, b: Vec3) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.z - b.z) ** 2)


def step_towards(current: Vec3, target: Vec3, max_step: float) -> Vec3:
    dx = target.x - current.x
    dz = target.z - current.z
    distance = math.sqrt(dx * dx + dz * dz)
    if distance < 1e-6:
        return Vec3(current.x, 0.0, current.z)

    ratio = min(1.0, max_step / distance)
    return clamp_position(Vec3(current.x + dx * ratio, 0.0, current.z + dz * ratio))


def step_away_from(current: Vec3, threat: Vec3, max_step: float) -> Vec3:
    dx = current.x - threat.x
    dz = current.z - threat.z
    distance = math.sqrt(dx * dx + dz * dz)
    if distance < 1e-6:
        dx, dz = 1.0, 0.0
        distance = 1.0

    ratio = min(1.0, max_step / distance)
    return clamp_position(Vec3(current.x + dx * ratio, 0.0, current.z + dz * ratio))


def pick_wander_target(agent_id: str, tick: int) -> Vec3:
    seed = sum(ord(ch) for ch in agent_id) + tick * 17
    x = ((seed * 31) % 3200) / 3200.0 * 30.0 - 15.0
    z = ((seed * 53) % 3200) / 3200.0 * 30.0 - 15.0
    return clamp_position(Vec3(x, 0.0, z))
