from __future__ import annotations

import logging
from typing import Any


TEMPLATES: dict[str, list[str]] = {
    "respond_user": [
        "Ага.",
        "Угу.",
        "Хм, ясно.",
        "Мм, понял.",
        "Окей.",
        "Ну да.",
    ],
    "respond_agent": [
        "{target_name}, ага.",
        "{target_name}, угу.",
        "{target_name}, мм.",
        "{target_name}, окей.",
        "{target_name}, ясно.",
        "{target_name}, понял.",
    ],
    "support": [
        "Мм.",
        "Угу.",
        "Ага.",
        "Ну да, спокойно.",
        "Тихо, тихо.",
        "Дышим.",
    ],
    "conflict": [
        "Хм.",
        "Мм, спорно.",
        "Ну... не уверен.",
        "Эм, пауза.",
        "Ладно, без резких шагов.",
        "Окей, подождем.",
    ],
    "explore": [
        "Хм.",
        "Угу.",
        "Ага.",
        "Мм, смотрю.",
        "Ну, двигаюсь.",
        "Тихо.",
    ],
    "panic": [
        "Ой.",
        "Эм.",
        "Ух.",
        "Мм, тревожно.",
        "Пауза.",
        "Отойду на шаг.",
    ],
    "memory": [
        "{name}: мм... {topic}.",
        "{name}: угу, {topic}.",
        "{name}: ага, {topic}.",
        "{name}: хм, {topic}.",
        "{name}: ну да, {topic}.",
        "{name}: ясно, {topic}.",
    ],
    "agent_message": [
        "{target_name}, угу?",
        "{target_name}, ага?",
        "{target_name}, коротко?",
        "{target_name}, на связи?",
        "{target_name}, мм, статус?",
        "{target_name}, окей?",
        "{target_name}, все ровно?",
        "{target_name}, если что, дай знак.",
    ],
    "voice_empathic": [
        "Я рядом.",
        "Спокойно, держимся вместе.",
        "Давай без паники.",
        "Слышу тебя.",
    ],
    "voice_aggressive": [
        "Без лишних слов.",
        "Действуем быстро.",
        "По делу.",
        "Хватит тянуть.",
    ],
    "voice_cool": [
        "Принял.",
        "Отмечено.",
        "Фиксирую.",
        "Понял сигнал.",
    ],
}

LOGGER = logging.getLogger("app.sim.templates")


def _mix_selector(selector: int) -> int:
    # Deterministic integer mixer to avoid obvious modulo cycles on sequential ticks.
    value = abs(int(selector)) & 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x45D9F3B) & 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x45D9F3B) & 0xFFFFFFFF
    value ^= value >> 16
    return value


def choose_template(kind: str, selector: int) -> str:
    options = TEMPLATES.get(kind, [])
    if not options:
        return "..."
    return options[_mix_selector(selector) % len(options)]


def render(kind: str, selector: int, **kwargs: Any) -> str:
    template = choose_template(kind, selector)
    payload = dict(kwargs)
    if "target_name" not in payload and "target" in payload:
        payload["target_name"] = payload["target"]
    if "target" not in payload and "target_name" in payload:
        payload["target"] = payload["target_name"]
    payload.setdefault("topic", "текущей ситуации")
    payload.setdefault("name", "Агент")

    class _SafeFormatDict(dict):
        def __missing__(self, key: str) -> str:  # pragma: no cover - defensive fallback
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.debug(
                    "Template missing key kind=%s key=%s payload_keys=%s",
                    kind,
                    key,
                    sorted(payload.keys()),
                )
            return ""

    return template.format_map(_SafeFormatDict(payload))
