from __future__ import annotations

import logging
from typing import Any


TEMPLATES: dict[str, list[str]] = {
    "respond_user": [
        "Я услышал тебя, сейчас гляну внимательнее вокруг.",
        "Понял, я здесь, скажу прямо что вижу.",
        "Ок, присмотрюсь и отвечу без лишних слов.",
        "Хорошо, я проверю это на месте.",
        "Слышу тебя, мне самому интересно, что тут происходит.",
        "Ладно, сейчас разберусь и скажу по-человечески.",
    ],
    "respond_agent": [
        "{target_name}, я подойду ближе, мне не по себе от этого места.",
        "{target_name}, подожди у центра, хочу обсудить это лично.",
        "{target_name}, гляну рядом и сразу тебе скажу, что заметил.",
        "{target_name}, я рядом, давай посмотрим вместе.",
        "{target_name}, кажется, тут что-то странное, подойди на минуту.",
        "{target_name}, я двигаюсь к тебе, не хочу оставаться с этим один.",
    ],
    "support": [
        "Я рядом, ты не один.",
        "Давай просто подышим, всё ещё под контролем.",
        "Слышишь? Я тоже это заметил, держимся вместе.",
        "Не уходи далеко, мне спокойнее рядом с тобой.",
        "Мне тревожно, но вместе справимся.",
        "Я с тобой, давай смотреть по сторонам.",
    ],
    "conflict": [
        "Стоп, меня это злит, давай на секунду остынем.",
        "Не дави, я и так на взводе.",
        "Я слышу тебя, но сейчас слишком напряжено.",
        "Давай без крика, у меня уже гудит в голове.",
        "Мне не нравится этот тон, говори спокойнее.",
        "Секунду, мне нужно перевести дух.",
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
        "{target_name}, подойди к арке, я хочу показать одну странную вещь.",
        "{target_name}, я заметил движение у тропы, глянь туда тоже.",
        "{target_name}, мне тревожно, останься рядом на минуту.",
        "{target_name}, у входа что-то не так, подойди ко мне.",
        "{target_name}, я нашел следы у рынка, хочу твоего мнения.",
        "{target_name}, у лавки шум, мне нужен второй взгляд.",
        "{target_name}, я возле указателя, тут странная деталь.",
        "{target_name}, кажется, ветер перевернул вещи у дорожки, подойди.",
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
        "Я понял.",
        "Вижу.",
        "Слышу тебя.",
        "Хорошо.",
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
