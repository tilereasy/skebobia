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
        "{target_name}, у меня внутри неприятный холод от этого места.",
        "{target_name}, я зацепился взглядом за странную деталь у тропы.",
        "{target_name}, я услышал резкий шорох и до сих пор напряжен.",
        "{target_name}, мне тревожно, когда тут так внезапно стихает.",
        "{target_name}, я заметил след у дорожки и не могу это забыть.",
        "{target_name}, я пытаюсь понять, что здесь именно не так.",
    ],
    "support": [
        "Я рядом, ты не один.",
        "Я дышу ровнее, когда слышу твой голос.",
        "Я тоже это заметил, и у меня по коже мурашки.",
        "Мне тревожно, но я не отворачиваюсь от этого.",
        "Я с тобой и внимательно смотрю вокруг.",
        "Мне страшновато, но я держусь.",
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
        "{target_name}, у арки сейчас странная тишина, будто звук провалился.",
        "{target_name}, у тропы мелькнула тень, и меня это зацепило.",
        "{target_name}, у входа хлопнула створка, хотя ветра почти нет.",
        "{target_name}, у рынка появился мокрый след и быстро исчез.",
        "{target_name}, у лавки снова короткий металлический скрежет.",
        "{target_name}, у указателя дрожит табличка, будто кто-то стукнул по ней.",
        "{target_name}, у дорожки ветер перевернул бумагу с непонятными знаками.",
        "{target_name}, у камней слышен тихий свист, и я не понимаю источник.",
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
