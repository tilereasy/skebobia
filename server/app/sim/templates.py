from __future__ import annotations

import logging
from typing import Any


TEMPLATES: dict[str, list[str]] = {
    "respond_user": [
        "Принял сообщение. Сейчас проверю обстановку и вернусь с результатом.",
        "Понял. Сделаю короткую проверку по теме и дам обновление.",
        "Принял. Предлагаю один шаг: собираю факты и отвечаю по делу.",
        "Ок, действую: уточняю детали и сообщаю, что получилось.",
        "Сообщение получил. Беру задачу и отвечу после проверки.",
        "Принято. Сейчас сверю данные и отправлю конкретный вывод.",
    ],
    "respond_agent": [
        "{target_name}, принял. Подхожу к тебе и сверяю план на месте.",
        "{target_name}, понял. Предлагаю встретиться у центра и синхронизировать шаги.",
        "{target_name}, фиксирую. Проверю сектор рядом и сразу отпишу результат.",
        "{target_name}, принял задачу. Беру этот пункт и начинаю выполнение.",
        "{target_name}, вижу запрос. Сначала уточню факт, затем подтвержу план.",
        "{target_name}, понял. Предлагаю конкретный шаг и двигаюсь к точке встречи.",
    ],
    "support": [
        "Держимся вместе и действуем по шагам.",
        "Я рядом, берем один понятный шаг и продолжаем.",
        "Спокойно, собираем факты и двигаемся дальше.",
        "Без паники, фиксируем задачу и делаем ее по порядку.",
        "Поддерживаю: сначала проверка, потом решение.",
        "Остаемся в контакте и синхронизируем действия.",
    ],
    "conflict": [
        "Спорный момент, беру паузу и проверяю факты.",
        "Сейчас без резких решений, сначала сверим данные.",
        "Вижу напряжение, предлагаю короткий шаг для деэскалации.",
        "Стоп, сначала уточняю детали и потом предлагаю план.",
        "Не спешим, фиксирую риски и действую аккуратно.",
        "Остановимся на минуту и выберем безопасный ход.",
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
        "{target_name}, предлагаю встретиться у центральной точки и сверить план.",
        "{target_name}, проверь северный сектор и пришли краткий статус.",
        "{target_name}, беру свою часть, согласуй со мной следующий шаг.",
        "{target_name}, давай соберемся у арки через минуту и распределим задачи.",
        "{target_name}, вижу риск в зоне рынка, проверь и отпишись по факту.",
        "{target_name}, предлагаю план из одного шага: фиксируем обстановку и синхронизируемся.",
        "{target_name}, уточни состояние у входа, я подойду к тебе для координации.",
        "{target_name}, отмечаю событие рядом, проверь детали и подтвердим решение.",
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
