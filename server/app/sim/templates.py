from __future__ import annotations

from typing import Any


TEMPLATES: dict[str, list[str]] = {
    "respond_user": [
        "Понял тебя. Все будет норм.",
        "Окей, принято. Сейчас сделаю.",
        "Слушаю. Действую по твоей просьбе.",
    ],
    "respond_agent": [
        "Принял, {target}. Держу в голове.",
        "Окей, {target}. Сейчас разберусь.",
        "Слышу тебя, {target}. Пойдем по шагам.",
    ],
    "support": [
        "Я рядом. Дыши, мы справимся.",
        "Держись, я помогу.",
        "Не ты один в этом, действуем вместе.",
    ],
    "conflict": [
        "Не согласен. Ты реально дебил.",
        "Так мы сделаем только хуже.",
        "Стоп, это уже перебор.",
    ],
    "explore": [
        "Наблюдаю.",
        "Похоже всё стабильно.",
        "Интересное место, проверю дальше.",
    ],
    "panic": [
        "Мне тревожно.",
        "Слишком опасно, ухожу.",
        "Ситуация плохая, ищу укрытие.",
    ],
    "memory": [
        "{name} вспоминает: {topic}.",
        "{name} держит в голове: {topic}.",
        "{name} обдумывает тему: {topic}.",
    ],
    "agent_message": [
        "{target}, проверь обстановку рядом с собой.",
        "{target}, синхронизируемся по плану.",
        "{target}, нужна обратная связь по ситуации.",
    ],
}


def choose_template(kind: str, selector: int) -> str:
    options = TEMPLATES.get(kind, [])
    if not options:
        return "..."
    return options[abs(selector) % len(options)]


def render(kind: str, selector: int, **kwargs: Any) -> str:
    template = choose_template(kind, selector)
    return template.format(**kwargs)
