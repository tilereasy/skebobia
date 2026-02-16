from __future__ import annotations

from typing import Any


TEMPLATES: dict[str, list[str]] = {
    "respond_user": [
        "Понял тебя. Все будет норм.",
        "Окей, принято. Сейчас сделаю.",
        "Слушаю. Действую по твоей просьбе.",
        "Принял. Проверю обстановку и вернусь с апдейтом.",
        "Есть контакт. Начинаю разбор ситуации.",
        "Окей, беру в работу. Отпишусь по результату.",
    ],
    "respond_agent": [
        "Принял, {target_name}. Держу в голове.",
        "Окей, {target_name}. Сейчас разберусь.",
        "Слышу тебя, {target_name}. Пойдем по шагам.",
        "{target_name}, вижу запрос. Давай синхронизируем план.",
        "{target_name}, понял сигнал. Дам обратную связь после проверки.",
        "{target_name}, беру на себя этот участок.",
    ],
    "support": [
        "Я рядом. Дыши, мы справимся.",
        "Держись, я помогу.",
        "Не ты один в этом, действуем вместе.",
        "Я с тобой. Разделим задачу и снизим риск.",
        "Давай спокойно: ты держишь линию, я подстрахую.",
        "Вижу напряжение. Действуем шаг за шагом, без паники.",
    ],
    "conflict": [
        "Не согласен. Ты реально дебил.",
        "Так мы сделаем только хуже.",
        "Стоп, это уже перебор.",
        "Это решение рискованное, предлагаю другой вариант.",
        "Сейчас мы теряем контроль. Нужен более спокойный план.",
        "Не поддерживаю этот ход, давай пересоберем приоритеты.",
    ],
    "explore": [
        "Наблюдаю.",
        "Похоже всё стабильно.",
        "Интересное место, проверю дальше.",
        "Сканирую сектор и отмечаю изменения.",
        "Проверяю периметр, пока критичного не вижу.",
        "Делаю круг наблюдения и собираю сигналы.",
    ],
    "panic": [
        "Мне тревожно.",
        "Слишком опасно, ухожу.",
        "Ситуация плохая, ищу укрытие.",
        "Риск растет, смещаюсь в безопасную точку.",
        "Слишком много угроз, нужна перегруппировка.",
        "Давление высокое, беру паузу и ухожу из зоны риска.",
    ],
    "memory": [
        "{name} вспоминает: {topic}.",
        "{name} держит в голове: {topic}.",
        "{name} обдумывает тему: {topic}.",
        "{name} фиксирует в памяти: {topic}.",
        "{name} возвращается мыслью к теме: {topic}.",
        "{name} связывает текущее с прошлым: {topic}.",
    ],
    "agent_message": [
        "{target_name}, проверь обстановку рядом с собой.",
        "{target_name}, синхронизируемся по плану.",
        "{target_name}, нужна обратная связь по текущей ситуации.",
        "{target_name}, дай короткий статус по теме: {topic}.",
        "{target_name}, что видишь у себя? сверим шаги.",
        "{target_name}, отпишись, если нужна помощь на месте.",
        "{target_name}, подтверждай: держим тот же вектор?",
        "{target_name}, сообщи, где узкое место прямо сейчас.",
    ],
}


def choose_template(kind: str, selector: int) -> str:
    options = TEMPLATES.get(kind, [])
    if not options:
        return "..."
    return options[abs(selector) % len(options)]


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
            return ""

    return template.format_map(_SafeFormatDict(payload))
