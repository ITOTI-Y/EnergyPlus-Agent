from typing import Any, Final

from dateutil.parser import parse

_DAY_TYPES: Final = frozenset(
    {
        "weekdays",
        "weekends",
        "holidays",
        "alldays",
        "summerdesignday",
        "winterdesignday",
        "sunday",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "customday1",
        "customday2",
        "allotherdays",
    }
)


def _require(item: Any, key: str, where: str) -> Any:
    if not isinstance(item, dict) or key not in item:
        raise ValueError(f"{where} is missing {key!r}")
    return item[key]


def _flatten_day(day: dict[str, Any], where: str) -> list[str]:
    day_type = _require(day, "For", where)
    if day_type.lower() not in _DAY_TYPES:
        raise ValueError(f"{where}: invalid day type {day_type!r}")
    fields: list[str] = [f"For: {day_type}"]
    times = _require(day, "Times", where)
    for k, row in enumerate(times):
        row_where = f"{where}.Times[{k}]"
        until = _require(row, "Until", row_where)
        time = _require(until, "Time", row_where)
        value = float(_require(until, "Value", row_where))
        if k == len(times) - 1:
            if time != "24:00":
                raise ValueError(f"{row_where}: last time must be 24:00, got {time!r}")
        else:
            time = parse(time).strftime("%H:%M")
        fields.extend((f"Until: {time}", str(value)))
    return fields


def flatten_schedule_data(data: list[dict[str, Any]]) -> list[str]:
    if not data:
        raise ValueError("Schedule data needs at least one Through block")
    fields: list[str] = []
    for i, block in enumerate(data):
        where = f"Data[{i}]"
        date = parse(_require(block, "Through", where)).strftime("%m/%d")
        if i == len(data) - 1 and date != "12/31":
            raise ValueError("Schedule data must end with Through: 12/31")
        fields.append(f"Through: {date}")
        for j, day in enumerate(_require(block, "Days", where)):
            fields.extend(_flatten_day(day, f"{where}.Days[{j}]"))
    return fields
