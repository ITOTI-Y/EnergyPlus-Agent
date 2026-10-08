"""Reference objects taken from one prototype model.

Materials, constructions and schedules come out as plain data that the agent
tools accept as they are: a construction carries its layer materials, and a
schedule is converted to the nested Through/For/Until periods of
``ThroughSchema``, whichever schedule objects the model uses.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Final, Literal

from idfpy import IDF, IDFBaseModel
from idfpy.models.constructions import Construction
from idfpy.models.schedules import (
    ScheduleCompact,
    ScheduleDayHourly,
    ScheduleTypeLimits,
    ScheduleWeekCompact,
    ScheduleYear,
)
from pydantic import ValidationError

from src.modeling.envelope import MATERIAL_TYPES, layer_names
from src.modeling.schedules import ThroughSchema

type Kind = Literal["material", "construction", "schedule"]


@dataclass(slots=True)
class ReferenceObject:
    """One object as the agent would recreate it, and where it is used."""

    kind: Kind
    object_type: str
    name: str
    data: dict[str, Any]
    roles: set[str] = field(default_factory=set)


@dataclass(slots=True)
class Extraction:
    objects: list[ReferenceObject]
    skipped: list[str]
    """Schedules that cannot be expressed as compact periods, with the reason."""


def _fields(obj: IDFBaseModel) -> dict[str, Any]:
    return obj.model_dump(exclude_none=True)


def _materials(idf: IDF) -> dict[str, ReferenceObject]:
    found = {}
    for material_type in MATERIAL_TYPES:
        for material in idf.all_of_type(material_type).values():
            found[material.name.lower()] = ReferenceObject(
                "material",
                material.idf_object_type(),
                material.name,
                _fields(material),
            )
    return found


def _constructions(
    idf: IDF, materials: dict[str, ReferenceObject], roles: dict[str, set[str]]
) -> Iterator[ReferenceObject]:
    for construction in idf.all_of_type(Construction).values():
        used = roles.get(construction.name.lower())
        if not used:
            continue  # defined but on no surface, e.g. for spaces the model lacks
        layers = layer_names(construction)
        if any(layer.lower() not in materials for layer in layers):
            continue  # e.g. a layer defined by another construction object type
        layer_data = [
            {"object_type": m.object_type, **m.data}
            for m in (materials[layer.lower()] for layer in layers)
        ]
        for layer in layers:
            materials[layer.lower()].roles |= used
        yield ReferenceObject(
            "construction",
            "Construction",
            construction.name,
            {"name": construction.name, "layers": layers, "materials": layer_data},
            used,
        )


_DAY_TYPES: Final = {
    name.lower(): name
    for name in (
        "Weekdays", "Weekends", "Holidays", "AllDays", "SummerDesignDay",
        "WinterDesignDay", "Sunday", "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "CustomDay1", "CustomDay2",
        "AllOtherDays",
    )
} | {"holiday": "Holidays", "weekday": "Weekdays", "weekend": "Weekends"} | {
    f"{day}s": day.capitalize()
    for day in ("sunday", "monday", "tuesday", "wednesday", "thursday", "friday",
                "saturday")
}  # fmt: skip
_KEYWORD: Final = re.compile(r"\s*(through|for|until|interpolate)\s*:?\s*(.*)", re.I)


def _day_types(text: str) -> list[str]:
    """Separate day-type blocks for 'Weekdays SummerDesignDay' and the like.

    Raises:
        ValueError: On an unknown day type.
    """
    names = text.split()
    unknown = [n for n in names if n.lower() not in _DAY_TYPES]
    if unknown or not names:
        raise ValueError(f"unknown day type in {text!r}")
    return [_DAY_TYPES[n.lower()] for n in names]


def compact_periods(fields: list[str]) -> list[dict[str, Any]]:
    """Schedule:Compact fields as nested Through/For/Until periods.

    Raises:
        ValueError: On interpolation, unknown day types or malformed fields.
    """
    periods: list[dict[str, Any]] = []
    days: list[dict[str, Any]] = []
    times: list[dict[str, Any]] | None = None
    pending: str | None = None
    for raw in fields:
        match = _KEYWORD.fullmatch(raw)
        if match is None:
            if pending is None or times is None:
                raise ValueError(f"value {raw!r} without an Until time")
            times.append({"Until": {"Time": pending, "Value": float(raw)}})
            pending = None
            continue
        keyword, rest = match[1].lower(), match[2].strip()
        if keyword == "through":
            days = []
            periods.append({"Through": rest, "Days": days})
        elif keyword == "for":
            if not periods:
                raise ValueError("For before Through")
            times = []
            # One shared list: blocks named together get the same values.
            days.extend({"For": d, "Times": times} for d in _day_types(rest))
        elif keyword == "interpolate":
            if rest.lower() != "no":
                raise ValueError(f"interpolation {rest!r} is not supported")
        else:
            time, _, value = rest.partition(",")
            if value.strip():
                if times is None:
                    raise ValueError("Until before For")
                times.append({"Until": {"Time": time.strip(), "Value": float(value)}})
            else:
                pending = time.strip()
    return periods


def _hourly(day: ScheduleDayHourly) -> list[dict[str, Any]]:
    values = [getattr(day, f"hour_{h}") for h in range(1, 25)]
    times = []
    for hour, value in enumerate(values, start=1):
        if hour < 24 and values[hour] == value:
            continue  # merge equal consecutive hours
        times.append({"Until": {"Time": f"{hour:02d}:00", "Value": float(value)}})
    return times


def year_periods(idf: IDF, year: ScheduleYear) -> list[dict[str, Any]]:
    """Schedule:Year over Week:Compact and Day:Hourly as compact periods.

    Raises:
        ValueError: If a week or day schedule is of another type or missing.
    """
    periods = []
    for item in year.schedule_weeks or []:
        week = idf.get(ScheduleWeekCompact, item.schedule_week_name)
        if week is None:
            raise ValueError(f"week {item.schedule_week_name!r} is not compact")
        days = []
        for entry in week.data or []:
            day = idf.get(ScheduleDayHourly, entry.schedule_day_name)
            if day is None:
                raise ValueError(f"day {entry.schedule_day_name!r} is not hourly")
            text = re.sub(r"^\s*for\s*:?", "", entry.daytype_list, flags=re.I)
            days += [{"For": d, "Times": _hourly(day)} for d in _day_types(text)]
        through = f"{item.end_month:02d}/{item.end_day:02d}"
        periods.append({"Through": through, "Days": days})
    return periods


def _limits(idf: IDF, name: str | None) -> dict[str, Any] | None:
    limits = idf.get(ScheduleTypeLimits, name or "")
    return None if limits is None else _fields(limits)


def _periods(
    idf: IDF, schedule: ScheduleCompact | ScheduleYear
) -> list[dict[str, Any]]:
    match schedule:
        case ScheduleCompact():
            return compact_periods(
                [str(d.field) for d in schedule.data or [] if d.field is not None]
            )
        case ScheduleYear():
            return year_periods(idf, schedule)


def _schedules(idf: IDF, skipped: list[str]) -> Iterator[ReferenceObject]:
    schedules = [
        *idf.all_of_type(ScheduleCompact).values(),
        *idf.all_of_type(ScheduleYear).values(),
    ]
    for schedule in schedules:
        try:
            periods = _periods(idf, schedule)
            # The schedule tool rejects what ThroughSchema rejects.
            for period in periods:
                ThroughSchema.model_validate(period)
        except (ValueError, ValidationError) as e:
            skipped.append(f"{schedule.name}: {str(e).splitlines()[0]}")
            continue
        yield ReferenceObject(
            "schedule",
            "Schedule:Compact",
            schedule.name,
            {
                "name": schedule.name,
                "schedule_type_limits": _limits(
                    idf, schedule.schedule_type_limits_name
                ),
                "periods": periods,
            },
        )


def extract_objects(idf: IDF, roles: dict[str, set[str]]) -> Extraction:
    """Materials, constructions and schedules of one model.

    Constructions on no surface, and materials only in those, are left out:
    prototypes define envelopes for spaces some building types lack.

    Args:
        idf: The model.
        roles: Where each construction is used, by lower-case name.
    """
    materials = _materials(idf)
    constructions = list(_constructions(idf, materials, roles))
    used = [m for m in materials.values() if m.roles]
    skipped: list[str] = []
    schedules = list(_schedules(idf, skipped))
    return Extraction([*used, *constructions, *schedules], skipped)
