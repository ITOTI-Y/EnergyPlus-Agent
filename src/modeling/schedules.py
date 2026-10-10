"""Nested Schedule:Compact input, checked for syntax and flattened into fields.

EnergyPlus reports compact-schedule syntax errors poorly: a day that does not
reach 24:00 yields hundreds of identical Severe lines, and a schedule whose
last period stops before 12/31 crashes the run. Checking the structure here
gives the caller one precise message instead.
"""

import re
from datetime import date
from typing import Literal

from idfpy.models.schedules import ScheduleCompact, ScheduleCompactDataItem
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

type DayType = Literal[
    "Weekdays",
    "Weekends",
    "Holidays",
    "AllDays",
    "SummerDesignDay",
    "WinterDesignDay",
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "CustomDay1",
    "CustomDay2",
    "AllOtherDays",
]

_TIME = re.compile(r"(\d{1,2}):(\d{2})")


class _ScheduleInputSchema(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class UntilSchema(_ScheduleInputSchema):
    time: str = Field(
        alias="Time", description="HH:MM; the last entry of a day is 24:00"
    )
    value: float = Field(alias="Value")

    @field_validator("time")
    @classmethod
    def _normalize_time(cls, value: str) -> str:
        match = _TIME.fullmatch(value.strip())
        if match is None:
            raise ValueError(f"time {value!r} is not HH:MM")
        hours, minutes = int(match[1]), int(match[2])
        if not ((hours < 24 and minutes < 60) or (hours, minutes) == (24, 0)):
            raise ValueError(f"time {value!r} is outside 00:00-24:00")
        return f"{hours:02d}:{minutes:02d}"


class TimeValueSchema(_ScheduleInputSchema):
    until: UntilSchema = Field(alias="Until")


class DayScheduleSchema(_ScheduleInputSchema):
    day_type: DayType = Field(alias="For")
    times: list[TimeValueSchema] = Field(alias="Times", min_length=1)

    @model_validator(mode="after")
    def _times_cover_the_day(self) -> "DayScheduleSchema":
        times = [t.until.time for t in self.times]
        if times != sorted(set(times)):
            raise ValueError(f"For {self.day_type}: times must increase, got {times}")
        if times[-1] != "24:00":
            raise ValueError(f"For {self.day_type}: the last time must be 24:00")
        return self


class ThroughSchema(_ScheduleInputSchema):
    through: str = Field(alias="Through", description="MM/DD, inclusive end date")
    days: list[DayScheduleSchema] = Field(alias="Days", min_length=1)

    @field_validator("through")
    @classmethod
    def _normalize_date(cls, value: str) -> str:
        month, _, day = value.strip().partition("/")
        try:
            # 2000 is a leap year, so 02/29 is accepted as EnergyPlus does.
            parsed = date(2000, int(month), int(day))
        except ValueError as e:
            raise ValueError(f"date {value!r} is not a valid MM/DD") from e
        return parsed.strftime("%m/%d")


def schedule_fields(periods: list[ThroughSchema]) -> list[str]:
    """Flatten periods into Schedule:Compact fields.

    ``Until`` time and value are separate fields: epJSON stores a combined
    ``Until: 24:00, 1.0`` as one value, which EnergyPlus cannot read.

    Raises:
        ValueError: If dates do not increase or the last one is not 12/31.
    """
    dates = [p.through for p in periods]
    if not dates or dates != sorted(set(dates)) or dates[-1] != "12/31":
        raise ValueError(f"Through dates must increase and end at 12/31, got {dates}")
    fields: list[str] = []
    for period in periods:
        fields.append(f"Through: {period.through}")
        for day in period.days:
            fields.append(f"For: {day.day_type}")
            for entry in day.times:
                fields.extend((f"Until: {entry.until.time}", str(entry.until.value)))
    return fields


def compact_schedule(
    name: str, schedule_type_limits_name: str, periods: list[ThroughSchema]
) -> ScheduleCompact:
    return ScheduleCompact(
        name=name,
        schedule_type_limits_name=schedule_type_limits_name,
        data=[ScheduleCompactDataItem(field=f) for f in schedule_fields(periods)],
    )
