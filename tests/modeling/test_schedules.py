import pytest
from pydantic import ValidationError

from src.modeling.schedules import ThroughSchema, schedule_fields


def _day(day: str, times: list[tuple[str, float]]) -> dict:
    return {"For": day, "Times": [{"Until": {"Time": t, "Value": v}} for t, v in times]}


def _period(through: str, times: list[tuple[str, float]], day="AllDays") -> dict:
    return {"Through": through, "Days": [_day(day, times)]}


OFF = [("24:00", 0.0)]


def test_fields_separate_until_time_and_value():
    office = [("8:00", 0.0), ("18:00", 1.0), ("24:00", 0.0)]
    period = {
        "Through": "12/31",
        "Days": [_day("Weekdays", office), _day("AllOtherDays", OFF)],
    }

    assert schedule_fields([ThroughSchema.model_validate(period)]) == [
        "Through: 12/31",
        "For: Weekdays",
        "Until: 08:00",
        "0.0",
        "Until: 18:00",
        "1.0",
        "Until: 24:00",
        "0.0",
        "For: AllOtherDays",
        "Until: 24:00",
        "0.0",
    ]


def test_day_must_end_at_midnight():
    with pytest.raises(ValidationError, match="24:00"):
        ThroughSchema.model_validate(_period("12/31", [("18:00", 1.0)]))


def test_last_period_must_reach_year_end():
    periods = [ThroughSchema.model_validate(_period("06/30", [("24:00", 1.0)]))]

    with pytest.raises(ValueError, match="12/31"):
        schedule_fields(periods)


def test_unknown_day_type_is_rejected():
    with pytest.raises(ValidationError, match="For"):
        ThroughSchema.model_validate(_period("12/31", [("24:00", 1.0)], "Workdays"))


def test_period_must_give_every_day_type_a_value():
    # EnergyPlus only warns, and runs weekends and design days without values.
    with pytest.raises(ValidationError, match="Sunday, Saturday, Holiday"):
        ThroughSchema.model_validate(_period("12/31", OFF, "Weekdays"))


def test_all_other_days_cannot_come_first():
    period = {
        "Through": "12/31",
        "Days": [_day("AllOtherDays", OFF), _day("Weekdays", OFF)],
    }

    with pytest.raises(ValidationError, match="only follow"):
        ThroughSchema.model_validate(period)


def test_day_types_listed_one_by_one_cover_the_period():
    listed = [
        "Weekdays",
        "Weekends",
        "Holidays",
        "SummerDesignDay",
        "WinterDesignDay",
        "CustomDay1",
        "CustomDay2",
    ]
    period = {"Through": "12/31", "Days": [_day(d, OFF) for d in listed]}

    ThroughSchema.model_validate(period)
