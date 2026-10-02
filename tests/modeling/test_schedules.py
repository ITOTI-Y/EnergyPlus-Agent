import pytest
from pydantic import ValidationError

from src.modeling.schedules import ThroughSchema, schedule_fields


def _period(through: str, times: list[tuple[str, float]], day="AllDays") -> dict:
    return {
        "Through": through,
        "Days": [
            {
                "For": day,
                "Times": [{"Until": {"Time": t, "Value": v}} for t, v in times],
            }
        ],
    }


def test_fields_separate_until_time_and_value():
    periods = [
        ThroughSchema.model_validate(
            _period(
                "12/31", [("8:00", 0.0), ("18:00", 1.0), ("24:00", 0.0)], "Weekdays"
            )
        )
    ]

    assert schedule_fields(periods) == [
        "Through: 12/31",
        "For: Weekdays",
        "Until: 08:00",
        "0.0",
        "Until: 18:00",
        "1.0",
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
