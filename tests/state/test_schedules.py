from typing import Any

import pytest

from src.state.schedules import flatten_schedule_data


def _weekday_schedule(
    through: str = "12/31", last_time: str = "24:00"
) -> list[dict[str, Any]]:
    return [
        {
            "Through": through,
            "Days": [
                {
                    "For": "Weekdays",
                    "Times": [
                        {"Until": {"Time": "8:00", "Value": 0}},
                        {"Until": {"Time": last_time, "Value": 1}},
                    ],
                }
            ],
        }
    ]


def test_flatten_normalizes_times_and_splits_until_fields():
    assert flatten_schedule_data(_weekday_schedule()) == [
        "Through: 12/31",
        "For: Weekdays",
        "Until: 08:00",
        "0.0",
        "Until: 24:00",
        "1.0",
    ]


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ([], "at least one Through"),
        (_weekday_schedule(through="6/30"), "Through: 12/31"),
        (_weekday_schedule(last_time="18:00"), "last time must be 24:00"),
        (
            [{"Through": "12/31", "Days": [{"For": "Funday", "Times": []}]}],
            "invalid day type 'Funday'",
        ),
        ([{"Through": "12/31"}], r"Data\[0\] is missing 'Days'"),
    ],
)
def test_flatten_rejects_invalid_schedule(data, message):
    with pytest.raises(ValueError, match=message):
        flatten_schedule_data(data)
