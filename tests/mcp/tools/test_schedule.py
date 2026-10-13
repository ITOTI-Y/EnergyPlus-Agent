from typing import Any

from idfpy.models import ScheduleCompact, ScheduleTypeLimits

from src.mcp.tools.schedule import ScheduleCompactTool
from src.state.config_state import ConfigState


def _tool() -> ScheduleCompactTool:
    state = ConfigState()
    state.idf.add(
        ScheduleTypeLimits(
            name="Fraction",
            lower_limit_value=0.0,
            upper_limit_value=1.0,
            numeric_type="Continuous",
        )
    )
    return ScheduleCompactTool(state)


def _payload(last_time: str = "24:00") -> dict[str, Any]:
    return {
        "Name": "Office",
        "Schedule Type Limits Name": "Fraction",
        "Data": [
            {
                "Through": "12/31",
                "Days": [
                    {
                        "For": "AllDays",
                        "Times": [
                            {"Until": {"Time": "08:00", "Value": 0.0}},
                            {"Until": {"Time": last_time, "Value": 1.0}},
                        ],
                    }
                ],
            }
        ],
    }


def _fields(tool: ScheduleCompactTool) -> list[str]:
    schedule = tool.state.idf.all_of_type(ScheduleCompact)["Office"]
    return [str(item.field) for item in schedule.data or []]


def test_create_flattens_nested_schedule():
    tool = _tool()

    response = tool.create(_payload())

    assert response.success, response.message
    assert _fields(tool) == [
        "Through: 12/31",
        "For: AllDays",
        "Until: 08:00",
        "0.0",
        "Until: 24:00",
        "1.0",
    ]


def test_create_rejects_day_not_ending_at_midnight():
    tool = _tool()

    response = tool.create(_payload(last_time="18:00"))

    assert not response.success
    assert "last time must be 24:00" in response.message
    assert not tool.state.idf.all_of_type(ScheduleCompact)


def test_update_without_data_keeps_existing_fields():
    tool = _tool()
    tool.create(_payload())
    before = _fields(tool)

    response = tool.update("Office", {"Schedule Type Limits Name": "Fraction"})

    assert response.success, response.message
    assert _fields(tool) == before
