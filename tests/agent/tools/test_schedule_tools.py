import json

from idfpy.models.schedules import ScheduleCompact, ScheduleCompactDataItem

from src.agent.tools.schedule_tools import make_schedule_tools
from src.mcp.state import ConfigState


def _update_tool(config: ConfigState):
    return next(
        tool
        for tool in make_schedule_tools(config)
        if tool.name == "update_schedule_compact"
    )


def _config() -> ConfigState:
    config = ConfigState()
    config.idf.add(
        ScheduleCompact(
            name="Office_Schedule",
            schedule_type_limits_name="Fraction",
            data=[
                ScheduleCompactDataItem(field="Through: 12/31"),
                ScheduleCompactDataItem(field="For: AllDays"),
                ScheduleCompactDataItem(field="Until: 24:00, 1.0"),
            ],
        )
    )
    return config


def _valid_data(value: float) -> list[dict]:
    return [
        {
            "Through": "12/31",
            "Days": [
                {
                    "For": "AllDays",
                    "Times": [{"Until": {"Time": "24:00", "Value": value}}],
                }
            ],
        }
    ]


def test_update_schedule_rejects_type_limits_and_data_atomically() -> None:
    config = _config()
    obj = config.idf.get(ScheduleCompact, "Office_Schedule")
    assert obj is not None
    before = obj.model_dump()

    result = json.loads(
        _update_tool(config).invoke(
            {
                "name": "Office_Schedule",
                "schedule_type_limits_name": "",
                "data": _valid_data(0.5),
            }
        )
    )

    assert not result["success"]
    assert obj.model_dump() == before


def test_update_schedule_rejects_data_before_type_limits_mutation() -> None:
    config = _config()
    obj = config.idf.get(ScheduleCompact, "Office_Schedule")
    assert obj is not None
    before = obj.model_dump()
    invalid_data = [
        {
            "Through": "06/30",
            "Days": [
                {
                    "For": "AllDays",
                    "Times": [{"Until": {"Time": "24:00", "Value": 0.5}}],
                }
            ],
        }
    ]

    result = json.loads(
        _update_tool(config).invoke(
            {
                "name": "Office_Schedule",
                "schedule_type_limits_name": "Any Number",
                "data": invalid_data,
            }
        )
    )

    assert not result["success"]
    assert obj.model_dump() == before


def test_update_schedule_applies_valid_fields_together() -> None:
    config = _config()
    obj = config.idf.get(ScheduleCompact, "Office_Schedule")
    assert obj is not None

    result = json.loads(
        _update_tool(config).invoke(
            {
                "name": "Office_Schedule",
                "schedule_type_limits_name": "Any Number",
                "data": _valid_data(0.5),
            }
        )
    )

    assert result["success"]
    assert obj.schedule_type_limits_name == "Any Number"
    assert obj.data is not None
    assert [item.field for item in obj.data] == [
        "Through: 12/31",
        "For: AllDays",
        "Until: 24:00, 0.5",
    ]
