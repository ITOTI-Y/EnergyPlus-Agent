from src.agent.state import merge_config_state
from src.mcp.state import ConfigState
from src.validator import ScheduleCollectionSchema, ZoneSchema


def test_merge_named_list_union():
    a = ConfigState.model_validate(
        {"zones": [ZoneSchema.model_validate({"Name": "A"})]}
    )
    b = ConfigState.model_validate(
        {"zones": [ZoneSchema.model_validate({"Name": "B"})]}
    )
    merged = merge_config_state(a, b)
    assert {z.name for z in merged.zones} == {"A", "B"}


def test_merge_new_wins_on_conflict():
    a = ConfigState.model_validate(
        {"zones": [ZoneSchema.model_validate({"Name": "A", "X Origin": 1.0})]}
    )
    b = ConfigState.model_validate(
        {"zones": [ZoneSchema.model_validate({"Name": "A", "X Origin": 9.0})]}
    )
    merged = merge_config_state(a, b)
    assert merged.zones[0].x_origin == 9.0


_SCHEDULE_DATA = [
    {
        "Through": "12/31",
        "Days": [
            {
                "For": "AllDays",
                "Times": [{"Until": {"Time": "24:00", "Value": 1.0}}],
            }
        ],
    }
]


def test_merge_schedules_nested():
    a = ConfigState.model_validate(
        {
            "schedules": ScheduleCollectionSchema.model_validate(
                {
                    "schedules": [
                        {
                            "Name": "S1",
                            "Schedule Type Limits Name": "F",
                            "Data": _SCHEDULE_DATA,
                        }
                    ]
                }
            )
        }
    )
    b = ConfigState.model_validate(
        {
            "schedules": ScheduleCollectionSchema.model_validate(
                {
                    "schedules": [
                        {
                            "Name": "S2",
                            "Schedule Type Limits Name": "F",
                            "Data": _SCHEDULE_DATA,
                        }
                    ]
                }
            )
        }
    )
    merged = merge_config_state(a, b)
    assert {s.name for s in merged.schedules.schedules} == {"S1", "S2"}
