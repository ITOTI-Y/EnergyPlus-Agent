import json

from idfpy.models import ScheduleCompact, Zone

from src.agent.tools.people_tools import make_people_tools
from src.state.config_state import ConfigState

PEOPLE_ARGS = {
    "zone_names": ["Zone_1", "Zone_2"],
    "number_of_people_schedule_name": "Occupancy",
    "activity_level_schedule_name": "Activity",
    "number_of_people": 10.0,
}


def _create_people_tool(config: ConfigState):
    tools = make_people_tools(config, {"Office": ["Zone_1", "Zone_2"]})
    return next(t for t in tools if t.name == "create_people")


def _with_schedules() -> ConfigState:
    config = ConfigState()
    config.idf.add(
        ScheduleCompact(name="Occupancy", schedule_type_limits_name="Fraction")
    )
    config.idf.add(
        ScheduleCompact(name="Activity", schedule_type_limits_name="Any Number")
    )
    return config


def test_create_people_fails_when_no_zone_gets_its_object():
    config = ConfigState()
    result = json.loads(_create_people_tool(config).invoke(PEOPLE_ARGS))

    assert not result["success"]
    [zone_1, zone_2] = result["data"]["failed"]
    assert zone_1.startswith("Zone_1:")
    assert all(name in zone_1 for name in ("Zone_1", "Occupancy", "Activity"))
    assert zone_2.startswith("Zone_2:")
    assert not config.idf.all_of_type("People")


def test_create_people_reports_only_the_zones_that_failed():
    config = _with_schedules()
    config.idf.add(Zone(name="Zone_1"))

    result = json.loads(_create_people_tool(config).invoke(PEOPLE_ARGS))

    assert result["success"]
    assert result["message"] == "Created People in 1 of 2 zones."
    [failed] = result["data"]["failed"]
    assert failed.startswith("Zone_2:")
    assert config.idf.has("People", "Zone_1_People")


def test_create_people_in_every_zone_names_them_after_the_zone():
    config = _with_schedules()
    config.idf.add(Zone(name="Zone_1"))
    config.idf.add(Zone(name="Zone_2"))

    result = json.loads(_create_people_tool(config).invoke(PEOPLE_ARGS))

    assert result == {
        "success": True,
        "message": "Created People in 2 of 2 zones.",
        "data": None,
    }
    assert set(config.idf.all_of_type("People")) == {"Zone_1_People", "Zone_2_People"}


def test_people_given_by_plan_go_in_every_zone_of_the_plan():
    config = _with_schedules()
    for zone in ("Zone_1", "Zone_2"):
        config.idf.add(Zone(name=zone))
    by_plan = {k: v for k, v in PEOPLE_ARGS.items() if k != "zone_names"}

    result = json.loads(
        _create_people_tool(config).invoke(by_plan | {"plans": ["Office"]})
    )

    assert result["success"]
    assert set(config.idf.all_of_type("People")) == {"Zone_1_People", "Zone_2_People"}


def test_people_without_plans_or_zones_are_refused():
    by_nothing = {k: v for k, v in PEOPLE_ARGS.items() if k != "zone_names"}

    result = _create_people_tool(_with_schedules()).invoke(by_nothing)

    assert "give the zones by `plans`, `zone_names` or both" in result
