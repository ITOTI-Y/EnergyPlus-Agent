import json

from idfpy.models.schedules import ScheduleCompact
from idfpy.models.thermal_zones import Zone

from src.agent.tools.people_tools import make_people_tools
from src.mcp.state import ConfigState

PEOPLE_ARGS = {
    "name": "Office_People",
    "zone_name": "Zone_1",
    "number_of_people_schedule_name": "Occupancy",
    "activity_level_schedule_name": "Activity",
    "number_of_people": 10.0,
}


def _create_people_tool(config: ConfigState):
    return next(t for t in make_people_tools(config) if t.name == "create_people")


def test_create_people_fails_on_missing_references():
    config = ConfigState()
    result = json.loads(_create_people_tool(config).invoke(PEOPLE_ARGS))
    assert not result["success"]
    missing = result["data"]["missing_references"]
    assert len(missing) == 3
    assert any("Zone_1" in m for m in missing)
    assert any("Occupancy" in m for m in missing)
    assert any("Activity" in m for m in missing)
    assert not config.idf.has("People", "Office_People")


def test_create_people_succeeds_with_existing_references():
    config = ConfigState()
    config.idf.add(Zone(name="Zone_1"))
    config.idf.add(
        ScheduleCompact(name="Occupancy", schedule_type_limits_name="Fraction")
    )
    config.idf.add(
        ScheduleCompact(name="Activity", schedule_type_limits_name="Any Number")
    )
    result = json.loads(_create_people_tool(config).invoke(PEOPLE_ARGS))
    assert result["success"]
    assert config.idf.has("People", "Office_People")
