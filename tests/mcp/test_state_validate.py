from idfpy.models.internal_gains import Lights, People
from idfpy.models.schedules import ScheduleCompact, ScheduleCompactDataItem
from idfpy.models.thermal_zones import Zone

from src.mcp.state import ConfigState


def test_validate_references_empty_state_returns_no_errors():
    assert ConfigState().validate_references() == []


def test_validate_references_reports_schedule_missing_type_limits():
    config = ConfigState()
    config.idf.add(
        ScheduleCompact(
            name="Office_Occupancy",
            schedule_type_limits_name="Missing_Limits",
            data=[
                ScheduleCompactDataItem(field="Through: 12/31"),
                ScheduleCompactDataItem(field="For: AllDays"),
                ScheduleCompactDataItem(field="Until: 24:00, 1.0"),
            ],
        )
    )

    errors = config.validate_references()

    assert any("Missing_Limits" in e for e in errors)


def test_validate_references_reports_people_comfort_schedules():
    config = ConfigState()
    config.idf.add(Zone(name="Z1"))
    config.idf.add(
        People(
            name="Z1_People",
            zone_or_zonelist_or_space_or_spacelist_name="Z1",
            number_of_people_calculation_method="People",
            number_of_people=2.0,
            number_of_people_schedule_name="Missing_Occupancy",
            activity_level_schedule_name="Missing_Activity",
            work_efficiency_schedule_name="Missing_Work_Efficiency",
            clothing_insulation_schedule_name="Missing_Clothing",
            air_velocity_schedule_name="Missing_Air_Velocity",
        )
    )

    errors = config.validate_references()

    for missing in (
        "Missing_Occupancy",
        "Missing_Activity",
        "Missing_Work_Efficiency",
        "Missing_Clothing",
        "Missing_Air_Velocity",
    ):
        assert any(missing in e for e in errors)


def test_validate_references_reports_lights_missing_zone():
    config = ConfigState()
    config.idf.add(
        Lights(
            name="Z1_Lights",
            zone_or_zonelist_or_space_or_spacelist_name="Missing_Zone",
            schedule_name="Missing_Schedule",
            design_level_calculation_method="Watts/Area",
            watts_per_floor_area=10.0,
        )
    )

    errors = config.validate_references()

    assert any("Missing_Zone" in e for e in errors)
    assert any("Missing_Schedule" in e for e in errors)
