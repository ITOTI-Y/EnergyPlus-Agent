import pytest
from idfpy.models.internal_gains import Lights, People
from idfpy.models.schedules import ScheduleCompact, ScheduleCompactDataItem
from idfpy.models.simulation import Version
from idfpy.models.thermal_zones import FenestrationSurfaceDetailed, Zone

from src.state.config_state import ConfigState


def _window() -> FenestrationSurfaceDetailed:
    return FenestrationSurfaceDetailed(
        name="South_Window",
        surface_type="Window",
        construction_name="Window_Construction",
        building_surface_name="South_Wall",
        number_of_vertices=4,
        vertex_1_x_coordinate=1.0,
        vertex_1_y_coordinate=0.0,
        vertex_1_z_coordinate=2.5,
        vertex_2_x_coordinate=1.0,
        vertex_2_y_coordinate=0.0,
        vertex_2_z_coordinate=1.0,
        vertex_3_x_coordinate=4.0,
        vertex_3_y_coordinate=0.0,
        vertex_3_z_coordinate=1.0,
        vertex_4_x_coordinate=4.0,
        vertex_4_y_coordinate=0.0,
        vertex_4_z_coordinate=2.5,
    )


def _dump(state: ConfigState) -> set[str]:
    # Full dumps, since an IDF file spells out fields left at their default;
    # a set, since files store objects sorted by type.
    return {repr(obj) for obj in state.idf}


@pytest.mark.parametrize("filename", ["model.idf", "model.epJSON"])
def test_model_round_trip_keeps_objects(tmp_path, filename):
    state = ConfigState()
    state.idf.add(_window())
    state.idf.add(
        ScheduleCompact(
            name="Always_On",
            schedule_type_limits_name="Fraction",
            data=[
                ScheduleCompactDataItem(field="Through: 12/31"),
                ScheduleCompactDataItem(field="For: AllDays"),
                ScheduleCompactDataItem(field="Until: 24:00"),
                ScheduleCompactDataItem(field="1.0"),
            ],
        )
    )

    restored = ConfigState()
    restored.load_model(state.save_model(tmp_path / filename))

    assert _dump(restored) == _dump(state)


def test_save_model_creates_missing_directories(tmp_path):
    path = ConfigState().save_model(tmp_path / "output" / "model" / "model.epJSON")

    assert path.is_file()


def test_save_model_rejects_unknown_suffix(tmp_path):
    with pytest.raises(ValueError, match=r"\.idf or \.epJSON"):
        ConfigState().save_model(tmp_path / "model.yaml")


def test_load_model_rejects_invalid_epjson_object(tmp_path):
    path = tmp_path / "model.epJSON"
    path.write_text('{"Zone": {"Z1": {"multiplier": "many"}}}', encoding="utf-8")

    with pytest.raises(ValueError, match="multiplier"):
        ConfigState().load_model(path)


def test_clear_restores_default_model():
    state = ConfigState()
    state.idf.add(Zone(name="Z1"))

    state.clear()

    assert not state.idf.all_of_type(Zone)
    assert state.idf.all_of_type(Version)


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
