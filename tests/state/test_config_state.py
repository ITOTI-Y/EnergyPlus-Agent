import pytest
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
