from idfpy.models.schedules import ScheduleCompact, ScheduleCompactDataItem
from idfpy.models.thermal_zones import FenestrationSurfaceDetailed

from src.mcp.state import ConfigState


def test_yaml_roundtrip_preserves_fenestration_vertices(tmp_path):
    state = ConfigState()
    state.idf.add(
        FenestrationSurfaceDetailed(
            name="South_Window",
            surface_type="Window",
            construction_name="Window_Construction",
            building_surface_name="South_Wall",
            number_of_vertices=4,
            vertex_1_x_coordinate=1.0,
            vertex_1_y_coordinate=0.0,
            vertex_1_z_coordinate=1.0,
            vertex_2_x_coordinate=4.0,
            vertex_2_y_coordinate=0.0,
            vertex_2_z_coordinate=1.0,
            vertex_3_x_coordinate=4.0,
            vertex_3_y_coordinate=0.0,
            vertex_3_z_coordinate=2.5,
            vertex_4_x_coordinate=1.0,
            vertex_4_y_coordinate=0.0,
            vertex_4_z_coordinate=2.5,
        )
    )

    yaml_path = tmp_path / "config.yaml"
    state.export_yaml(yaml_path)
    restored = ConfigState.load_yaml(yaml_path)

    window = restored.idf.get(FenestrationSurfaceDetailed, "South_Window")
    assert window is not None
    assert window.vertex_1_x_coordinate == 1.0
    assert window.vertex_4_z_coordinate == 2.5


def test_yaml_roundtrip_preserves_compact_schedule_fields(tmp_path):
    state = ConfigState()
    state.idf.add(
        ScheduleCompact(
            name="Always_On",
            schedule_type_limits_name="Fraction",
            data=[
                ScheduleCompactDataItem(field="Through: 12/31"),
                ScheduleCompactDataItem(field="For: AllDays"),
                ScheduleCompactDataItem(field="Until: 24:00, 1.0"),
            ],
        )
    )

    yaml_path = tmp_path / "config.yaml"
    state.export_yaml(yaml_path)
    restored = ConfigState.load_yaml(yaml_path)

    schedule = restored.idf.get(ScheduleCompact, "Always_On")
    assert schedule is not None
    assert schedule.data is not None
    assert [item.field for item in schedule.data] == [
        "Through: 12/31",
        "For: AllDays",
        "Until: 24:00, 1.0",
    ]
