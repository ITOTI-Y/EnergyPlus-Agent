from idfpy.models.thermal_zones import (
    BuildingSurfaceDetailed,
    BuildingSurfaceDetailedVerticesItem,
    Zone,
)

from src.mcp.state import ConfigState
from src.mcp.tools.workflow import validate_geometry_completeness


def _add_zone(state: ConfigState, name: str = "Office") -> None:
    state.idf.add(Zone(name=name))


def _add_surface(state: ConfigState, zone_name: str = "Office") -> None:
    state.idf.add(
        BuildingSurfaceDetailed(
            name=f"{zone_name}_South_Wall",
            surface_type="Wall",
            construction_name="Wall_Construction",
            zone_name=zone_name,
            outside_boundary_condition="Outdoors",
            sun_exposure="SunExposed",
            wind_exposure="WindExposed",
            vertices=[
                BuildingSurfaceDetailedVerticesItem(
                    vertex_x_coordinate=0.0,
                    vertex_y_coordinate=0.0,
                    vertex_z_coordinate=0.0,
                ),
                BuildingSurfaceDetailedVerticesItem(
                    vertex_x_coordinate=5.0,
                    vertex_y_coordinate=0.0,
                    vertex_z_coordinate=0.0,
                ),
                BuildingSurfaceDetailedVerticesItem(
                    vertex_x_coordinate=5.0,
                    vertex_y_coordinate=0.0,
                    vertex_z_coordinate=3.0,
                ),
                BuildingSurfaceDetailedVerticesItem(
                    vertex_x_coordinate=0.0,
                    vertex_y_coordinate=0.0,
                    vertex_z_coordinate=3.0,
                ),
            ],
        )
    )


def test_geometry_preflight_rejects_empty_model():
    errors = validate_geometry_completeness(ConfigState())

    assert any("0 Zone objects" in e for e in errors)
    assert any("0 BuildingSurface:Detailed objects" in e for e in errors)


def test_geometry_preflight_rejects_zone_without_surfaces():
    state = ConfigState()
    _add_zone(state, "Office")

    errors = validate_geometry_completeness(state)

    assert any("0 BuildingSurface:Detailed objects" in e for e in errors)


def test_geometry_preflight_rejects_one_zone_without_surface():
    state = ConfigState()
    _add_zone(state, "Office")
    _add_zone(state, "Storage")
    _add_surface(state, "Office")

    errors = validate_geometry_completeness(state)

    assert any("Zone 'Storage' has 0 BuildingSurface:Detailed" in e for e in errors)


def test_geometry_preflight_accepts_zone_with_surface():
    state = ConfigState()
    _add_zone(state, "Office")
    _add_surface(state, "Office")

    assert validate_geometry_completeness(state) == []
