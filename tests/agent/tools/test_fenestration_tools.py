import json

import pytest
from idfpy.models.constructions import (
    Construction,
    Material,
    WindowMaterialSimpleGlazingSystem,
)
from idfpy.models.thermal_zones import (
    BuildingSurfaceDetailed,
    BuildingSurfaceDetailedVerticesItem,
    FenestrationSurfaceDetailed,
    Zone,
)

from src.agent.tools.fenestration_tools import make_fenestration_tools
from src.mcp.state import ConfigState


def _window(config: ConfigState) -> FenestrationSurfaceDetailed:
    obj = config.idf.get(FenestrationSurfaceDetailed, "Office_Window")
    assert obj is not None
    return obj


def _update_tool(config: ConfigState):
    return next(
        tool
        for tool in make_fenestration_tools(config)
        if tool.name == "update_fenestration"
    )


def _config() -> ConfigState:
    config = ConfigState()
    config.idf.add(Zone(name="Office"))
    config.idf.add(
        Material(
            name="Brick",
            roughness="MediumRough",
            thickness=0.1,
            conductivity=0.89,
            density=1920.0,
            specific_heat=790.0,
        )
    )
    config.idf.add(Construction(name="Wall_Construction", outside_layer="Brick"))
    config.idf.add(
        BuildingSurfaceDetailed(
            name="Office_South_Wall",
            surface_type="Wall",
            construction_name="Wall_Construction",
            zone_name="Office",
            outside_boundary_condition="Outdoors",
            sun_exposure="SunExposed",
            wind_exposure="WindExposed",
            number_of_vertices=4,
            vertices=[
                BuildingSurfaceDetailedVerticesItem(
                    vertex_x_coordinate=x,
                    vertex_y_coordinate=0.0,
                    vertex_z_coordinate=z,
                )
                for x, z in ((0.0, 3.0), (0.0, 0.0), (6.0, 0.0), (6.0, 3.0))
            ],
        )
    )
    for material_name in ("Glass_1", "Glass_2"):
        config.idf.add(
            WindowMaterialSimpleGlazingSystem(
                name=material_name,
                u_factor=1.8,
                solar_heat_gain_coefficient=0.4,
            )
        )
        config.idf.add(
            Construction(
                name=f"{material_name}_Construction", outside_layer=material_name
            )
        )
    config.idf.add(
        FenestrationSurfaceDetailed(
            name="Office_Window",
            surface_type="Window",
            construction_name="Glass_1_Construction",
            building_surface_name="Office_South_Wall",
            multiplier=1,
            number_of_vertices=4,
            vertex_1_x_coordinate=1.0,
            vertex_1_y_coordinate=0.0,
            vertex_1_z_coordinate=0.8,
            vertex_2_x_coordinate=3.0,
            vertex_2_y_coordinate=0.0,
            vertex_2_z_coordinate=0.8,
            vertex_3_x_coordinate=3.0,
            vertex_3_y_coordinate=0.0,
            vertex_3_z_coordinate=2.0,
            vertex_4_x_coordinate=1.0,
            vertex_4_y_coordinate=0.0,
            vertex_4_z_coordinate=2.0,
        )
    )
    return config


@pytest.mark.parametrize(
    "vertices",
    [
        [
            {"X": 0.0, "Y": 0.0, "Z": 0.0},
            {"X": 1.0, "Y": 0.0, "Z": 0.0},
        ],
        [{"X": float(index), "Y": 0.0, "Z": 0.0} for index in range(5)],
        [
            {"X": 0.0, "Y": 0.0, "Z": 0.0},
            {"X": 1.0, "Y": 0.0, "Z": 0.0},
            {"X": 1.0, "Y": 0.0},
        ],
        [
            {"X": 0.0, "Y": 0.0, "Z": 0.0},
            {"X": 1.0, "Y": 0.0, "Z": 0.0},
            {"X": 1.0, "Y": 0.0, "Z": float("nan")},
        ],
    ],
    ids=["too-few", "too-many", "missing-axis", "non-finite"],
)
def test_update_fenestration_rejects_invalid_vertices_atomically(vertices) -> None:
    config = _config()
    obj = _window(config)
    before = obj.model_dump()

    result = json.loads(
        _update_tool(config).invoke(
            {
                "name": "Office_Window",
                "construction_name": "Glass_2_Construction",
                "vertices": vertices,
            }
        )
    )

    assert not result["success"]
    assert obj.model_dump() == before


def test_update_fenestration_validates_all_fields_before_mutation() -> None:
    config = _config()
    obj = _window(config)
    before = obj.model_dump()

    result = json.loads(
        _update_tool(config).invoke(
            {
                "name": "Office_Window",
                "construction_name": "Glass_2_Construction",
                "multiplier": 0,
            }
        )
    )

    assert not result["success"]
    assert obj.model_dump() == before


def test_update_fenestration_applies_valid_fields_together() -> None:
    config = _config()
    obj = _window(config)
    vertices = [
        {"X": 1.0, "Y": 0.0, "Z": 0.8},
        {"X": 2.0, "Y": 0.0, "Z": 0.8},
        {"X": 1.5, "Y": 0.0, "Z": 1.8},
    ]

    result = json.loads(
        _update_tool(config).invoke(
            {
                "name": "Office_Window",
                "construction_name": "Glass_2_Construction",
                "multiplier": 2,
                "vertices": vertices,
            }
        )
    )

    assert result["success"]
    assert obj.construction_name == "Glass_2_Construction"
    assert obj.multiplier == 2
    assert obj.number_of_vertices == 3
    assert obj.vertex_4_x_coordinate is None
