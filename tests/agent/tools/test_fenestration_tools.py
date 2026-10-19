import json

from idfpy.models import (
    Building,
    Construction,
    FenestrationSurfaceDetailed,
    Material,
    WindowMaterialSimpleGlazingSystem,
    Zone,
)

from src.agent.tools import make_fenestration_tools
from src.modeling.geometry import (
    PlanPointSchema,
    ZoneConstructions,
    create_zone_geometry,
)
from src.state.config_state import ConfigState


def _two_plans() -> ConfigState:
    """Plan A (zone S1_A) and, beside it, plan B (zone S1_B)."""
    config = ConfigState()
    idf = config.idf
    idf.add(Building(name="B"))
    idf.add(
        WindowMaterialSimpleGlazingSystem(
            name="Glass", u_factor=2.0, solar_heat_gain_coefficient=0.4
        )
    )
    idf.add(Construction(name="Window", outside_layer="Glass"))
    idf.add(
        Material(
            name="Brick",
            roughness="Rough",
            thickness=0.2,
            conductivity=0.9,
            density=1900.0,
            specific_heat=800.0,
        )
    )
    idf.add(Construction(name="Ext", outside_layer="Brick"))
    for name, x0 in (("S1_A", 0), ("S1_B", 10)):
        idf.add(Zone(name=name))
        corners = [(x0, 0), (x0 + 10, 0), (x0 + 10, 8), (x0, 8)]
        create_zone_geometry(
            idf,
            name,
            [PlanPointSchema(X=x, Y=y) for x, y in corners],
            0.0,
            3.5,
            ZoneConstructions("Ext", "Ext", "Ext", "Ext", "Ext"),
        )
    return config


def _windows_by_ratio(config: ConfigState):
    tools = {
        t.name: t
        for t in make_fenestration_tools(config, {"A": ["S1_A"], "B": ["S1_B"]})
    }
    return tools["create_windows_by_ratio"]


def test_windows_given_by_plan_go_on_the_plan_zones_only():
    config = _two_plans()

    reply = json.loads(
        _windows_by_ratio(config).invoke(
            {
                "window_to_wall_ratio": 0.3,
                "construction_name": "Window",
                "plans": ["A"],
                "facings": ["South"],
            }
        )
    )

    assert reply["success"]
    walls = {
        f.building_surface_name
        for f in config.idf.all_of_type(FenestrationSurfaceDetailed).values()
    }
    assert len(walls) == 1 and all(w.startswith("S1_A") for w in walls)


def test_an_unknown_plan_is_refused_with_the_known_ones():
    reply = _windows_by_ratio(_two_plans()).invoke(
        {"window_to_wall_ratio": 0.3, "construction_name": "Window", "plans": ["Tower"]}
    )

    assert "unknown plans ['Tower']; the plans are ['A', 'B']" in reply


def test_windows_given_by_zone_name_go_on_those_zones_only():
    config = _two_plans()

    _windows_by_ratio(config).invoke(
        {
            "window_to_wall_ratio": 0.3,
            "construction_name": "Window",
            "zone_names": ["S1_B"],
        }
    )

    walls = {
        f.building_surface_name
        for f in config.idf.all_of_type(FenestrationSurfaceDetailed).values()
    }
    assert walls and all(w.startswith("S1_B") for w in walls)


def test_windows_without_plans_or_zones_are_refused_not_put_everywhere():
    config = _two_plans()

    reply = _windows_by_ratio(config).invoke(
        {"window_to_wall_ratio": 0.3, "construction_name": "Window"}
    )

    assert "give the zones by `plans`, `zone_names` or both" in reply
    assert not config.idf.all_of_type(FenestrationSurfaceDetailed)
