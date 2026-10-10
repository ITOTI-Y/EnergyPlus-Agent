"""Surface phase: zones extruded in code, sloped geometry by the LLM.

The LLM test replays a recorded cassette; record with
`pytest --record-mode=once` and a real LLM_API_KEY.
"""

import pytest
from idfpy.models.constructions import Construction, Material
from idfpy.models.thermal_zones import BuildingSurfaceDetailed, Zone

from src.agent.nodes.surface import surface_agent
from src.agent.phases import owner
from src.agent.state import AgentState
from src.modeling.validation import model_issues
from src.state.config_state import ConfigState
from tests.agent.intake_data import intake, zone_spec


def _seeded(brick: Material, *zones: str) -> ConfigState:
    seeded = ConfigState()
    seeded.idf.add(brick)
    for name in ("Wall", "Slab"):
        seeded.idf.add(Construction(name=name, outside_layer=brick.name))
    for zone in zones:
        seeded.idf.add(Zone(name=zone))
    return seeded


def _surfaces(state: ConfigState, zone: str) -> list[BuildingSurfaceDetailed]:
    return [
        s
        for s in state.idf.all_of_type(BuildingSurfaceDetailed).values()
        if s.zone_name == zone
    ]


def _closed(surfaces: list[BuildingSurfaceDetailed]) -> bool:
    total = [sum(s.area * s.normal[k] for s in surfaces) for k in range(3)]
    return max(map(abs, total)) < 1e-3


def test_zones_are_extruded_and_paired(brick: Material):
    output = intake(
        zones=[
            zone_spec("West", [(0, 0), (5, 0), (5, 8), (0, 8)]),
            zone_spec("East", [(5, 0), (10, 0), (10, 8), (5, 8)]),
        ]
    )
    state = AgentState(
        pending_phases=["surface"],
        intake_output=output,
        config_state=_seeded(brick, "West", "East"),
    )

    out = surface_agent(state)

    model = out["config_state"]
    assert out["build_issues"] == []
    assert model_issues(model.idf) == []
    assert all(_closed(_surfaces(model, zone)) for zone in ("West", "East"))
    paired = [
        s.name
        for s in model.idf.all_of_type(BuildingSurfaceDetailed).values()
        if s.outside_boundary_condition == "Surface"
    ]
    assert len(paired) == 2


def test_extrusion_problems_go_to_their_owners(brick: Material):
    output = intake(
        zones=[
            zone_spec("A", [(0, 0), (5, 0), (5, 5), (0, 5)], roof_construction="Tile"),
            zone_spec("B", [(0, 0), (5, 0), (5, 5), (0, 5)]),
            zone_spec("C", [(4, 0), (9, 0), (9, 5), (4, 5)]),
        ]
    )
    state = AgentState(
        pending_phases=["surface"],
        intake_output=output,
        config_state=_seeded(brick, "A", "B", "C"),
    )

    issues = surface_agent(state)["build_issues"]

    # A lacks its roof construction; C overlaps B, which was extruded first.
    assert [(owner(i), i.object_name) for i in issues] == [
        ("construction", "Tile"),
        ("zone", "C"),
    ]


@pytest.mark.vcr
@pytest.mark.usefixtures("pinned_llm_env")
def test_flat_roof_is_replaced_by_a_gable_roof(brick: Material):
    output = intake(
        zones=[zone_spec("Hall", [(0, 0), (6, 0), (6, 8), (0, 8)])],
        surface_specs=(
            "Replace the flat roof of zone 'Hall' (6 m by 8 m, walls 3 m high) "
            "with a gable roof whose ridge runs along Y at x=3, z=4.5: two roof "
            "planes 'Hall_Roof_West' and 'Hall_Roof_East' with construction "
            "'Wall', and triangular gable walls 'Hall_Gable_South' at y=0 and "
            "'Hall_Gable_North' at y=8 with construction 'Wall', all outdoors."
        ),
    )
    state = AgentState(
        pending_phases=["surface"],
        intake_output=output,
        config_state=_seeded(brick, "Hall"),
    )

    out = surface_agent(state)

    surfaces = _surfaces(out["config_state"], "Hall")
    roofs = [s for s in surfaces if s.surface_type == "Roof"]
    assert sorted(s.name for s in roofs) == ["Hall_Roof_East", "Hall_Roof_West"]
    assert _closed(surfaces)


def _rect_plan(key: str, x0: float, y0: float, x1: float, y1: float) -> dict:
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return {
        "key": key,
        "plan": [{"X": x, "Y": y} for x, y in corners],
        "zoning": "perimeter_core",
        "exterior_wall_construction": "Wall",
        "roof_construction": "Wall",
        "ground_floor_construction": "Slab",
        "interior_wall_construction": "Wall",
        "interior_floor_construction": "Slab",
    }


def tower_on_podium():
    """A 40 x 30 m podium of two storeys under a 30 x 20 m tower of 1 + 10 + 1."""
    return intake(
        zone_plans=[
            _rect_plan("Podium", 0, 0, 40, 30),
            _rect_plan("Tower", 5, 5, 35, 25),
        ],
        storeys=[
            {"name": "G", "height": 4.5, "zones": [{"plan": "Podium"}]},
            {"name": "L2", "height": 4.0, "zones": [{"plan": "Podium"}]},
            {"name": "L3", "height": 3.8, "zones": [{"plan": "Tower"}]},
            {"name": "T", "height": 3.8, "multiplier": 10, "zones": [{"plan": "Tower"}]},
            {"name": "Top", "height": 3.8, "zones": [{"plan": "Tower"}]},
        ],
    )  # fmt: skip


def test_perimeter_core_plans_extrude_into_closed_paired_zones(brick: Material):
    output = tower_on_podium()
    names = [z.name for z in output.zones]
    seeded = _seeded(brick, *names)
    for zone in output.zones:
        seeded.idf.all_of_type(Zone)[zone.name].multiplier = zone.multiplier
    state = AgentState(
        pending_phases=["surface"], intake_output=output, config_state=seeded
    )

    out = surface_agent(state)

    assert names[:5] == [
        "G_Podium_S",
        "G_Podium_E",
        "G_Podium_N",
        "G_Podium_W",
        "G_Podium_Core",
    ]
    assert len(names) == 25
    model = out["config_state"]
    assert out["build_issues"] == []
    assert model_issues(model.idf) == []
    assert all(_closed(_surfaces(model, zone)) for zone in names)
