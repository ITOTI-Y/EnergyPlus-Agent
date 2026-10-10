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
