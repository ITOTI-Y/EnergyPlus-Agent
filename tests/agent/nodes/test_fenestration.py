"""Phase agent test replayed from a recorded LLM cassette.

Record with `pytest --record-mode=once` and a real LLM_API_KEY; tools run for
real on every replay, so the assertions cover prompt -> tool call -> model.
"""

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

from src.agent.nodes.fenestration import fenestration_agent
from src.agent.state import AgentState
from src.state.config_state import ConfigState

pytestmark = pytest.mark.usefixtures("pinned_llm_env")


@pytest.mark.vcr
def test_fenestration_agent_creates_window(brick: Material):
    seeded = ConfigState()
    seeded.idf.add(Zone(name="F1_Office"))
    seeded.idf.add(brick)
    seeded.idf.add(Construction(name="ExtWall_Simple", outside_layer="Brick_100mm"))
    seeded.idf.add(
        BuildingSurfaceDetailed(
            name="F1_Office_South_Wall",
            surface_type="Wall",
            construction_name="ExtWall_Simple",
            zone_name="F1_Office",
            outside_boundary_condition="Outdoors",
            number_of_vertices=4,
            vertices=[
                BuildingSurfaceDetailedVerticesItem(
                    vertex_x_coordinate=x,
                    vertex_y_coordinate=0.0,
                    vertex_z_coordinate=z,
                )
                for x, z in [(0.0, 3.0), (0.0, 0.0), (6.0, 0.0), (6.0, 3.0)]
            ],
        )
    )
    seeded.idf.add(
        WindowMaterialSimpleGlazingSystem(
            name="Glass_U18", u_factor=1.8, solar_heat_gain_coefficient=0.4
        )
    )
    seeded.idf.add(Construction(name="Window_Simple", outside_layer="Glass_U18"))
    state = AgentState(
        pending_phases=["fenestration"],
        config_state=seeded,
        user_input=(
            "Create exactly one window named 'F1_Office_South_Wall_Window' on "
            "surface 'F1_Office_South_Wall' with construction 'Window_Simple': "
            "1.5m wide x 1.2m tall, centered horizontally, sill at 0.8m."
        ),
    )

    out = fenestration_agent(state)

    fens = out["config_state"].idf.all_of_type(FenestrationSurfaceDetailed)
    assert "F1_Office_South_Wall_Window" in fens
    assert fens["F1_Office_South_Wall_Window"].building_surface_name == (
        "F1_Office_South_Wall"
    )
    assert str(out["messages"][0].content).startswith("[fenestration]")
