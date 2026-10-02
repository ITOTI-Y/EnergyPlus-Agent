"""Phase agent test replayed from a recorded LLM cassette.

Record with `pytest --record-mode=once` and a real LLM_API_KEY; tools run for
real on every replay, so the assertions cover prompt -> tool call -> model.
"""

import pytest
from idfpy.models.constructions import (
    Construction,
    Material,
)
from idfpy.models.thermal_zones import (
    BuildingSurfaceDetailed,
    Zone,
)

from src.agent.nodes.surface import surface_agent
from src.agent.state import AgentState
from src.state.config_state import ConfigState

pytestmark = pytest.mark.usefixtures("pinned_llm_env")


@pytest.mark.vcr
def test_surface_agent_creates_surface(brick: Material):
    seeded = ConfigState()
    seeded.idf.add(Zone(name="F1_Office"))
    seeded.idf.add(brick)
    seeded.idf.add(Construction(name="ExtWall_Simple", outside_layer="Brick_100mm"))
    state = AgentState(
        config_state=seeded,
        user_input=(
            "Create exactly one floor surface named 'F1_Office_Floor' for zone "
            "'F1_Office' with construction 'ExtWall_Simple': a 6m x 6m square "
            "at ground level (z=0, corners (0,0), (6,0), (6,6), (0,6)), "
            "outside boundary condition Ground."
        ),
    )

    out = surface_agent(state)

    surfaces = out["config_state"].idf.all_of_type(BuildingSurfaceDetailed)
    assert "F1_Office_Floor" in surfaces
    assert surfaces["F1_Office_Floor"].zone_name == "F1_Office"
    assert str(out["messages"][0].content).startswith("[surface]")
