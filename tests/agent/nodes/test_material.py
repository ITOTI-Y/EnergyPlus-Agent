"""Phase agent test replayed from a recorded LLM cassette.

Record with `pytest --record-mode=once` and a real LLM_API_KEY; tools run for
real on every replay, so the assertions cover prompt -> tool call -> model.
"""

import pytest
from idfpy.models.constructions import (
    Material,
)
from langgraph.runtime import Runtime

from src.agent.nodes.material import material_agent
from src.agent.state import AgentState, SimContext

pytestmark = pytest.mark.usefixtures("pinned_llm_env")


@pytest.mark.vcr
def test_material_agent_creates_material(
    runtime: Runtime[SimContext],
):
    state = AgentState(
        pending_phases=["material"],
        user_input=(
            "Create exactly one standard material named 'Brick_100mm': "
            "thickness 0.1 m, conductivity 0.89 W/m-K, density 1920 kg/m3, "
            "specific heat 790 J/kg-K, roughness MediumRough."
        ),
    )

    out = material_agent(state, runtime)

    materials = out["config_state"].idf.all_of_type(Material)
    assert "Brick_100mm" in materials
    assert materials["Brick_100mm"].thickness == 0.1
    assert str(out["messages"][0].content).startswith("[material]")
