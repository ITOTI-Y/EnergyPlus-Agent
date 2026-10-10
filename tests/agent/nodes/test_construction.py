"""Phase agent test replayed from a recorded LLM cassette.

Record with `pytest --record-mode=once` and a real LLM_API_KEY; tools run for
real on every replay, so the assertions cover prompt -> tool call -> model.
"""

import pytest
from idfpy.models.constructions import (
    Construction,
    Material,
)

from src.agent.nodes.construction import construction_agent
from src.agent.state import AgentState
from src.state.config_state import ConfigState

pytestmark = pytest.mark.usefixtures("pinned_llm_env")


@pytest.mark.vcr
def test_construction_agent_creates_construction(brick: Material):
    seeded = ConfigState()
    seeded.idf.add(brick)
    state = AgentState(
        pending_phases=["construction"],
        config_state=seeded,
        user_input=(
            "Create exactly one construction named 'ExtWall_Simple' with a "
            "single layer 'Brick_100mm'."
        ),
    )

    out = construction_agent(state)

    constructions = out["config_state"].idf.all_of_type(Construction)
    assert "ExtWall_Simple" in constructions
    assert constructions["ExtWall_Simple"].outside_layer == "Brick_100mm"
    assert str(out["messages"][0].content).startswith("[construction]")
