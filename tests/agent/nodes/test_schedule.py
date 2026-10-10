"""Phase agent test replayed from a recorded LLM cassette.

Record with `pytest --record-mode=once` and a real LLM_API_KEY; tools run for
real on every replay, so the assertions cover prompt -> tool call -> model.
"""

import pytest
from idfpy.models.schedules import ScheduleCompact

from src.agent.nodes.schedule import schedule_agent
from src.agent.state import AgentState

pytestmark = pytest.mark.usefixtures("pinned_llm_env")


@pytest.mark.vcr
def test_schedule_agent_creates_schedule():
    state = AgentState(
        pending_phases=["schedule"],
        user_input=(
            "Create the ScheduleTypeLimits 'Fraction' (0.0 to 1.0, CONTINUOUS, "
            "Dimensionless) and exactly one Schedule:Compact named "
            "'Office_Occupancy' using it: 1.0 for all days, all year."
        ),
    )

    out = schedule_agent(state)

    schedules = out["config_state"].idf.all_of_type(ScheduleCompact)
    assert "Office_Occupancy" in schedules
    assert str(out["messages"][0].content).startswith("[schedule]")
