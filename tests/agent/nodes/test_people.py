"""Phase agent test replayed from a recorded LLM cassette.

Record with `pytest --record-mode=once` and a real LLM_API_KEY; tools run for
real on every replay, so the assertions cover prompt -> tool call -> model.
"""

from collections.abc import Callable

import pytest
from idfpy.models.internal_gains import People
from idfpy.models.schedules import ScheduleCompact, ScheduleTypeLimits
from idfpy.models.thermal_zones import (
    Zone,
)

from src.agent.nodes.people import people_agent
from src.agent.state import AgentState
from src.state.config_state import ConfigState

pytestmark = pytest.mark.usefixtures("pinned_llm_env")


@pytest.mark.vcr
def test_people_agent_creates_people(
    constant_schedule: Callable[[str, str, float], ScheduleCompact],
    fraction_limits: ScheduleTypeLimits,
):
    seeded = ConfigState()
    seeded.idf.add(Zone(name="F1_Office"))
    seeded.idf.add(fraction_limits)
    seeded.idf.add(
        ScheduleTypeLimits(
            name="Activity Level",
            lower_limit_value=0.0,
            upper_limit_value=1000.0,
            numeric_type="Continuous",
        )
    )
    seeded.idf.add(constant_schedule("Office_Occupancy", "Fraction", 1.0))
    seeded.idf.add(constant_schedule("Office_Activity", "Activity Level", 120.0))
    state = AgentState(
        config_state=seeded,
        user_input=(
            "Create exactly one People object named 'F1_Office_People' for "
            "zone 'F1_Office': method People/Area with 0.1 people/m2, "
            "occupancy schedule 'Office_Occupancy', activity level schedule "
            "'Office_Activity'."
        ),
    )

    out = people_agent(state)

    people = out["config_state"].idf.all_of_type(People)
    assert "F1_Office_People" in people
    assert str(out["messages"][0].content).startswith("[people]")
