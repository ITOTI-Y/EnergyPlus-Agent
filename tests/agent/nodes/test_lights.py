"""Phase agent test replayed from a recorded LLM cassette.

Record with `pytest --record-mode=once` and a real LLM_API_KEY; tools run for
real on every replay, so the assertions cover prompt -> tool call -> model.
"""

from collections.abc import Callable

import pytest
from idfpy.models.internal_gains import Lights
from idfpy.models.schedules import ScheduleCompact, ScheduleTypeLimits
from idfpy.models.thermal_zones import (
    Zone,
)

from src.agent.nodes.lights import lights_agent
from src.agent.state import AgentState
from src.state.config_state import ConfigState

pytestmark = pytest.mark.usefixtures("pinned_llm_env")


@pytest.mark.vcr
def test_lights_agent_creates_lights(
    constant_schedule: Callable[[str, str, float], ScheduleCompact],
    fraction_limits: ScheduleTypeLimits,
):
    seeded = ConfigState()
    seeded.idf.add(Zone(name="F1_Office"))
    seeded.idf.add(fraction_limits)
    seeded.idf.add(constant_schedule("Office_Lighting", "Fraction", 1.0))
    state = AgentState(
        config_state=seeded,
        user_input=(
            "Create exactly one Lights object named 'F1_Office_Lights' for "
            "zone 'F1_Office': method Watts/Area at 10 W/m2, schedule "
            "'Office_Lighting'."
        ),
    )

    out = lights_agent(state)

    lights = out["config_state"].idf.all_of_type(Lights)
    assert "F1_Office_Lights" in lights
    assert str(out["messages"][0].content).startswith("[lights]")
