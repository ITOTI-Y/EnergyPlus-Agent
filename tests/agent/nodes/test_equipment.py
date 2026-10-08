"""Phase agent test replayed from a recorded LLM cassette.

Record with `pytest --record-mode=once` and a real LLM_API_KEY; tools run for
real on every replay, so the assertions cover prompt -> tool call -> model.
"""

from collections.abc import Callable

import pytest
from idfpy.models.internal_gains import ElectricEquipment
from idfpy.models.schedules import ScheduleCompact, ScheduleTypeLimits
from idfpy.models.thermal_zones import Zone

from src.agent.nodes.equipment import equipment_agent
from src.agent.state import AgentState
from src.state.config_state import ConfigState

pytestmark = pytest.mark.usefixtures("pinned_llm_env")


@pytest.mark.vcr
def test_equipment_agent_creates_equipment(
    constant_schedule: Callable[[str, str, float], ScheduleCompact],
    fraction_limits: ScheduleTypeLimits,
):
    seeded = ConfigState()
    seeded.idf.add(Zone(name="F1_Office"))
    seeded.idf.add(fraction_limits)
    seeded.idf.add(constant_schedule("Office_Equipment", "Fraction", 1.0))
    state = AgentState(
        config_state=seeded,
        user_input=(
            "Create exactly one ElectricEquipment object named "
            "'F1_Office_Equipment' for zone 'F1_Office': method Watts/Area at "
            "12 W/m2, schedule 'Office_Equipment'."
        ),
    )

    out = equipment_agent(state)

    equipment = out["config_state"].idf.all_of_type(ElectricEquipment)
    assert equipment["F1_Office_Equipment"].watts_per_floor_area == 12.0
    assert str(out["messages"][0].content).startswith("[equipment]")
