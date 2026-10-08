"""Phase agent test replayed from a recorded LLM cassette.

Record with `pytest --record-mode=once` and a real LLM_API_KEY; tools run for
real on every replay, so the assertions cover prompt -> tool call -> model.
"""

from collections.abc import Callable

import pytest
from idfpy.models.hvac_templates import HVACTemplateZoneIdealLoadsAirSystem
from idfpy.models.schedules import ScheduleCompact, ScheduleTypeLimits
from idfpy.models.thermal_zones import (
    Zone,
)

from src.agent.nodes.hvac import hvac_agent
from src.agent.state import AgentState
from src.state.config_state import ConfigState

pytestmark = pytest.mark.usefixtures("pinned_llm_env")


@pytest.mark.vcr
def test_hvac_agent_creates_thermostat_and_ideal_loads(
    constant_schedule: Callable[[str, str, float], ScheduleCompact],
):
    seeded = ConfigState()
    seeded.idf.add(Zone(name="F1_Office"))
    seeded.idf.add(
        ScheduleTypeLimits(
            name="Temperature",
            lower_limit_value=-100.0,
            upper_limit_value=100.0,
            numeric_type="Continuous",
        )
    )
    seeded.idf.add(constant_schedule("Heating_Setpoint", "Temperature", 20.0))
    seeded.idf.add(constant_schedule("Cooling_Setpoint", "Temperature", 24.0))
    state = AgentState(
        pending_phases=["hvac"],
        config_state=seeded,
        user_input=(
            "Create exactly one thermostat named 'Office_Thermostat' using "
            "heating setpoint schedule 'Heating_Setpoint' and cooling setpoint "
            "schedule 'Cooling_Setpoint', then one IdealLoadsAirSystem for "
            "zone 'F1_Office' using that thermostat."
        ),
    )

    out = hvac_agent(state)

    idf = out["config_state"].idf
    assert "Office_Thermostat" in idf.all_of_type("HVACTemplate:Thermostat")
    ideal_loads = idf.all_of_type(HVACTemplateZoneIdealLoadsAirSystem).values()
    assert "F1_Office" in {ils.zone_name for ils in ideal_loads}
    assert str(out["messages"][0].content).startswith("[hvac]")
