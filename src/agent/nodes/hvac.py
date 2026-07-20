from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.types import Command

from src.agent.llm import create_llm
from src.agent.nodes._share import (
    clone_for_phase,
    invoke_with_self_repair,
    maybe_backhop,
)
from src.agent.react import build_react_agent
from src.agent.state import AgentState, AgentStateUpdate
from src.agent.tools import make_hvac_tools
from src.agent.tools.rag_tools import _get_rag
from src.agent.trace import TraceCollector, record_phase_trace

HVAC_SYSTEM_PROMPT = """You are an HVAC configuration expert for EnergyPlus.
Given HVAC specifications, create Thermostat templates and one
IdealLoadsAirSystem per conditioned zone.

Workflow:
1. FIRST call `list_schedules` to see the exact names of all Schedule:Compact
   objects (you need these for setpoint + availability references).
2. FIRST call `list_zones` to see the exact zone names (you need these for
   create_ideal_loads_system).
3. Create one or more HVACTemplate:Thermostat via create_thermostat, using
   schedule names from step 1.
4. For each conditioned zone, create HVACTemplate:Zone:IdealLoadsAirSystem
   via create_ideal_loads_system(zone_name=..., template_thermostat_name=...).
5. Call list_thermostats and list_ideal_loads_systems once at the end.

Rules:
- `zone_name`, `heating_setpoint_schedule_name`, `cooling_setpoint_schedule_name`,
  `template_thermostat_name`, `system_availability_schedule_name` MUST all
  appear verbatim in the respective list_* results.
- If a needed zone or schedule is missing, STOP and report; do NOT invent names.
- Typical office setpoints: heating 20 C occupied / 15 C unoccupied,
  cooling 24 C occupied / 28 C unoccupied.
- If the spec gives one thermostat for all zones, reuse the same
  template_thermostat_name across all zones.

Reference database:
- Call search_energyplus_reference with location and season to look up
  design-day parameters (e.g. 'Beijing summer design day dry bulb temperature').
  Use the returned values as context for thermostat setpoints or HVAC sizing
  notes. These are reference values only — do not create SizingPeriod objects
  unless explicitly requested.
"""


# Legal back-hop targets for hvac: a missing zone hops to zone; a missing
# schedule hops to schedule. Declared on the return type so LangGraph accepts
# Command(goto=...). (Mirrors surface.py / construction.py.)
_HvacRoute = Literal["zone", "schedule"]


def hvac_agent(state: AgentState) -> Command[_HvacRoute] | AgentStateUpdate:
    local = clone_for_phase(state)
    tools = make_hvac_tools(local, rag=_get_rag())
    collector = TraceCollector(phase="hvac")

    agent = build_react_agent(
        llm=create_llm(),
        tools=tools,
        system_prompt=HVAC_SYSTEM_PROMPT,
        trace_collector=collector,
    )

    specs = state.intake_output.hvac_specs if state.intake_output else state.user_input
    result = invoke_with_self_repair(
        agent,
        local,
        specs,
        phase="hvac",
        is_revision=state.is_revision,
        validation_errors=state.validation_errors,
    )

    record_phase_trace("hvac", collector.export())

    # Back-hop: a missing zone / schedule (detected by invoke_with_self_repair)
    # routes to the owning earlier phase so it can create the object, then
    # normal graph edges carry flow back forward to hvac.
    hop = maybe_backhop(result, state, local, "hvac")
    if hop is not None:
        return hop

    final = [
        m for m in result["messages"] if isinstance(m, AIMessage) and not m.tool_calls
    ]
    summary = final[-1].content if final else "hvac done"

    return AgentStateUpdate(
        config_state=local,
        upstream_request={},  # consume any inbound back-hop request
        messages=[AIMessage(content=f"[hvac] {summary}")],
    )
