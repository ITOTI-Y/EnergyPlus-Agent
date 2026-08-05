from typing import Literal

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from src.agent.llm import build_agent
from src.agent.nodes._share import invoke_with_self_repair
from src.agent.state import AgentState, AgentStateUpdate
from src.agent.tools import make_hvac_tools
from src.agent.trace import TraceCollector, record_phase_trace, trace_middleware

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
"""


class HVACResponse(BaseModel):
    """Structured summary returned by the HVAC phase agent."""

    thermostat_names: list[str] = Field(description="Names of all thermostats created")
    ideal_loads_zone_names: list[str] = Field(
        description="Zone names that received an IdealLoadsAirSystem"
    )
    summary: str = Field(description="One-line summary of the HVAC creation result")


def hvac_agent(state: AgentState) -> AgentStateUpdate:
    local = state.config_state.model_copy(deep=True)
    tools = make_hvac_tools(local)
    collector = TraceCollector(phase="hvac")

    agent = build_agent(
        tools=tools,
        system_prompt=HVAC_SYSTEM_PROMPT,
        response_format=HVACResponse,
        middleware=[trace_middleware(collector)],
    )

    specs = state.intake_output.hvac_specs if state.intake_output else state.user_input
    result = invoke_with_self_repair(agent, local, specs, phase="hvac")

    response: HVACResponse | None = result.get("structured_response")
    summary = response.summary if response else "hvac done"

    record_phase_trace("hvac", collector.export())
    return AgentStateUpdate(
        config_state=local,
        upstream_request={},  # consume any inbound back-hop request
        messages=[AIMessage(content=f"[hvac] {summary}")],
    )
