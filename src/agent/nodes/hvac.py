from langchain_core.messages import AIMessage
from pydantic import Field

from src.agent.llm import build_agent
from src.agent.nodes._share import (
    PhaseReport,
    invoke_with_self_repair,
    last_message_text,
    missing_input_issues,
    skipped,
    with_feedback,
)
from src.agent.state import AgentState, AgentStateUpdate
from src.agent.tools import make_hvac_tools
from src.agent.trace import TraceCollector, record_phase_trace, trace_middleware

HVAC_SYSTEM_PROMPT = """You are an HVAC configuration expert for EnergyPlus.
Given HVAC specifications, create thermostat templates and one
IdealLoadsAirSystem per conditioned zone.

Workflow:
1. FIRST call `list_schedules` to see the exact names of all Schedule:Compact
   objects (you need these for setpoint + availability references).
2. FIRST call `list_zones` to see the exact zone names (you need these for
   create_ideal_loads_systems).
3. Create one HVACTemplate:Thermostat via create_thermostat for each
   distinct pair of heating and cooling setpoint schedules, NOT one per
   zone, using schedule names from step 1.
4. Give every conditioned zone an HVACTemplate:Zone:IdealLoadsAirSystem:
   call `create_ideal_loads_systems` once per thermostat, listing its zones
   in `zone_names`. Its reply names only zones that failed; the rest got
   their system. Do not list afterwards.

Rules:
- Zone names, `heating_setpoint_schedule_name`, `cooling_setpoint_schedule_name`,
  `template_thermostat_name`, `system_availability_schedule_name` MUST all
  appear verbatim in the respective list_* results.
- If a needed zone or schedule is missing, do NOT invent a name and do NOT
  list again: give your final answer at once, with it in `missing_inputs`.
- Every zone with the same setpoint schedules references the same
  thermostat template. Each zone is still controlled on its own: EnergyPlus
  expands the template into a separate ZoneControl:Thermostat per zone.
- Use the thermostat names the specification gives; otherwise name each
  after its setpoint group, e.g. 'Office_Thermostat'.
- Setpoint values live in the schedules; this phase only references them.
"""


class HVACResponse(PhaseReport):
    """Structured summary returned by the HVAC phase agent."""

    thermostat_names: list[str] = Field(description="Names of all thermostats created")
    ideal_loads_zone_names: list[str] = Field(
        description="Zone names that received an IdealLoadsAirSystem"
    )
    summary: str = Field(description="One-line summary of the HVAC creation result")


def hvac_agent(state: AgentState) -> AgentStateUpdate:
    if skipped(state, "hvac"):
        return AgentStateUpdate()
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
    result = invoke_with_self_repair(
        agent, local, with_feedback(specs, state, "hvac"), phase="hvac"
    )

    response: HVACResponse | None = result.get("structured_response")
    summary = response.summary if response else last_message_text(result)

    record_phase_trace("hvac", collector.export())
    return AgentStateUpdate(
        config_state=local,
        build_issues=missing_input_issues("hvac", response),
        messages=[AIMessage(content=f"[hvac] {summary}")],
    )
