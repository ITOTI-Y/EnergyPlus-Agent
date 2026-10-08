from langchain_core.messages import AIMessage

from src.agent.llm import build_agent
from src.agent.nodes._share import (
    invoke_with_self_repair,
    last_message_text,
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
   create_ideal_loads_system).
3. Create one HVACTemplate:Thermostat via create_thermostat for each
   distinct pair of heating and cooling setpoint schedules, NOT one per
   zone, using schedule names from step 1.
4. For each conditioned zone, create HVACTemplate:Zone:IdealLoadsAirSystem
   via create_ideal_loads_system(zone_name=..., template_thermostat_name=...).
5. Call list_thermostats and list_ideal_loads_systems once at the end.

Rules:
- `zone_name`, `heating_setpoint_schedule_name`, `cooling_setpoint_schedule_name`,
  `template_thermostat_name`, `system_availability_schedule_name` MUST all
  appear verbatim in the respective list_* results.
- If a needed zone or schedule is missing, STOP and report; do NOT invent names.
- Every zone with the same setpoint schedules references the same
  thermostat template. Each zone is still controlled on its own: EnergyPlus
  expands the template into a separate ZoneControl:Thermostat per zone.
- Use the thermostat names the specification gives; otherwise name each
  after its setpoint group, e.g. 'Office_Thermostat'.
- Setpoint values live in the schedules; this phase only references them.
"""


def hvac_agent(state: AgentState) -> AgentStateUpdate:
    if skipped(state, "hvac"):
        return AgentStateUpdate()
    local = state.config_state.model_copy(deep=True)
    tools = make_hvac_tools(local)
    collector = TraceCollector(phase="hvac")

    agent = build_agent(
        tools=tools,
        system_prompt=HVAC_SYSTEM_PROMPT,
        middleware=[trace_middleware(collector)],
    )

    specs = state.intake_output.hvac_specs if state.intake_output else state.user_input
    result = invoke_with_self_repair(
        agent, local, with_feedback(specs, state, "hvac"), phase="hvac"
    )

    summary = last_message_text(result)

    record_phase_trace("hvac", collector.export())
    return AgentStateUpdate(
        config_state=local,
        messages=[AIMessage(content=f"[hvac] {summary}")],
    )
