from langchain_core.messages import AIMessage, HumanMessage

from src.agent.llm import create_llm
from src.agent.react import ReactState, build_react_agent
from src.agent.state import AgentState, AgentStateUpdate
from src.agent.tools import make_hvac_tools
from src.agent.trace import TRACE_STORE, TraceCollector

HVAC_SYSTEM_PROMPT = """You are an HVAC configuration expert for EnergyPlus.
Given HVAC specifications, create Thermostat templates and one
IdealLoadsAirSystem per conditioned zone.

Steps:
1. Create one or more HVACTemplate:Thermostat via create_thermostat.
   - heating_setpoint_schedule_name and cooling_setpoint_schedule_name
     MUST reference existing Schedule:Compact objects of Temperature type.
2. For each conditioned zone, create an HVACTemplate:Zone:IdealLoadsAirSystem
   via create_ideal_loads_system(zone_name=..., template_thermostat_name=...).
   - zone_name is the IdealLoadsSystem's identity key (one system per zone).
   - system_availability_schedule_name is optional (defaults to always on).

Rules:
- Typical office setpoints: heating 20 C occupied / 15 C unoccupied,
  cooling 24 C occupied / 28 C unoccupied.
- If the spec gives one thermostat for all zones, reuse the same
  template_thermostat_name across all zones.
- Call list_thermostats and list_ideal_loads_systems once at the end.
"""


def hvac_agent(state: AgentState) -> AgentStateUpdate:
    local = state.config_state.model_copy(deep=True)
    tools = make_hvac_tools(local)
    collector = TraceCollector(phase="hvac")

    agent = build_react_agent(
        llm=create_llm(),
        tools=tools,
        system_prompt=HVAC_SYSTEM_PROMPT,
        trace_collector=collector,
    )

    specs = state.intake_output.hvac_specs if state.intake_output else state.user_input
    result = agent.invoke(ReactState(messages=[HumanMessage(content=specs)]))

    final = [
        m for m in result["messages"] if isinstance(m, AIMessage) and not m.tool_calls
    ]
    summary = final[-1].content if final else "hvac done"

    TRACE_STORE.setdefault("hvac", []).extend(collector.export())
    return AgentStateUpdate(
        config_state=local,
        messages=[AIMessage(content=f"[hvac] {summary}")],
    )
