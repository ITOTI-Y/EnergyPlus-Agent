from langchain_core.messages import AIMessage

from src.agent.llm import create_llm
from src.agent.nodes._share import clone_for_phase, invoke_with_self_repair
from src.agent.react import build_react_agent
from src.agent.state import AgentState, AgentStateUpdate
from src.agent.tools import make_zone_tools
from src.agent.trace import TraceCollector, record_phase_trace

ZONE_SYSTEM_PROMPT = """You are a thermal zone creation expert for EnergyPlus.
Given zone specifications, create all required zones using create_zone tool.

Rules:
- Zone names must be unique and descriptive, typically '{floor}_{usage}_{direction}'
  (e.g., 'F1_Office_North', 'F2_Corridor').
- Set z_origin to the floor's lower elevation: ground floor = 0,
  floor 2 = first-floor height (e.g., 3.0), etc.
- direction_of_relative_north is 0 unless the description specifies a rotation.
- multiplier is 1 unless the description explicitly duplicates a typical floor.
- After creating all zones, call list_zones once to verify, then stop with
  a one-line summary of zone count and names.
"""


def zone_agent(state: AgentState) -> AgentStateUpdate:
    local = clone_for_phase(state)
    tools = make_zone_tools(local)
    collector = TraceCollector(phase="zone")

    agent = build_react_agent(
        llm=create_llm(),
        tools=tools,
        system_prompt=ZONE_SYSTEM_PROMPT,
        trace_collector=collector,
    )

    specs = state.intake_output.zone_specs if state.intake_output else state.user_input
    # If reached via a back-hop from surface (needed a zone), append.
    upstream = state.upstream_request
    if upstream and upstream.get("target") == "zone":
        specs = f"{specs}\n\n{upstream['specs']}"
    result = invoke_with_self_repair(
        agent,
        local,
        specs,
        phase="zone",
        is_revision=state.is_revision,
        validation_errors=state.validation_errors,
    )

    final = [
        m for m in result["messages"] if isinstance(m, AIMessage) and not m.tool_calls
    ]
    summary = final[-1].content if final else "zone done"

    record_phase_trace("zone", collector.export())

    return AgentStateUpdate(
        config_state=local,
        messages=[AIMessage(content=f"[zone] {summary}")],
    )
