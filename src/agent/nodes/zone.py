from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from src.agent.llm import build_agent
from src.agent.state import AgentState, AgentStateUpdate
from src.agent.tools import make_zone_tools
from src.agent.trace import TraceCollector, record_phase_trace, trace_middleware

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


class ZoneResponse(BaseModel):
    """Structured summary returned by the zone phase agent."""

    zone_names: list[str] = Field(description="Names of all zones created")
    summary: str = Field(description="One-line summary of the zone creation result")


def zone_agent(state: AgentState) -> AgentStateUpdate:
    local = state.config_state.model_copy(deep=True)
    tools = make_zone_tools(local)
    collector = TraceCollector(phase="zone")

    agent = build_agent(
        tools=tools,
        system_prompt=ZONE_SYSTEM_PROMPT,
        response_format=ZoneResponse,
        middleware=[trace_middleware(collector)],
    )

    specs = state.intake_output.zone_specs if state.intake_output else state.user_input
    result = agent.invoke({"messages": [HumanMessage(content=specs)]})

    response: ZoneResponse | None = result.get("structured_response")
    summary = response.summary if response else "zone done"

    record_phase_trace("zone", collector.export())

    return AgentStateUpdate(
        config_state=local,
        messages=[AIMessage(content=f"[zone] {summary}")],
    )
