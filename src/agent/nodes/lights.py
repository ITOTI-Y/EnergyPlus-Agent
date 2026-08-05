from typing import Literal

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from src.agent.llm import build_agent
from src.agent.nodes._share import invoke_with_self_repair
from src.agent.state import AgentState, AgentStateUpdate
from src.agent.tools import make_lights_tools
from src.agent.trace import TraceCollector, record_phase_trace, trace_middleware

LIGHTS_SYSTEM_PROMPT = """You are a lighting-load expert for EnergyPlus.
For each specified zone, create a Lights object via create_light.

Workflow:
1. FIRST call `list_zones` to see the exact zone names.
2. FIRST call `list_schedules` to see the exact Schedule:Compact names
   (you need a lighting fraction schedule).
3. Create a Lights object per zone via `create_light`.
4. Call `list_lights` once at the end to confirm.

Rules:
- `zone_name` and `schedule_name` MUST appear verbatim in the list_zones /
  list_schedules results.
- If a needed zone or schedule is missing, STOP and report; do NOT invent names.
- name convention: '{zone}_Lights'.
- design_level_calculation_method:
    * 'LightingLevel' -> supply lighting_level (W, absolute)
    * 'Watts/Area' -> supply watts_per_floor_area (W/m^2)
    * 'Watts/Person' -> supply watts_per_person (W/person)
- Typical office LPD: 8-12 W/m^2 (Watts/Area). Use 10 when unspecified.
- fraction_radiant ~ 0.7 for recessed fluorescent/LED, 0.42 for pendant.
- fraction_visible ~ 0.18 for LED.
"""


class LightsResponse(BaseModel):
    """Structured summary returned by the lights phase agent."""

    lights_names: list[str] = Field(description="Names of all Lights objects created")
    summary: str = Field(description="One-line summary of the lights creation result")


def lights_agent(state: AgentState) -> AgentStateUpdate:
    local = state.config_state.model_copy(deep=True)
    tools = make_lights_tools(local)
    collector = TraceCollector(phase="lights")

    agent = build_agent(
        tools=tools,
        system_prompt=LIGHTS_SYSTEM_PROMPT,
        response_format=LightsResponse,
        middleware=[trace_middleware(collector)],
    )

    specs = (
        state.intake_output.lights_specs if state.intake_output else state.user_input
    )
    result = invoke_with_self_repair(agent, local, specs, phase="lights")

    response: LightsResponse | None = result.get("structured_response")
    summary = response.summary if response else "lights done"

    record_phase_trace("lights", collector.export())
    return AgentStateUpdate(
        config_state=local,
        upstream_request={},  # consume any inbound back-hop request
        messages=[AIMessage(content=f"[lights] {summary}")],
    )
