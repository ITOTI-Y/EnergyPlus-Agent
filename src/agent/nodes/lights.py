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
from src.agent.tools import make_lights_tools
from src.agent.trace import TraceCollector, record_phase_trace, trace_middleware

LIGHTS_SYSTEM_PROMPT = """You are a lighting-load expert for EnergyPlus.
Create the Lights objects of the specified zones with create_lights,
one call per group of zones with the same values.

Workflow:
1. FIRST call `list_zones` to see the exact zone names.
2. FIRST call `list_schedules` to see the exact Schedule:Compact names
   (you need a lighting fraction schedule).
3. Call `create_lights` once per group of zones with the same schedule
   and values, listing the zones in `zone_names`. Its reply names only
   zones that failed; the rest got their object. Do not list afterwards.

Rules:
- Zone names and `schedule_name` MUST appear verbatim in the list_zones /
  list_schedules results.
- If a needed zone or schedule is missing, do NOT invent a name and do NOT
  list again: give your final answer at once, with it in `missing_inputs`.
- Objects are named '{zone}_Lights'; `name_suffix` changes the suffix,
  e.g. for a second Lights object in a zone.
- design_level_calculation_method:
    * 'LightingLevel' -> supply lighting_level (W, absolute)
    * 'Watts/Area' -> supply watts_per_floor_area (W/m^2)
    * 'Watts/Person' -> supply watts_per_person (W/person)
- Typical office LPD: 8-12 W/m^2 (Watts/Area). Use 10 when unspecified.
- fraction_radiant ~ 0.7 for recessed fluorescent/LED, 0.42 for pendant.
- fraction_visible ~ 0.18 for LED.
"""


class LightsResponse(PhaseReport):
    """Structured summary returned by the lights phase agent."""

    lights_names: list[str] = Field(description="Names of all Lights objects created")
    summary: str = Field(description="One-line summary of the lights creation result")


def lights_agent(state: AgentState) -> AgentStateUpdate:
    if skipped(state, "lights"):
        return AgentStateUpdate()
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
    result = invoke_with_self_repair(
        agent, local, with_feedback(specs, state, "lights"), phase="lights"
    )

    response: LightsResponse | None = result.get("structured_response")
    summary = response.summary if response else last_message_text(result)

    record_phase_trace("lights", collector.export())
    return AgentStateUpdate(
        config_state=local,
        build_issues=missing_input_issues("lights", response),
        messages=[AIMessage(content=f"[lights] {summary}")],
    )
