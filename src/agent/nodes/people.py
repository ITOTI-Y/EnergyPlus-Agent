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
from src.agent.tools import make_people_tools
from src.agent.trace import TraceCollector, record_phase_trace, trace_middleware

PEOPLE_SYSTEM_PROMPT = """You are an occupancy-load expert for EnergyPlus.
Create the People objects of the specified zones with create_people,
one call per group of zones with the same values.

Workflow:
1. FIRST call `list_zones` to see the exact zone names.
2. FIRST call `list_schedules` to see the exact Schedule:Compact names
   (you need occupancy fraction + activity level schedules).
3. Call `create_people` once per group of zones with the same schedules
   and values, listing the zones in `zone_names`. Its reply names only
   zones that failed; the rest got their object. Do not list afterwards.

Rules:
- Zone names, `number_of_people_schedule_name`, `activity_level_schedule_name`
  MUST all appear verbatim in the list_zones / list_schedules results.
- If a needed zone or schedule is missing, do NOT invent a name and do NOT
  list again: give your final answer at once, with it in `missing_inputs`.
- Objects are named '{zone}_People'; `name_suffix` changes the suffix.
- Choose number_of_people_calculation_method based on input:
    * 'People' -> supply number_of_people (absolute count)
    * 'People/Area' -> supply people_per_floor_area (people/m^2)
    * 'Area/Person' -> supply floor_area_per_person (m^2/person)
- Typical office density: 10 m^2/person (People/Area ~ 0.1).
- fraction_radiant defaults to 0.3 for seated activity.
"""


class PeopleResponse(PhaseReport):
    """Structured summary returned by the people phase agent."""

    people_names: list[str] = Field(description="Names of all People objects created")
    summary: str = Field(description="One-line summary of the people creation result")


def people_agent(state: AgentState) -> AgentStateUpdate:
    if skipped(state, "people"):
        return AgentStateUpdate()
    local = state.config_state.model_copy(deep=True)
    tools = make_people_tools(local)
    collector = TraceCollector(phase="people")

    agent = build_agent(
        tools=tools,
        system_prompt=PEOPLE_SYSTEM_PROMPT,
        response_format=PeopleResponse,
        middleware=[trace_middleware(collector)],
    )

    specs = (
        state.intake_output.people_specs if state.intake_output else state.user_input
    )
    result = invoke_with_self_repair(
        agent, local, with_feedback(specs, state, "people"), phase="people"
    )

    response: PeopleResponse | None = result.get("structured_response")
    summary = response.summary if response else last_message_text(result)

    record_phase_trace("people", collector.export())
    return AgentStateUpdate(
        config_state=local,
        build_issues=missing_input_issues("people", response),
        messages=[AIMessage(content=f"[people] {summary}")],
    )
