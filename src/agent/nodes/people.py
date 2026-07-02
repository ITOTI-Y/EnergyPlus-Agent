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
from src.agent.tools import make_people_tools
from src.agent.trace import TraceCollector, record_phase_trace

PEOPLE_SYSTEM_PROMPT = """You are an occupancy-load expert for EnergyPlus.
For each specified zone, create a People object via create_people.

Workflow:
1. FIRST call `list_zones` to see the exact zone names.
2. FIRST call `list_schedules` to see the exact Schedule:Compact names
   (you need occupancy fraction + activity level schedules).
3. Create a People object per zone via `create_people`.
4. Call `list_people` once at the end to confirm.

Rules:
- `zone_name`, `number_of_people_schedule_name`, `activity_level_schedule_name`
  MUST all appear verbatim in the list_zones / list_schedules results.
- If a needed zone or schedule is missing, STOP and report; do NOT invent names.
- name convention: '{zone}_People'.
- Choose number_of_people_calculation_method based on input:
    * 'People' -> supply number_of_people (absolute count)
    * 'People/Area' -> supply people_per_floor_area (people/m^2)
    * 'Area/Person' -> supply floor_area_per_person (m^2/person)
- Typical office density: 10 m^2/person (People/Area ~ 0.1).
- fraction_radiant defaults to 0.3 for seated activity.
"""


# Legal back-hop targets for people: a missing zone hops to zone; a missing
# schedule hops to schedule. Declared on the return type so LangGraph accepts
# Command(goto=...). (Mirrors surface.py / construction.py.)
_PeopleRoute = Literal["zone", "schedule"]


def people_agent(state: AgentState) -> Command[_PeopleRoute] | AgentStateUpdate:
    local = clone_for_phase(state)
    tools = make_people_tools(local)
    collector = TraceCollector(phase="people")

    agent = build_react_agent(
        llm=create_llm(),
        tools=tools,
        system_prompt=PEOPLE_SYSTEM_PROMPT,
        trace_collector=collector,
    )

    specs = (
        state.intake_output.people_specs if state.intake_output else state.user_input
    )
    result = invoke_with_self_repair(
        agent,
        local,
        specs,
        phase="people",
        is_revision=state.is_revision,
        validation_errors=state.validation_errors,
    )

    record_phase_trace("people", collector.export())

    # Back-hop: a missing zone / schedule (detected by invoke_with_self_repair)
    # routes to the owning earlier phase so it can create the object, then
    # normal graph edges carry flow back forward to people.
    hop = maybe_backhop(result, state, local, "people")
    if hop is not None:
        return hop

    final = [
        m for m in result["messages"] if isinstance(m, AIMessage) and not m.tool_calls
    ]
    summary = final[-1].content if final else "people done"

    return AgentStateUpdate(
        config_state=local,
        upstream_request={},  # consume any inbound back-hop request
        messages=[AIMessage(content=f"[people] {summary}")],
    )
