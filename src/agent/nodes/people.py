from langchain_core.messages import AIMessage, HumanMessage

from src.agent.llm import create_llm
from src.agent.react import ReactState, build_react_agent
from src.agent.state import AgentState, AgentStateUpdate
from src.agent.tools import make_people_tools
from src.agent.trace import TRACE_STORE, TraceCollector

PEOPLE_SYSTEM_PROMPT = """You are an occupancy-load expert for EnergyPlus.
For each specified zone, create a People object via create_people.

Rules:
- name convention: '{zone}_People'.
- zone_name must reference an existing Zone.
- number_of_people_schedule_name and activity_level_schedule_name must
  reference existing Schedule:Compact objects (Fraction and Activity Level).
- Choose number_of_people_calculation_method based on input:
    * 'People' -> supply number_of_people (absolute count)
    * 'People/Area' -> supply people_per_floor_area (people/m^2)
    * 'Area/Person' -> supply floor_area_per_person (m^2/person)
- Typical office density: 10 m^2/person (People/Area ~ 0.1).
- fraction_radiant defaults to 0.3 for seated activity.
- Call list_people once at the end.
"""


def people_agent(state: AgentState) -> AgentStateUpdate:
    local = state.config_state.model_copy(deep=True)
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
    result = agent.invoke(ReactState(messages=[HumanMessage(content=specs)]))

    final = [
        m for m in result["messages"] if isinstance(m, AIMessage) and not m.tool_calls
    ]
    summary = final[-1].content if final else "people done"

    TRACE_STORE.setdefault("people", []).extend(collector.export())
    return AgentStateUpdate(
        config_state=local,
        messages=[AIMessage(content=f"[people] {summary}")],
    )
