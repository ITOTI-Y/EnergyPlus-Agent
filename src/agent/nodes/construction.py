from langchain_core.messages import AIMessage, HumanMessage

from src.agent.llm import create_llm
from src.agent.react import ReactState, build_react_agent
from src.agent.state import AgentState, AgentStateUpdate
from src.agent.tools import make_construction_tools
from src.agent.trace import TRACE_STORE, TraceCollector

CONSTRUCTION_SYSTEM_PROMPT = """You are a construction-assembly expert for EnergyPlus.
Given construction specifications, create all required Construction objects.

Rules:
- Each Construction is a named ordered list of material names (>= 1 layer),
  from OUTSIDE to INSIDE.
- All layer names must already exist in the materials list. If the spec
  mentions a material not yet created, halt and report the missing material
  rather than creating a construction with a broken reference.
- Use separate constructions per surface type when thermal properties differ
  (e.g., 'ExtWall_Office', 'IntWall_Office', 'Roof_Office', 'Floor_Office',
  'Window_Office').
- For fenestration, the construction's only layer is the glazing material.
- Call list_constructions once at the end.
"""


def construction_agent(state: AgentState) -> AgentStateUpdate:
    local = state.config_state.model_copy(deep=True)
    tools = make_construction_tools(local)
    collector = TraceCollector(phase="construction")

    agent = build_react_agent(
        llm=create_llm(),
        tools=tools,
        system_prompt=CONSTRUCTION_SYSTEM_PROMPT,
        trace_collector=collector,
    )

    specs = (
        state.intake_output.construction_specs
        if state.intake_output
        else state.user_input
    )
    result = agent.invoke(ReactState(messages=[HumanMessage(content=specs)]))

    final = [
        m for m in result["messages"] if isinstance(m, AIMessage) and not m.tool_calls
    ]
    summary = final[-1].content if final else "construction done"

    TRACE_STORE.setdefault("construction", []).extend(collector.export())
    return AgentStateUpdate(
        config_state=local,
        messages=[AIMessage(content=f"[construction] {summary}")],
    )
