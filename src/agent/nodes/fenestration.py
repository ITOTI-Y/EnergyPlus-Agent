from langchain_core.messages import AIMessage, HumanMessage

from src.agent.llm import create_llm
from src.agent.react import ReactState, build_react_agent
from src.agent.state import AgentState, AgentStateUpdate
from src.agent.tools import make_fenestration_tools
from src.agent.trace import TRACE_STORE, TraceCollector

FENESTRATION_SYSTEM_PROMPT = """You are a window/door geometry expert for EnergyPlus.
Given fenestration specifications, create FenestrationSurface:Detailed
objects (windows, doors, skylights) that lie on existing parent surfaces.

Vertices MUST be a list of dicts with explicit X / Y / Z keys (not a
bare [x, y, z] list). Example: a 1.5m x 1.2m window centered on a south
wall that spans x=0..5 at y=0, window sill at 0.8m:

    [
      {"X": 1.75, "Y": 0.0, "Z": 0.8},
      {"X": 3.25, "Y": 0.0, "Z": 0.8},
      {"X": 3.25, "Y": 0.0, "Z": 2.0},
      {"X": 1.75, "Y": 0.0, "Z": 2.0}
    ]

Rules:
- building_surface_name must reference an existing Surface.
- construction_name is a Glazing construction (windows/skylights) or a
  door construction; verify the construction exists.
- >= 3 vertices, counter-clockwise from OUTSIDE, and MUST lie on the
  parent surface's plane (coplanar — share one coordinate for walls).
- surface_type is Window, Door, or GlassDoor.
- Typical window-to-wall ratio: 0.3-0.4 on facade walls; derive vertex
  coordinates from the parent wall's corners and the WWR.
- Naming: '{parent_surface}_Window' or '{zone}_{direction}_Window_{index}'.
- Call list_fenestrations once at the end.
"""


def fenestration_agent(state: AgentState) -> AgentStateUpdate:
    local = state.config_state.model_copy(deep=True)
    tools = make_fenestration_tools(local)
    collector = TraceCollector(phase="fenestration")

    agent = build_react_agent(
        llm=create_llm(),
        tools=tools,
        system_prompt=FENESTRATION_SYSTEM_PROMPT,
        trace_collector=collector,
    )

    specs = (
        state.intake_output.fenestration_specs
        if state.intake_output
        else state.user_input
    )
    result = agent.invoke(ReactState(messages=[HumanMessage(content=specs)]))

    final = [
        m for m in result["messages"] if isinstance(m, AIMessage) and not m.tool_calls
    ]
    summary = final[-1].content if final else "fenestration done"

    TRACE_STORE.setdefault("fenestration", []).extend(collector.export())
    return AgentStateUpdate(
        config_state=local,
        messages=[AIMessage(content=f"[fenestration] {summary}")],
    )
