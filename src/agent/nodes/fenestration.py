from langchain_core.messages import AIMessage

from src.agent.llm import create_llm
from src.agent.nodes._share import invoke_with_self_repair
from src.agent.react import build_react_agent
from src.agent.state import AgentState, AgentStateUpdate
from src.agent.tools import make_fenestration_tools
from src.agent.trace import TraceCollector, record_phase_trace

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

Workflow:
1. FIRST call `list_surfaces` to see parent surface names AND their
   vertex geometry — you need the parent surface's plane to place the
   fenestration's coplanar vertices correctly.
2. THEN call `list_constructions` to find glazing/door construction names.
3. Create each fenestration via `create_fenestration`.
4. Call `list_fenestrations` once at the end to confirm.

Rules:
- `building_surface_name` and `construction_name` MUST appear verbatim
  in the list_surfaces / list_constructions results.
- If a needed surface or construction is missing after list, STOP and
  report; do NOT invent names.
- construction_name should be a Glazing construction for windows/skylights.
- >= 3 vertices, counter-clockwise from OUTSIDE, and MUST lie on the
  parent surface's plane (coplanar — share one coordinate for walls).
- surface_type is Window, Door, or GlassDoor.
- Typical window-to-wall ratio: 0.3-0.4 on facade walls; derive vertex
  coordinates from the parent wall's corners and the WWR.
- Naming: '{parent_surface}_Window' or '{zone}_{direction}_Window_{index}'.
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
    result = invoke_with_self_repair(agent, local, specs, phase="fenestration", is_revision=state.is_revision)

    final = [
        m for m in result["messages"] if isinstance(m, AIMessage) and not m.tool_calls
    ]
    summary = final[-1].content if final else "fenestration done"

    record_phase_trace("fenestration", collector.export())
    return AgentStateUpdate(
        config_state=local,
        messages=[AIMessage(content=f"[fenestration] {summary}")],
    )
