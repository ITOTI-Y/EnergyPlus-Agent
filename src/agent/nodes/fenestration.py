from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.types import Command

from src.agent.llm import create_llm
from src.agent.nodes._share import clone_for_phase, invoke_with_self_repair, maybe_backhop
from src.agent.react import build_react_agent
from src.agent.state import AgentState, AgentStateUpdate
from src.agent.tools import make_fenestration_tools
from src.agent.trace import TraceCollector, record_phase_trace


# Legal back-hop targets for fenestration: a missing window construction
# hops to construction; a missing parent surface hops to surface. Declared
# as a Literal on the return type so LangGraph accepts Command(goto=...).
_FenestrationRoute = Literal["construction", "surface"]

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
- For Window / GlassDoor / TubularDaylight* the construction MUST be a
  glazing construction — one whose layers include a
  WindowMaterial:SimpleGlazingSystem. An opaque construction (only
  Material / Material:NoMass layers) is invalid for windows and will be
  rejected. If the only window construction available is opaque, STOP and
  report — do not create the fenestration against it; the material phase
  must first create a WindowMaterial and construction must rebuild it.
- surface_type is Window, Door, or GlassDoor.
- >= 3 vertices and MUST lie on the parent surface's plane (coplanar —
  share one coordinate for walls). Winding direction does not matter —
  the tool auto-aligns the window's outward normal to the parent wall's.
- Typical window-to-wall ratio: 0.3-0.4 on facade walls; derive vertex
  coordinates from the parent wall's corners and the WWR.
- Naming: '{parent_surface}_Window' or '{zone}_{direction}_Window_{index}'.
"""


def fenestration_agent(state: AgentState) -> Command[_FenestrationRoute] | AgentStateUpdate:
    local = clone_for_phase(state)
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
    result = invoke_with_self_repair(
        agent,
        local,
        specs,
        phase="fenestration",
        is_revision=state.is_revision,
        validation_errors=state.validation_errors,
    )

    record_phase_trace("fenestration", collector.export())

    # Back-hop: a missing window construction / parent surface routes to
    # the owning earlier phase so it can create the object, then normal
    # graph edges carry flow back forward to fenestration.
    hop = maybe_backhop(result, state, local, "fenestration")
    if hop is not None:
        return hop

    final = [
        m for m in result["messages"] if isinstance(m, AIMessage) and not m.tool_calls
    ]
    summary = final[-1].content if final else "fenestration done"
    return AgentStateUpdate(
        config_state=local,
        upstream_request=None,  # consume any stale back-hop request
        messages=[AIMessage(content=f"[fenestration] {summary}")],
    )
