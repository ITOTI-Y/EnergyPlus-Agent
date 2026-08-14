from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from src.agent.llm import build_agent
from src.agent.nodes._share import (
    clone_for_phase,
    invoke_with_self_repair,
    maybe_backhop,
)
from src.agent.state import AgentState, AgentStateUpdate
from src.agent.tools import make_fenestration_tools
from src.agent.trace import TraceCollector, record_phase_trace, trace_middleware

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
- construction_name should be a Glazing construction for windows/skylights.
- >= 3 vertices, counter-clockwise from OUTSIDE, and MUST lie on the
  parent surface's plane (coplanar — share one coordinate for walls).
- surface_type is Window, Door, or GlassDoor.
- Typical window-to-wall ratio: 0.3-0.4 on facade walls; derive vertex
  coordinates from the parent wall's corners and the WWR.
- Naming: '{parent_surface}_Window' or '{zone}_{direction}_Window_{index}'.
"""


class FenestrationResponse(BaseModel):
    """Structured summary returned by the fenestration phase agent."""

    fenestration_names: list[str] = Field(
        description="Names of all fenestration surfaces created"
    )
    summary: str = Field(
        description="One-line summary of the fenestration creation result"
    )


def fenestration_agent(
    state: AgentState,
) -> Command[_FenestrationRoute] | AgentStateUpdate:
    local = clone_for_phase(state)
    tools = make_fenestration_tools(local)
    collector = TraceCollector(phase="fenestration")

    agent = build_agent(
        tools=tools,
        system_prompt=FENESTRATION_SYSTEM_PROMPT,
        response_format=FenestrationResponse,
        middleware=[trace_middleware(collector)],
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

    response: FenestrationResponse | None = result.get("structured_response")
    summary = response.summary if response else "fenestration done"

    return AgentStateUpdate(
        config_state=local,
        upstream_request={},  # consume any stale back-hop request
        messages=[AIMessage(content=f"[fenestration] {summary}")],
    )
