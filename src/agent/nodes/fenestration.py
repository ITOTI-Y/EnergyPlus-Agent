from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from src.agent.llm import build_agent
from src.agent.nodes._share import (
    invoke_with_self_repair,
    last_message_text,
    skipped,
    with_feedback,
)
from src.agent.state import AgentState, AgentStateUpdate
from src.agent.tools import make_fenestration_tools
from src.agent.trace import TraceCollector, record_phase_trace, trace_middleware

FENESTRATION_SYSTEM_PROMPT = """You are a window/door geometry expert for EnergyPlus.
Given fenestration specifications, create FenestrationSurface:Detailed
objects (windows, doors) on existing parent surfaces.

Workflow:
1. Call `list_constructions`; each entry has a `kind`.
2. Windows given by window-to-wall ratio (the usual case): call
   `create_windows_by_ratio` once per ratio, with the facings and zones it
   applies to. It sizes and places a strip window on every matching
   exterior wall; walls it reports in `not_created` (too small for the
   ratio) get no window, which is acceptable.
3. Only for openings with given sizes or positions (doors, a specific
   window): call `list_surfaces` FILTERED to the zone and type you need
   (never unfiltered: large buildings have hundreds of surfaces), then
   `create_fenestration` with corners on that surface.
4. Call `list_fenestrations` once at the end to confirm.

Rules:
- Construction names MUST appear verbatim in the list_constructions
  result; if one is missing, STOP and report; do NOT invent names.
- Window and GlassDoor need a construction of kind `window`; Door needs
  kind `opaque`.
- For create_fenestration: 3 or 4 vertices as dicts with X / Y / Z keys,
  on the parent surface's plane and inside its outline; the order is
  corrected automatically. On a wall between two zones, create the opening
  once; the matching opening in the adjacent zone is added automatically.
- Use the ratios, sizes and names the specification gives. Without a
  ratio, use 0.3-0.4 on facade walls.
"""


class FenestrationResponse(BaseModel):
    """Structured summary returned by the fenestration phase agent."""

    fenestration_names: list[str] = Field(
        description="Names of all fenestration surfaces created"
    )
    summary: str = Field(
        description="One-line summary of the fenestration creation result"
    )


def fenestration_agent(state: AgentState) -> AgentStateUpdate:
    if skipped(state, "fenestration"):
        return AgentStateUpdate()
    local = state.config_state.model_copy(deep=True)
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
        agent, local, with_feedback(specs, state, "fenestration"), phase="fenestration"
    )

    response: FenestrationResponse | None = result.get("structured_response")
    summary = response.summary if response else last_message_text(result)

    record_phase_trace("fenestration", collector.export())
    return AgentStateUpdate(
        config_state=local,
        messages=[AIMessage(content=f"[fenestration] {summary}")],
    )
