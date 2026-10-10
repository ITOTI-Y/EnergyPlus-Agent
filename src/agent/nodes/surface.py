from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from src.agent.llm import build_agent
from src.agent.nodes._share import invoke_with_self_repair, last_message_text
from src.agent.state import AgentState, AgentStateUpdate
from src.agent.tools import make_surface_tools
from src.agent.trace import TraceCollector, record_phase_trace, trace_middleware

SURFACE_SYSTEM_PROMPT = """You are a building geometry expert for EnergyPlus.
Given surface specifications, create the BuildingSurface:Detailed objects
(walls, floors, roofs, ceilings) of every zone.

Workflow:
1. FIRST call `list_zones` for the exact zone names, THEN
   `list_constructions` for the opaque constructions you may use.
2. For every zone with vertical walls and a flat top, call
   `create_zone_geometry` once: the floor plan corners (X, Y in meters, in
   order around the zone), floor level, height and five constructions
   (exterior wall, roof, ground floor, interior wall, interior floor).
   It creates walls, floor and roof, and turns faces shared with other
   zones into interzone pairs automatically, in any order, also for zones
   of different height and storeys whose plans do not line up. Do NOT
   build such zones surface by surface.
3. Only for geometry an extrusion cannot express (sloped or pitched roofs,
   gable walls, sloped walls), call `create_surfaces` with all those
   surfaces at once; resend only entries reported as failed.
4. Call `list_surfaces` once at the end to confirm.

Rules:
- `zone_name` and construction names MUST appear verbatim in the
  list_zones / list_constructions results.
- If a needed zone or construction is missing, STOP and report; do NOT
  invent names.
- Floor level 0 is the ground; upper storeys start at the sum of the
  storey heights below them.
- Vertices for create_surfaces are dicts with X / Y / Z keys in meters,
  counter-clockwise seen from OUTSIDE the zone.
"""


class SurfaceResponse(BaseModel):
    """Structured summary returned by the surface phase agent."""

    surface_names: list[str] = Field(description="Names of all surfaces created")
    summary: str = Field(description="One-line summary of the surface creation result")


def surface_agent(state: AgentState) -> AgentStateUpdate:
    local = state.config_state.model_copy(deep=True)
    tools = make_surface_tools(local)
    collector = TraceCollector(phase="surface")

    agent = build_agent(
        tools=tools,
        system_prompt=SURFACE_SYSTEM_PROMPT,
        response_format=SurfaceResponse,
        middleware=[trace_middleware(collector)],
    )

    specs = (
        state.intake_output.surface_specs if state.intake_output else state.user_input
    )
    result = invoke_with_self_repair(agent, local, specs, phase="surface")

    response: SurfaceResponse | None = result.get("structured_response")
    summary = response.summary if response else last_message_text(result)

    record_phase_trace("surface", collector.export())
    return AgentStateUpdate(
        config_state=local,
        messages=[AIMessage(content=f"[surface] {summary}")],
    )
