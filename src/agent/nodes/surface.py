from langchain_core.messages import AIMessage, HumanMessage

from src.agent.llm import create_llm
from src.agent.react import ReactState, build_react_agent
from src.agent.state import AgentState, AgentStateUpdate
from src.agent.tools import make_surface_tools
from src.agent.trace import TRACE_STORE, TraceCollector

SURFACE_SYSTEM_PROMPT = """You are a building geometry expert for EnergyPlus.
Given surface specifications, create all BuildingSurface:Detailed objects
(walls, floors, roofs, ceilings) with 3D vertex polygons.

Vertices MUST be a list of dicts, each with explicit X / Y / Z keys (not
a bare [x, y, z] list). Meters, in the global coordinate system. Example
shape for a 5m x 2m south wall at y=0 (ground to 2m tall):

    [
      {"X": 0.0, "Y": 0.0, "Z": 0.0},
      {"X": 5.0, "Y": 0.0, "Z": 0.0},
      {"X": 5.0, "Y": 0.0, "Z": 2.0},
      {"X": 0.0, "Y": 0.0, "Z": 2.0}
    ]

Rules:
- >= 3 vertices per surface; four-vertex rectangles are most common.
- Order counter-clockwise when viewed from OUTSIDE the zone.
- No two vertices may coincide (tolerance 1e-10 m).
- Assign each surface to an existing zone (zone_name) and construction.
- outside_boundary_condition:
    * Walls/roofs facing outdoors: 'Outdoors',
      sun_exposure='SunExposed', wind_exposure='WindExposed'
    * Floors on ground slab: 'Ground',
      sun_exposure='NoSun', wind_exposure='NoWind'
    * Internal partitions between zones: 'Surface',
      sun_exposure='NoSun', wind_exposure='NoWind',
      and outside_boundary_condition_object must reference the matching
      partner surface in the other zone
    * Adiabatic walls (e.g., between identical thermal zones): 'Adiabatic'
- surface_type is one of Wall, Floor, Roof, Ceiling (case-insensitive).
- Name convention: '{zone}_{direction}_{type}', e.g.,
  'F1_Office_North_Wall', 'F1_Office_Floor', 'F1_Office_Roof'.
- Call list_surfaces once at the end.
"""


def surface_agent(state: AgentState) -> AgentStateUpdate:
    local = state.config_state.model_copy(deep=True)
    tools = make_surface_tools(local)
    collector = TraceCollector(phase="surface")

    agent = build_react_agent(
        llm=create_llm(),
        tools=tools,
        system_prompt=SURFACE_SYSTEM_PROMPT,
        trace_collector=collector,
    )

    specs = (
        state.intake_output.surface_specs if state.intake_output else state.user_input
    )
    result = agent.invoke(ReactState(messages=[HumanMessage(content=specs)]))

    final = [
        m for m in result["messages"] if isinstance(m, AIMessage) and not m.tool_calls
    ]
    summary = final[-1].content if final else "surface done"

    TRACE_STORE.setdefault("surface", []).extend(collector.export())
    return AgentStateUpdate(
        config_state=local,
        messages=[AIMessage(content=f"[surface] {summary}")],
    )
