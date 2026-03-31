from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.graph.state import CompiledStateGraph

from src.agent.state import Phase
from src.agent.tools import create_geometry_tools
from src.mcp.state import ConfigState

GEOMETRY_PROMPT = SystemMessage(
    content="""
    You are an EnergyPlus building geometry expert.
Task: Create thermal zones (Zone) and building surfaces (Surface) based on user descriptions.

Rules that must be followed:
- Order of vertices: UpperLeftCorner start, counterclockwise from the exterior viewpoint
- Window surfaces must inherit the direction of the normal vector from the parent wall
- For interior walls, use 'Surface' boundary condition, specifying the name of the paired surface.
- For complex planes, use 'Adiabatic' boundary conditions to avoid geometric conflicts.
- Complete Floor, Roof/Ceiling and all Wall surfaces must be created for each thermal zone.

A summary of the current configuration is provided. Create only the missing components.
"""
)

_PHASE_CONFIG: dict[str, tuple] = {
    "geometry": (create_geometry_tools, GEOMETRY_PROMPT),
}


def create_subagent(
    llm: BaseChatModel, config_state: ConfigState, phase: Phase
) -> CompiledStateGraph:
    factory_fn, system_prompt = _PHASE_CONFIG[phase]
    tools = factory_fn(config_state)

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        name=f"{phase}_agent",
    )
