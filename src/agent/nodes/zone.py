from idfpy.models.thermal_zones import Zone
from langchain_core.messages import AIMessage

from src.agent.nodes._share import skipped
from src.agent.state import AgentState, AgentStateUpdate
from src.modeling import objects


def zone_node(state: AgentState) -> AgentStateUpdate:
    """Create the zones the intake output lists; their geometry comes later.

    Zone names are unique by the intake schema, and vertices are in world
    coordinates, so every zone keeps the default origin.
    """
    if skipped(state, "zone") or state.intake_output is None:
        return AgentStateUpdate()
    local = state.config_state.model_copy(deep=True)
    zones = state.intake_output.zones
    for zone in zones:
        objects.create(local.idf, Zone(name=zone.name, multiplier=zone.multiplier))
    return AgentStateUpdate(
        config_state=local,
        messages=[AIMessage(content=f"[zone] Created {len(zones)} zones.")],
    )
