import pickle
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agent.nodes import (
    construction_agent,
    cross_ref_complete_node,
    cross_ref_foundations_node,
    equipment_agent,
    fenestration_agent,
    hvac_agent,
    intake_node,
    lights_agent,
    material_agent,
    people_agent,
    schedule_agent,
    simulate_node,
    surface_agent,
    validate_node,
    zone_agent,
)
from src.agent.state import AgentState, SimContext


class _PickleSerde:
    """Checkpoint serializer that round-trips via pickle.

    LangGraph's default `JsonPlusSerializer` uses msgpack for Pydantic
    models, which serializes declared fields only; `ConfigState` keeps its
    idfpy model in a private attribute, which would be lost.

    Pickle preserves the full Python object graph. Acceptable because
    `InMemorySaver` is in-process only (no cross-version / cross-host
    compatibility concerns).
    """

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        return ("pickle", pickle.dumps(obj))

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        return pickle.loads(data[1])


def _cross_ref_router(state: AgentState) -> str:
    """Route after cross_ref_foundations: continue to construction, or short-circuit to validate on error."""
    return "validate" if state.validation_errors else "construction"


def build_graph() -> CompiledStateGraph[AgentState, SimContext, AgentState, AgentState]:
    """Build and compile the multi-phase agent graph.

    Topology:
        intake
          -> phase 1 [zone, material, schedule] (parallel)
          -> cross_ref_foundations -> construction -> surface -> fenestration
          -> phase 3 [hvac, people, lights, equipment] (parallel)
          -> cross_ref_complete -> validate
          -> (approved) simulate -> END
          -> (rejected) intake (loop)
    """
    builder = StateGraph(AgentState, context_schema=SimContext)

    builder.add_node("intake", intake_node)

    builder.add_node("zone", zone_agent)
    builder.add_node("material", material_agent)
    builder.add_node("schedule", schedule_agent)
    builder.add_node("cross_ref_foundations", cross_ref_foundations_node)

    builder.add_node("construction", construction_agent)
    builder.add_node("surface", surface_agent)
    builder.add_node("fenestration", fenestration_agent)

    builder.add_node("hvac", hvac_agent)
    builder.add_node("people", people_agent)
    builder.add_node("lights", lights_agent)
    builder.add_node("equipment", equipment_agent)
    builder.add_node("cross_ref_complete", cross_ref_complete_node)

    builder.add_node("validate", validate_node)
    builder.add_node("simulate", simulate_node)

    builder.add_edge(START, "intake")

    builder.add_edge("intake", "zone")
    builder.add_edge("intake", "material")
    builder.add_edge("intake", "schedule")
    builder.add_edge(["zone", "material", "schedule"], "cross_ref_foundations")

    builder.add_conditional_edges(
        "cross_ref_foundations",
        _cross_ref_router,
        ["construction", "validate"],
    )

    builder.add_edge("construction", "surface")
    builder.add_edge("surface", "fenestration")

    builder.add_edge("fenestration", "hvac")
    builder.add_edge("fenestration", "people")
    builder.add_edge("fenestration", "lights")
    builder.add_edge("fenestration", "equipment")

    builder.add_edge(["hvac", "people", "lights", "equipment"], "cross_ref_complete")

    builder.add_edge("cross_ref_complete", "validate")

    # validate routes will dynamically route via Command -> simulate or intake
    builder.add_edge("simulate", END)

    return builder.compile(checkpointer=InMemorySaver(serde=_PickleSerde()))
