import contextlib
import os
import pickle
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agent._share import ensure_schema_initialized
from src.agent.nodes import (
    analyze_node,
    construction_agent,
    cross_ref_complete_node,
    cross_ref_foundations_node,
    fenestration_agent,
    hvac_agent,
    intake_node,
    lights_agent,
    material_agent,
    people_agent,
    revise_node,
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
    models, which drops nested subclass information (e.g. restoring a
    `StandardMaterialSchema` instance as `dict` instead of the subclass).
    That breaks downstream code like `ConfigState.validate_references()`
    which accesses `material.name` on each entry.

    Pickle preserves the full Python object graph — nested Pydantic
    subclass instances round-trip identically. Acceptable because
    `InMemorySaver` is in-process only (no cross-version / cross-host
    compatibility concerns).

    Fallback: when a ConfigState loaded from an IDF file enters the graph,
    its idfpy IDF object contains weakref internals that cannot be pickled.
    On pickle failure we serialize the ConfigState to IDF text + a plain
    dict of its Pydantic fields, then rebuild on load.
    """

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        try:
            return ("pickle", pickle.dumps(obj))
        except (TypeError, pickle.PicklingError):
            return ("pickle", pickle.dumps(_make_picklable(obj)))

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        result = pickle.loads(data[1])
        return _restore_surrogates(result)


def _make_picklable(obj: Any) -> Any:
    """Replace any ConfigState in *obj* with a text-based surrogate.

    Recursively walks dicts/lists/tuples to find ConfigState instances
    (they may be nested inside AgentState or graph state dicts).
    """
    if _is_config_state(obj):
        return _ConfigStateSurrogate.from_instance(obj)
    if isinstance(obj, dict):
        return {k: _make_picklable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = type(obj)
        return t(_make_picklable(v) for v in obj)
    return obj


def _is_config_state(obj: Any) -> bool:
    cls = type(obj)
    return cls.__name__ == "ConfigState" and hasattr(obj, "_idf")


def _restore_surrogates(obj: Any) -> Any:
    """Walk *obj* and restore any _ConfigStateSurrogate back to ConfigState."""
    if isinstance(obj, _ConfigStateSurrogate):
        return obj.restore()
    if isinstance(obj, dict):
        return {k: _restore_surrogates(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = type(obj)
        return t(_restore_surrogates(v) for v in obj)
    return obj


class _ConfigStateSurrogate:
    """Picklable stand-in for ConfigState: stores IDF text + Pydantic fields."""

    def __init__(self, idf_text: str, fields: dict):
        self.idf_text = idf_text
        self.fields = fields

    @classmethod
    def from_instance(cls, cs: Any) -> "_ConfigStateSurrogate":
        import tempfile

        idf = cs._idf
        idf_text = ""
        if idf is not None:
            with tempfile.NamedTemporaryFile(suffix=".idf", delete=False) as tf:
                idf.save(Path(tf.name))
                idf_text = Path(tf.name).read_text(encoding="utf-8")
                os.unlink(tf.name)
        # Capture Pydantic fields (public ones, not PrivateAttr)
        fields = {
            k: v
            for k, v in cs.__dict__.items()
            if not k.startswith("_") and not k.startswith("__")
        }
        return cls(idf_text=idf_text, fields=fields)

    def restore(self) -> Any:
        """Rebuild a ConfigState from the stored IDF text + fields."""
        from src.mcp.state import ConfigState

        cs = ConfigState()
        if self.idf_text:
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".idf", delete=False
            ) as tf:
                tf.write(self.idf_text)
                cs.load_idf(Path(tf.name))
                os.unlink(tf.name)
        # Restore public Pydantic fields
        for k, v in self.fields.items():
            with contextlib.suppress(Exception):
                setattr(cs, k, v)
        return cs


def _cross_ref_router(state: AgentState) -> str:
    """Route after cross_ref_foundations: continue to construction, or short-circuit to validate on error."""
    return "validate" if state.validation_errors else "construction"


def _entry_router(state: AgentState) -> str:
    """START router: first turn → intake (build from scratch); revision turn → revise."""
    return "revise" if state.is_revision else "intake"


def build_graph() -> CompiledStateGraph[AgentState, SimContext, AgentState, AgentState]:
    """Build and compile the multi-phase agent graph.

    Topology:
        intake
          -> phase 1 [zone, material, schedule] (parallel)
          -> cross_ref_foundations -> construction -> surface -> fenestration
          -> phase 3 [hvac, people, lights] (parallel)
          -> cross_ref_complete -> validate
          -> (approved) simulate -> analyze -> END
          -> (rejected) intake (loop)
    """
    ensure_schema_initialized()

    builder = StateGraph(AgentState, context_schema=SimContext)

    builder.add_node("intake", intake_node)
    builder.add_node("revise", revise_node)

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
    builder.add_node("cross_ref_complete", cross_ref_complete_node)

    builder.add_node("validate", validate_node)
    builder.add_node("simulate", simulate_node)
    builder.add_node("analyze", analyze_node)

    # START: first turn → intake, revision turn → revise
    builder.add_conditional_edges(START, _entry_router, ["intake", "revise"])

    # Both intake and revise fan out to the same phase-1 nodes
    builder.add_edge("intake", "zone")
    builder.add_edge("intake", "material")
    builder.add_edge("intake", "schedule")
    builder.add_edge("revise", "zone")
    builder.add_edge("revise", "material")
    builder.add_edge("revise", "schedule")
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

    builder.add_edge(["hvac", "people", "lights"], "cross_ref_complete")

    builder.add_edge("cross_ref_complete", "validate")

    # validate routes will dynamically route via Command -> simulate or intake
    builder.add_edge("simulate", "analyze")
    builder.add_edge("analyze", END)

    return builder.compile(checkpointer=InMemorySaver(serde=_PickleSerde()))
