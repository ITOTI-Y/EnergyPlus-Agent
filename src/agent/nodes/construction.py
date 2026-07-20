from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.types import Command

from src.agent.llm import create_llm
from src.agent.nodes._share import (
    clone_for_phase,
    invoke_with_self_repair,
    maybe_backhop,
)
from src.agent.react import build_react_agent
from src.agent.state import AgentState, AgentStateUpdate
from src.agent.tools import make_construction_tools
from src.agent.tools.rag_tools import _get_rag
from src.agent.trace import TraceCollector, record_phase_trace

# Legal back-hop target for construction: a missing material layer hops
# to the material phase. Declared on the return type so LangGraph accepts
# Command(goto=...).
_ConstructionRoute = Literal["material"]

CONSTRUCTION_SYSTEM_PROMPT = """You are a construction-assembly expert for EnergyPlus.
Given construction specifications, create all required Construction objects.

Workflow:
1. FIRST call `list_materials` to discover which materials are already
   defined and their full properties (thickness, conductivity, U-Factor
   for glazing, etc.). DO NOT skip this step — the materials phase uses
   names that may differ from what the intake spec suggested.
2. Pick the correct layer composition for each construction using the
   material names returned by list_materials, verbatim.
3. Call `create_construction` for each construction in the spec.
4. Call `list_constructions` once at the end to confirm.

Rules:
- Layer names passed to `create_construction` MUST appear verbatim in
  the list_materials result (exact case, underscores, dashes, numbers).
- If a needed material is missing from list_materials, STOP and report
  the gap; do NOT invent names or call create with a broken reference.
- Each Construction is an ordered list of layers from OUTSIDE to INSIDE.
- Use separate constructions per surface type when thermal properties differ
  (e.g., 'ExtWall_Office', 'IntWall_Office', 'Roof_Office', 'Floor_Office',
  'Window_Office').
- For fenestration, the construction's only layer is the glazing material.
- FENESTRATION GLAZING RULE — DEFAULT TO SINGLE-LAYER SimpleGlazingSystem:
  1. PREFERRED: a SINGLE-LAYER whole window whose only layer is a
     WindowMaterial:SimpleGlazingSystem (created via create_glazing_material).
     Use this for ALL windows unless the material phase has explicitly
     created per-pane WindowMaterial:Glazing layers. Even for "double pane"
     or "triple pane" specs, prefer this — the material phase should have
     supplied an equivalent whole-window U-factor/SHGC via
     create_glazing_material. The construction has exactly ONE layer.
  2. MULTI-LAYER real assembly (ONLY if per-pane layers already exist):
     per-pane WindowMaterial:Glazing layers (created via
     create_glazing_layer_material) interleaved with Material:AirGap, e.g.
     layers=['Clear_Glass_3mm','Air_Gap_13mm','Clear_Glass_3mm']. Use this
     shape ONLY when list_materials shows WindowMaterial:Glazing objects;
     otherwise fall back to single-layer SimpleGlazingSystem.
  NEVER mix WindowMaterial:SimpleGlazingSystem with other layers (gas gaps,
  extra panes). SimpleGlazingSystem is a whole-window equivalent
  (U/SHGC/VT only) — combining it with other layers gives EnergyPlus no
  per-pane data and aborts with a Fatal convergence error.
  DECISION RULE: if list_materials contains a WindowMaterial:SimpleGlazingSystem,
  use it as the sole layer of the window construction. Only assemble
  per-pane layers if the spec explicitly provided per-pane optical data AND
  the material phase created WindowMaterial:Glazing objects.
- INTERIOR doors/windows/glass-doors between two zones (hosted on a wall that
  separates zones, i.e. the wall's outside boundary condition is 'Surface'):
  use create_airboundary_construction (a Construction:AirBoundary, no layers).
  This models the door/window as an OPEN passage between zones and avoids the
  EnergyPlus error "invalid blank Outside Boundary Condition Object" that a
  regular layered construction would trigger there. Name it e.g.
  'Interior_Door_Open'. EXTERIOR doors/windows still use a normal layered
  Construction (glazing for windows, opaque for doors).

Reference database:
- Call search_energyplus_reference to look up standard layer sequences for
  named construction types (e.g. 'ASHRAE 90.1 exterior wall office').
  Match returned layer names to existing materials via list_materials;
  if a reference layer has no local equivalent, use the nearest match by
  thermal properties.
"""


def construction_agent(
    state: AgentState,
) -> Command[_ConstructionRoute] | AgentStateUpdate:
    local = clone_for_phase(state)
    tools = make_construction_tools(local, rag=_get_rag())
    collector = TraceCollector(phase="construction")

    agent = build_react_agent(
        llm=create_llm(),
        tools=tools,
        system_prompt=CONSTRUCTION_SYSTEM_PROMPT,
        trace_collector=collector,
    )

    specs = (
        state.intake_output.construction_specs
        if state.intake_output
        else state.user_input
    )
    # If reached via a back-hop from a downstream phase (surface/fenestration
    # needed a construction that did not exist), append the request.
    upstream = state.upstream_request
    if upstream and upstream.get("target") == "construction":
        specs = f"{specs}\n\n{upstream['specs']}"

    result = invoke_with_self_repair(
        agent,
        local,
        specs,
        phase="construction",
        is_revision=state.is_revision,
        validation_errors=state.validation_errors,
    )

    record_phase_trace("construction", collector.export())

    # Back-hop: a missing material layer routes to the material phase.
    hop = maybe_backhop(result, state, local, "construction")
    if hop is not None:
        return hop

    final = [
        m for m in result["messages"] if isinstance(m, AIMessage) and not m.tool_calls
    ]
    summary = final[-1].content if final else "construction done"
    return AgentStateUpdate(
        config_state=local,
        upstream_request={},  # consume the back-hop request
        messages=[AIMessage(content=f"[construction] {summary}")],
    )
