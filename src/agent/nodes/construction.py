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
from src.agent.tools import make_construction_tools
from src.agent.trace import TraceCollector, record_phase_trace, trace_middleware

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
"""


class ConstructionResponse(BaseModel):
    """Structured summary returned by the construction phase agent."""

    construction_names: list[str] = Field(
        description="Names of all constructions created"
    )
    summary: str = Field(
        description="One-line summary of the construction creation result"
    )


def construction_agent(
    state: AgentState,
) -> Command[_ConstructionRoute] | AgentStateUpdate:
    local = clone_for_phase(state)
    tools = make_construction_tools(local)
    collector = TraceCollector(phase="construction")

    agent = build_agent(
        tools=tools,
        system_prompt=CONSTRUCTION_SYSTEM_PROMPT,
        response_format=ConstructionResponse,
        middleware=[trace_middleware(collector)],
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

    response: ConstructionResponse | None = result.get("structured_response")
    summary = response.summary if response else "construction done"

    return AgentStateUpdate(
        config_state=local,
        upstream_request=None,  # consume the back-hop request
        messages=[AIMessage(content=f"[construction] {summary}")],
    )
