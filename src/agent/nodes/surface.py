from typing import Final

from idfpy import IDF
from idfpy.models.constructions import Construction
from idfpy.models.thermal_zones import BuildingSurfaceDetailed, Zone
from langchain_core.messages import AIMessage
from pydantic import Field

from src.agent.llm import build_agent
from src.agent.nodes._share import (
    PhaseReport,
    invoke_with_self_repair,
    last_message_text,
    missing_input_issues,
    skipped,
    with_feedback,
)
from src.agent.state import AgentState, AgentStateUpdate, ZoneGeometrySchema
from src.agent.tools import make_surface_tools
from src.agent.trace import TraceCollector, record_phase_trace, trace_middleware
from src.modeling import geometry
from src.modeling.envelope import check_construction_fits
from src.modeling.errors import ModelingError
from src.modeling.geometry import ZoneConstructions
from src.modeling.validation import ModelIssue
from src.state.config_state import ConfigState

SURFACE_SYSTEM_PROMPT = """You are a building geometry expert for EnergyPlus.
Every zone already has vertical walls, a floor and a flat roof, extruded
from its floor plan; faces shared by two zones are interzone pairs. Your
task is only the geometry an extrusion cannot express: sloped or pitched
roofs and sloped walls.

Workflow:
1. Call `list_surfaces` FILTERED to the zones the sloped geometry
   changes (never unfiltered: large buildings have hundreds of surfaces),
   and `list_constructions` for the opaque constructions you may use.
2. `delete_surface` each flat face the sloped geometry replaces, e.g. the
   flat roof under a pitched roof.
3. Call `create_surfaces` once with all new surfaces (roof planes, gable
   walls); resend only entries reported as failed. Its reply is final: do
   not list the surfaces again.

Rules:
- `zone_name` and construction names MUST appear verbatim in the
  list results. If one is missing, do NOT invent a name and do NOT list
  again: give your final answer at once, with it in `missing_inputs`.
- A zone must stay closed: the new surfaces must cover exactly the
  opening left by the deleted faces.
- Vertices are dicts with X / Y / Z keys in meters, counter-clockwise
  seen from OUTSIDE the zone.
"""


class SurfaceResponse(PhaseReport):
    """Structured summary returned by the surface phase agent."""

    surface_names: list[str] = Field(description="Names of all surfaces created")
    summary: str = Field(description="One-line summary of the surface creation result")


_ROLES: Final = (
    ("exterior_wall_construction", "Wall"),
    ("roof_construction", "Roof"),
    ("ground_floor_construction", "Floor"),
    ("interior_wall_construction", "Wall"),
    ("interior_floor_construction", "Floor"),
)


def _construction_issues(idf: IDF, zone: ZoneGeometrySchema) -> list[ModelIssue]:
    """Constructions the zone names that are missing or of the wrong kind."""
    issues = []
    for field, surface_type in _ROLES:
        name: str = getattr(zone, field)
        role = field.removesuffix("_construction").replace("_", " ")
        try:
            if idf.get(Construction, name) is None:
                raise ValueError("it does not exist.")
            check_construction_fits(idf, name, surface_type)
        except ValueError as e:
            issues.append(
                ModelIssue(
                    Construction.idf_object_type(),
                    name,
                    None,
                    f"zone '{zone.name}' needs it as {role} construction, but {e}",
                )
            )
    return issues


def extrude_zones(idf: IDF, zones: list[ZoneGeometrySchema]) -> list[ModelIssue]:
    """Extrude every zone; returns why a zone could not be extruded.

    A missing or unsuitable construction is the construction phase's to
    fix; a plan that is invalid or overlaps another zone is the intake's.
    """
    issues = []
    for zone in zones:
        if found := _construction_issues(idf, zone):
            issues += found
            continue
        try:
            geometry.create_zone_geometry(
                idf,
                zone.name,
                zone.plan,
                zone.floor_z,
                zone.height,
                ZoneConstructions(
                    exterior_wall=zone.exterior_wall_construction,
                    roof=zone.roof_construction,
                    ground_floor=zone.ground_floor_construction,
                    interior_wall=zone.interior_wall_construction,
                    interior_floor=zone.interior_floor_construction,
                ),
            )
        except (ModelingError, ValueError) as e:
            issues.append(ModelIssue(Zone.idf_object_type(), zone.name, "plan", str(e)))
    return issues


def surface_agent(state: AgentState) -> AgentStateUpdate:
    if skipped(state, "surface") or state.intake_output is None:
        return AgentStateUpdate()
    local = state.config_state.model_copy(deep=True)
    issues = extrude_zones(local.idf, state.intake_output.zones)
    count = len(local.idf.all_of_type(BuildingSurfaceDetailed))
    summary = f"Extruded {len(state.intake_output.zones)} zones into {count} surfaces."
    specs = state.intake_output.surface_specs
    if specs.strip() and not issues:
        sloped, issues = _add_sloped_surfaces(
            local, with_feedback(specs, state, "surface")
        )
        summary += " " + sloped
    return AgentStateUpdate(
        config_state=local,
        build_issues=issues,
        messages=[AIMessage(content=f"[surface] {summary}")],
    )


def _add_sloped_surfaces(
    local: ConfigState, specs: str
) -> tuple[str, list[ModelIssue]]:
    collector = TraceCollector(phase="surface")
    agent = build_agent(
        tools=make_surface_tools(local),
        system_prompt=SURFACE_SYSTEM_PROMPT,
        response_format=SurfaceResponse,
        middleware=[trace_middleware(collector)],
    )
    result = invoke_with_self_repair(agent, local, specs, phase="surface")
    record_phase_trace("surface", collector.export())
    response: SurfaceResponse | None = result.get("structured_response")
    summary = response.summary if response else last_message_text(result)
    return summary, missing_input_issues("surface", response)
