from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Final

from idfpy import IDF
from idfpy.models.location import SiteLocation
from idfpy.models.simulation import Building
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator
from typing_extensions import TypedDict

from src.agent._share import DEFAULT_OUTPUT_DIR, MAX_GLOBAL_RETRIES
from src.agent.phases import Phase
from src.modeling.geometry import PlanPointSchema
from src.modeling.validation import ModelIssue
from src.state.config_state import ConfigState


class ZoneGeometrySchema(BaseModel):
    """One thermal zone as a prism: floor plan, floor level and height.

    Code creates the zone and extrudes its walls, floor and flat roof from
    this; faces shared with other zones become interzone pairs.
    """

    name: str = Field(description="Zone name, word characters and '_' only")
    plan: list[PlanPointSchema] = Field(
        description="Floor plan corners (X, Y in meters) in order around the zone"
    )
    floor_z: float = Field(description="Floor level in meters; 0 on the ground")
    height: float = Field(gt=0, description="Floor-to-ceiling height in meters")
    exterior_wall_construction: str
    roof_construction: str
    ground_floor_construction: str = Field(
        description="Floor on the ground, or above outdoor air"
    )
    interior_wall_construction: str = Field(
        description="Walls shared with another zone"
    )
    interior_floor_construction: str = Field(
        description="Floors and ceilings shared with another zone"
    )


class IntakeOutput(BaseModel):
    """Structured output from intake LLM call.

    `building` and `site_location` are idfpy models populated directly by
    the LLM's structured output; intake_node adds them to the IDF as is.

    Zones and their geometry are structured and built in code. All
    `*_specs` fields are natural-language task instructions passed to the
    corresponding phase agent.
    """

    building: Building = Field(
        description="Building object (name, orientation, terrain, tolerances)"
    )
    site_location: SiteLocation = Field(
        description="Site location (latitude, longitude, time zone, elevation)"
    )
    zones: list[ZoneGeometrySchema] = Field(
        description="Every thermal zone with its floor plan, level and height"
    )
    material_specs: str = Field(
        description="Material definitions with thermal properties"
    )
    schedule_specs: str = Field(
        description="Schedule definitions: occupancy, lighting, HVAC operation patterns"
    )
    construction_specs: str = Field(
        description="Construction assembly instructions referencing materials"
    )
    surface_specs: str = Field(
        description=(
            "Only geometry the zone prisms cannot express: sloped or pitched "
            "roofs and sloped walls, which replace faces of the extruded "
            "zones. Empty for buildings with vertical walls and flat roofs."
        )
    )
    fenestration_specs: str = Field(
        description="Window/door instructions referencing surfaces"
    )
    hvac_specs: str = Field(
        description="HVAC system type, thermostat setpoints, schedule references"
    )
    people_specs: str = Field(
        description="Occupancy: zone assignment, density, activity schedule per zone"
    )
    lights_specs: str = Field(
        description="Lighting: zone assignment, power density, schedule per zone"
    )
    equipment_specs: str = Field(
        description="Plug loads (ElectricEquipment): zone assignment, power "
        "density, schedule per zone; empty if the brief has none"
    )

    @field_validator("zones")
    @classmethod
    def _unique_zone_names(
        cls, zones: list[ZoneGeometrySchema]
    ) -> list[ZoneGeometrySchema]:
        names = [z.name for z in zones]
        if repeated := sorted({n for n in names if names.count(n) > 1}):
            raise ValueError(f"zone names must be unique, repeated: {repeated}")
        return zones


SPEC_PHASES: Final[dict[str, tuple[Phase, ...]]] = {
    "zones": ("zone", "surface"),  # only "surface" if the zone names stay
    "material_specs": ("material",),
    "schedule_specs": ("schedule",),
    "construction_specs": ("construction",),
    "surface_specs": ("surface",),
    "fenestration_specs": ("fenestration",),
    "hvac_specs": ("hvac",),
    "people_specs": ("people",),
    "lights_specs": ("lights",),
    "equipment_specs": ("equipment",),
}
"""Phases that build from each intake field; Building and Site:Location are
written by intake itself."""


class IntakePatch(BaseModel):
    """Changes to an earlier intake output; omitted fields stay as they were."""

    reason: str = Field(description="Which errors the changes fix, and how")
    building: Building | None = None
    site_location: SiteLocation | None = None
    zones: list[ZoneGeometrySchema] | None = Field(
        default=None, description="The complete new zone list, if any zone changes"
    )
    material_specs: str | None = None
    schedule_specs: str | None = None
    construction_specs: str | None = None
    surface_specs: str | None = None
    fenestration_specs: str | None = None
    hvac_specs: str | None = None
    people_specs: str | None = None
    lights_specs: str | None = None
    equipment_specs: str | None = None

    def apply(self, intake: IntakeOutput) -> tuple[IntakeOutput, set[Phase]]:
        """The patched intake output and the phases whose input changed.

        Raises:
            pydantic.ValidationError: If the patched output is invalid, e.g.
                repeats a zone name.
        """
        given = {
            field: value
            for field in IntakeOutput.model_fields
            if (value := getattr(self, field)) is not None
        }
        current = intake.model_dump()
        changed = {
            field for field, value in given.items() if _dumped(value) != current[field]
        }
        patched = IntakeOutput.model_validate(current | {f: given[f] for f in changed})
        phases = {p for field in changed for p in SPEC_PHASES.get(field, ())}
        if {z.name for z in patched.zones} == {z.name for z in intake.zones}:
            # Loads and HVAC name zones, not their geometry; keep them.
            phases.discard("zone")
        return patched, phases


def _dumped(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, list):
        return [_dumped(v) for v in value]
    return value


@dataclass(frozen=True)
class SimContext:
    """Immutable runtime context, passed via StateGraph context_schema."""

    epw_path: Path
    output_dir: Path = DEFAULT_OUTPUT_DIR


def _merge_idf(old_idf: IDF, new_idf: IDF) -> IDF:
    """Union-merge two IDF containers into a fresh one; new wins on conflict.

    Both inputs are serialized through the epJSON round-trip, so the result
    shares no object references with either side. Entries identical to the
    same-keyed `old` entry are skipped: nameless multi-instance objects
    (e.g. Output:Variable) get deterministic positional keys from
    ``to_dict``, so without this both branches copied from the same parent
    would duplicate them on every merge.
    """
    old_dict = old_idf.to_dict()
    merged = IDF.from_dict(old_dict)

    additions: dict[str, dict[str, dict[str, Any]]] = {}
    for object_type, objects in new_idf.to_dict().items():
        existing = old_dict.get(object_type, {})
        for key, fields in objects.items():
            if existing.get(key) == fields:
                continue
            additions.setdefault(object_type, {})[key] = fields

    merged.merge_dict(additions, on_conflict="replace")
    return merged


def merge_config_state(old: ConfigState, new: ConfigState) -> ConfigState:
    """Reducer for parallel branches: union of both IDFs, new wins on conflict."""
    merged = ConfigState()
    merged.attach_idf(_merge_idf(old.idf, new.idf))
    return merged


class AgentState(BaseModel):
    """Top-level graph state.

    `messages` holds only intake conversation and one-line phase summaries.
    Phase agent tool-calling history lives in TraceCollector and is
    extracted separately for fine-tuning.
    """

    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    user_input: str = ""
    image_paths: list[str] = Field(default_factory=list)

    config_state: Annotated[ConfigState, merge_config_state] = Field(
        default_factory=ConfigState
    )
    intake_output: IntakeOutput | None = None

    validation_errors: list[ModelIssue] = Field(default_factory=list)
    build_issues: Annotated[list[ModelIssue], operator.add] = Field(
        default_factory=list,
        description="Problems found while building objects in code, e.g. a "
        "zone that cannot be extruded; reset before each rerun",
    )
    pending_phases: list[Phase] = Field(
        default_factory=list, description="Phases that run in the current pass"
    )
    unfinished_phases: list[Phase] = Field(
        default_factory=list,
        description="Pending phases skipped because the foundation check failed",
    )
    phase_feedback: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Problems from the previous pass, by the phase that must fix them",
    )
    review_feedback: str = Field(
        default="", description="Corrections from the human reviewer"
    )
    subgroup_retried: bool = False
    global_retries: int = 0
    max_global_retries: int = MAX_GLOBAL_RETRIES


class AgentStateUpdate(TypedDict, total=False):
    """Partial update returned by graph nodes."""

    messages: Sequence[AnyMessage]
    user_input: str
    image_paths: list[str]
    config_state: ConfigState
    intake_output: IntakeOutput | None
    validation_errors: list[ModelIssue]
    build_issues: list[ModelIssue]
    pending_phases: list[Phase]
    unfinished_phases: list[Phase]
    phase_feedback: dict[str, list[str]]
    review_feedback: str
    subgroup_retried: bool
    global_retries: int
