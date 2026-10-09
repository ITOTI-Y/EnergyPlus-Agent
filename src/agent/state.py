from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path
from typing import Annotated, Any, Final, Literal, Self

from idfpy import IDF
from idfpy.models.location import SiteLocation
from idfpy.models.simulation import Building
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, model_validator
from typing_extensions import TypedDict

from src.agent._share import DEFAULT_OUTPUT_DIR, MAX_GLOBAL_RETRIES
from src.agent.phases import Phase
from src.modeling.geometry import PlanPointSchema
from src.modeling.validation import ModelIssue
from src.modeling.zoning import perimeter_core
from src.reference.search import ReferenceSearch
from src.state.config_state import ConfigState


class ZonePlanSchema(BaseModel):
    """A floor plan used on one or more storeys, with its constructions."""

    key: str = Field(
        description="Short id, word characters and '_' only, e.g. 'S1' or "
        "'Corridor'; a zone is named '<storey>_<key>'"
    )
    plan: list[PlanPointSchema] = Field(
        description="Floor plan corners (X, Y in meters) in order around the zone"
    )
    zoning: Literal["single", "perimeter_core"] = Field(
        default="single",
        description="'single': the plan is one zone '<storey>_<key>'. "
        "'perimeter_core': code splits the rectangle into four 4.57 m deep "
        "perimeter zones and a core zone, '<storey>_<key>_N', '_E', '_S', "
        "'_W' and '_Core' (needs both sides of at least 12.14 m)",
    )
    block: str | None = Field(
        default=None,
        description="Only with a photo reading: the name of the reading's "
        "block this plan belongs to; code then builds the storeys",
    )
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


class StoreyZoneSchema(BaseModel):
    """A plan placed on a storey."""

    plan: str = Field(description="Key of a zone plan")
    height: float | None = Field(
        default=None,
        gt=0,
        description="Only for a zone taller than its storey (e.g. an 8 m "
        "lobby through two storeys); the storey height otherwise",
    )


class StoreySchema(BaseModel):
    """One storey, or a run of identical storeys modelled once.

    Storeys stack from the ground up: each starts where the one below ends,
    so levels are computed, not given (Haiku got 8 + 18 x 3.5 wrong).
    """

    name: str = Field(
        description="Short id, e.g. 'G', 'L2', 'T' or 'Top'; '' for a "
        "single-storey building, whose zones are then named by plan key"
    )
    height: float = Field(gt=0, description="Floor-to-floor height in meters")
    multiplier: int = Field(
        default=1,
        ge=1,
        description="Number of identical storeys this one stands for "
        "(e.g. 18 for storeys 3-20); 1 otherwise",
    )
    zones: list[StoreyZoneSchema] = Field(description="Plans on this storey")


class ZoneGeometrySchema(BaseModel):
    """One thermal zone as a prism, derived from a storey and a plan."""

    name: str
    plan: list[PlanPointSchema]
    floor_z: float
    height: float
    multiplier: int
    exterior_wall_construction: str
    roof_construction: str
    ground_floor_construction: str
    interior_wall_construction: str
    interior_floor_construction: str


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
    zone_plans: list[ZonePlanSchema] = Field(
        description="Each distinct floor plan once, with its constructions"
    )
    storeys: list[StoreySchema] = Field(
        description="Every storey, bottom up, with the plans on it"
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
        description="HVAC system per conditioned zone; one thermostat template "
        "per distinct pair of heating/cooling setpoint schedules, with the "
        "zones that use it; availability schedule references"
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

    @property
    def zones(self) -> list[ZoneGeometrySchema]:
        """Every zone: each plan on each storey, named '<storey>_<plan key>'
        (the plan key alone on a storey named ''), at its storey's level; a
        'perimeter_core' plan gives five zones with the side as a suffix.

        Raises:
            ValueError: If a 'perimeter_core' plan cannot be split.
        """
        plans = {p.key: p for p in self.zone_plans}
        # One more bottom than storeys: the last is the top of the building.
        bottoms = accumulate(
            (s.height * s.multiplier for s in self.storeys), initial=0.0
        )
        zones = []
        for storey, floor_z in zip(self.storeys, bottoms, strict=False):
            for entry in storey.zones:
                plan = plans[entry.plan]
                name = f"{storey.name}_{entry.plan}" if storey.name else entry.plan
                parts = (
                    [(name, plan.plan)]
                    if plan.zoning == "single"
                    else [
                        (
                            f"{name}_{side}",
                            [PlanPointSchema(x=x, y=y) for x, y in corners],
                        )
                        for side, corners in perimeter_core(
                            [(q.x, q.y) for q in plan.plan]
                        )
                    ]
                )
                zones += [
                    ZoneGeometrySchema(
                        name=part_name,
                        plan=corners,
                        floor_z=floor_z,
                        height=entry.height or storey.height,
                        multiplier=storey.multiplier,
                        **plan.model_dump(exclude={"key", "plan", "zoning", "block"}),
                    )
                    for part_name, corners in parts
                ]
        return zones

    @model_validator(mode="after")
    def _plans_exist_and_names_are_unique(self) -> Self:
        keys = [p.key for p in self.zone_plans]
        if repeated := sorted({k for k in keys if keys.count(k) > 1}):
            raise ValueError(f"zone plan keys must be unique, repeated: {repeated}")
        unknown = sorted({e.plan for s in self.storeys for e in s.zones} - set(keys))
        if unknown:
            raise ValueError(f"storeys use plans that are not defined: {unknown}")
        names = [z.name for z in self.zones]
        if repeated := sorted({n for n in names if names.count(n) > 1}):
            raise ValueError(f"zone names must be unique, repeated: {repeated}")
        return self


SPEC_PHASES: Final[dict[str, tuple[Phase, ...]]] = {
    # Zones change with plans or storeys; only "surface" if the names stay.
    "zone_plans": ("zone", "surface"),
    "storeys": ("zone", "surface"),
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
    zone_plans: list[ZonePlanSchema] | None = Field(
        default=None, description="The complete new plan list, if a plan changes"
    )
    storeys: list[StoreySchema] | None = Field(
        default=None, description="The complete new storey list, if one changes"
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
    reference: ReferenceSearch | None = None
    """Prototype reference search, when configured; phases add its tools."""


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


class MassingBlockSchema(BaseModel):
    """A block of the building as a photo shows it."""

    name: str = Field(description="e.g. 'podium', 'tower', 'service core'")
    role: Literal["podium", "tower", "core", "wing", "other"]
    storeys: int = Field(
        ge=1, description="Storeys of this block, counted from its window bands"
    )
    bottom_storey: int = Field(
        ge=1, description="Storey the block starts on; 1 is the ground storey"
    )
    width_m: float = Field(gt=0, description="Estimated length along the front")
    depth_m: float = Field(gt=0, description="Estimated depth from the front")
    position: str = Field(description="Where it sits relative to the other blocks")


class FacadeSchema(BaseModel):
    """What one side of a block shows."""

    block: str = Field(description="Name of the block")
    side: Literal["front", "left", "right", "back"]
    windows: str = Field(
        description="e.g. 'continuous ribbon', 'punched', 'curtain wall', 'none'"
    )
    window_to_wall_ratio: float = Field(ge=0, le=1)


class PhotoReadingSchema(BaseModel):
    """A vision model's reading of the building's photos or drawings."""

    # First, so the model writes its counting before the numbers: the call
    # is a forced tool call and has no other place to reason.
    reading: str = Field(
        description="FIRST, step by step: count each block's window bands "
        "from the ground up, name the blocks, and name the scale cues "
        "(doors, people, cars, bays) behind the dimensions"
    )
    total_storeys: int = Field(ge=1)
    ground_storey_height_m: float = Field(
        gt=0, description="Estimated floor-to-floor height of the ground storey"
    )
    storey_height_m: float = Field(
        gt=0, description="Estimated floor-to-floor height of the other storeys"
    )
    blocks: list[MassingBlockSchema]
    facades: list[FacadeSchema]
    assumptions: list[str] = Field(
        description="What the images do not show and was assumed, e.g. the "
        "back facades or the depth"
    )

    @model_validator(mode="after")
    def _block_names_are_unique(self) -> Self:
        # Intake tags each plan with a block name; storeys follow from it.
        names = [b.name for b in self.blocks]
        if repeated := sorted({n for n in names if names.count(n) > 1}):
            raise ValueError(f"block names must be unique, repeated: {repeated}")
        return self


class AgentState(BaseModel):
    """Top-level graph state.

    `messages` holds only intake conversation and one-line phase summaries.
    Phase agent tool-calling history lives in TraceCollector and is
    extracted separately for fine-tuning.
    """

    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    user_input: str = ""
    image_paths: list[str] = Field(default_factory=list)
    photo_reading: PhotoReadingSchema | None = Field(
        default=None, description="The images read once, before intake"
    )

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
    photo_reading: PhotoReadingSchema | None
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
