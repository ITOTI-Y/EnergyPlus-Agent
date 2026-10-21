from __future__ import annotations

import operator
import re
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate, combinations
from pathlib import Path
from typing import Annotated, Any, Final, Literal, Self

from idfpy import IDF
from idfpy.models import Building, SiteLocation
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from loguru import logger
from pydantic import AfterValidator, BaseModel, Field, field_validator, model_validator
from typing_extensions import TypedDict

from src.agent._share import DEFAULT_OUTPUT_DIR, MAX_GLOBAL_RETRIES
from src.agent.phases import Phase
from src.modeling.geometry import PlanPointSchema
from src.modeling.validation import ModelIssue
from src.modeling.zoning import perimeter_core
from src.reference.search import ReferenceSearch
from src.state.config_state import ConfigState


def _block_name(name: str) -> str:
    """Letters, digits and single spaces: models write the same name with
    different dashes or spacing, e.g. 'lower-left' with U+2011 or U+002D."""
    return " ".join(re.findall(r"[^\W_]+", name))


type BlockName = Annotated[str, AfterValidator(_block_name)]


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
    block: BlockName | None = Field(
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
    so levels are computed from the heights and multipliers below.
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
    plan_key: str = Field(description="Key of the zone plan it is made from")
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
        description="Material definitions by name, with the values the rules ask for"
    )
    schedule_specs: str = Field(
        description="Schedule definitions: occupancy, lighting, HVAC operation; "
        "profiles as the rules ask for"
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
        "density, schedule per zone; typical densities for the space types "
        "when the brief gives none"
    )

    @field_validator("equipment_specs")
    @classmethod
    def _plug_loads_given(cls, value: str) -> str:
        # Occupied buildings have plug loads; leaving them out understates
        # the internal gains.
        if not value.strip():
            raise ValueError(
                "equipment_specs must give the plug loads of every zone, with "
                "typical densities for the space types when the brief gives none"
            )
        return value

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
                        plan_key=entry.plan,
                        plan=corners,
                        floor_z=floor_z,
                        height=entry.height or storey.height,
                        multiplier=storey.multiplier,
                        **plan.model_dump(exclude={"key", "plan", "zoning", "block"}),
                    )
                    for part_name, corners in parts
                ]
        return zones

    def plan_zones(self) -> dict[str, list[str]]:
        """The zones made from each plan, on every storey it is placed on."""
        result: dict[str, list[str]] = {p.key: [] for p in self.zone_plans}
        for zone in self.zones:
            result[zone.plan_key].append(zone.name)
        return result

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
    # Zones change with plans or storeys; only "surface" if the names and
    # multipliers stay.
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
        if {(z.name, z.multiplier) for z in patched.zones} == {
            (z.name, z.multiplier) for z in intake.zones
        }:
            # Loads and HVAC name zones, not their geometry; keep them. A new
            # multiplier changes the Zone objects, which the surfaces read.
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


OVERLAP_TOLERANCE_M: Final = 0.1
"""Blocks touching within this distance count as side by side."""

MIN_BLOCK_SIDE_M: Final = 3.0
"""What is left of a trimmed block beside taller ones is at least this wide;
a narrower strip is a misplaced block, not a zone."""

type Rectangle = tuple[float, float, float, float]
"""(x0, y0, x1, y1) in metres."""


class MassingBlockSchema(BaseModel):
    """A block of the building as a photo shows it."""

    name: BlockName = Field(description="e.g. 'podium', 'tower', 'service core'")
    role: Literal["podium", "tower", "core", "wing", "other"]
    storeys: int = Field(
        ge=1, description="Storeys of this block, counted from its window bands"
    )
    bottom_storey: int = Field(
        ge=1, description="Storey the block starts on; 1 is the ground storey"
    )
    width_m: float = Field(gt=0, description="Estimated length along the front")
    depth_m: float = Field(gt=0, description="Estimated depth from the front")
    x_m: float = Field(
        description="West end of the block, metres east of the main (tallest) "
        "block's west end, which is 0; negative further west"
    )
    front_setback_m: float = Field(
        description="How far the block's front face lies behind the main "
        "block's front face, which is 0: positive when it lies further back, "
        "negative when it projects in front"
    )
    volumes: list[int] = Field(
        description="Numbers of the boxed volumes this block is made of; empty "
        "for a block no box shows. Code may set the block's storeys and "
        "setback from a 3D measurement of these volumes"
    )

    @property
    def footprint(self) -> Rectangle:
        """(x0, y0, x1, y1) in metres, the front facing south at y0."""
        return (
            self.x_m,
            self.front_setback_m,
            self.x_m + self.width_m,
            self.front_setback_m + self.depth_m,
        )

    def spans(self, storey: int) -> bool:
        return self.bottom_storey <= storey < self.bottom_storey + self.storeys


class FacadeSchema(BaseModel):
    """What one side of a block shows."""

    block: BlockName = Field(description="Name of the block")
    side: Literal["front", "left", "right", "back"]
    # Before the ratio, so the ratio follows from what the side is.
    space: Literal[
        "office",
        "residential",
        "lobby or retail",
        "stair or service core",
        "plant or parking",
        "other",
    ] = Field(description="The space behind this side, judged from its windows")
    windows: str = Field(
        description="e.g. 'continuous ribbon', 'punched', 'curtain wall', 'none'"
    )
    pattern: Literal["ribbon", "punched"] = Field(
        description="'ribbon' for continuous bands or curtain walls, 'punched' "
        "for separate windows"
    )
    window_width_m: float | None = Field(
        gt=0, description="One punched window's width, from the photo; else null"
    )
    window_height_m: float | None = Field(
        gt=0, description="One punched window's height, from the photo; else null"
    )
    window_to_wall_ratio: float = Field(ge=0, le=1)


class PhotoReadingSchema(BaseModel):
    """A vision model's reading of the building's photos or drawings."""

    # First, so the model reasons before the numbers: the call is a forced
    # tool call and has no other place to reason.
    reading: str = Field(
        description="FIRST, step by step: how the storeys were counted, the "
        "blocks and where each stands, and the scale cues (doors, people, "
        "cars, bays) behind the dimensions"
    )
    # A list makes the model count one storey per entry; a free number or
    # "about 12" in `reading` lets it undercount.
    storey_rows: list[str] = Field(
        min_length=1,
        description="One entry per storey of the highest block, from the roof "
        "down to the ground: what shows this storey, e.g. 'window band 3', "
        "'hidden behind the podium, extrapolated from the side wall'",
    )
    ground_storey_height_m: float = Field(
        gt=0, description="Estimated floor-to-floor height of the ground storey"
    )
    storey_height_m: float = Field(
        gt=0, description="Estimated floor-to-floor height of the other storeys"
    )
    blocks: list[MassingBlockSchema] = Field(min_length=1)
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

    @model_validator(mode="after")
    def _blocks_reach_the_counted_storeys(self) -> Self:
        top = max(b.bottom_storey + b.storeys - 1 for b in self.blocks)
        if top != len(self.storey_rows):
            raise ValueError(
                f"`storey_rows` counts {len(self.storey_rows)} storeys but the "
                f"highest block ends on storey {top}; make the blocks' "
                "`bottom_storey` and `storeys` match the rows, or recount them"
            )
        return self

    @model_validator(mode="after")
    def _hidden_blocks_are_dropped(self) -> Self:
        # A block with no room beside the taller blocks on its storeys adds
        # no volume: theirs already fills its place.
        placed = self._placed()
        if hidden := [b.name for b in self.blocks if b.name not in placed]:
            logger.warning(
                "photo reading: blocks {} lie inside taller blocks and are dropped",
                hidden,
            )
            self.blocks = [b for b in self.blocks if b.name in placed]
        return self

    def _placed(self) -> dict[str, Rectangle]:
        """The footprint each block keeps beside the taller blocks sharing its
        storeys; a block without at least MIN_BLOCK_SIDE_M of room is left
        out.

        A vision model describes what it sees, e.g. a tower rising out of a
        podium, while zones on one storey must not overlap. Taller blocks
        keep their footprint; a lower block keeps the largest rectangle of
        its footprint beside them.
        """
        placed: dict[str, Rectangle] = {}
        tallest_first = sorted(
            self.blocks,
            key=lambda b: (-(b.bottom_storey + b.storeys), -b.width_m * b.depth_m),
        )
        for block in tallest_first:
            taller = [
                placed[other.name]
                for other in tallest_first
                if other.name in placed
                and any(
                    other.spans(k)
                    for k in range(
                        block.bottom_storey, block.bottom_storey + block.storeys
                    )
                )
            ]
            free = _largest_free_rectangle(block.footprint, taller)
            if free is not None:
                placed[block.name] = free
        return placed

    def footprints(self) -> dict[str, Rectangle]:
        """Each block's (x0, y0, x1, y1), shifted so the building starts at 0, 0."""
        placed = self._placed()
        dx = min(r[0] for r in placed.values())
        dy = min(r[1] for r in placed.values())
        return {
            b.name: (
                round(placed[b.name][0] - dx, 2),
                round(placed[b.name][1] - dy, 2),
                round(placed[b.name][2] - dx, 2),
                round(placed[b.name][3] - dy, 2),
            )
            for b in self.blocks
        }


def _overlap(a: Rectangle, b: Rectangle) -> bool:
    return (
        min(a[2], b[2]) - max(a[0], b[0]) > OVERLAP_TOLERANCE_M
        and min(a[3], b[3]) - max(a[1], b[1]) > OVERLAP_TOLERANCE_M
    )


def _largest_free_rectangle(
    rectangle: Rectangle, obstacles: list[Rectangle]
) -> Rectangle | None:
    """The largest rectangle inside ``rectangle`` overlapping no obstacle.

    Its sides lie on the edges of the rectangle and of the obstacles, so
    trying every pair of those coordinates finds it.
    """
    hits = [o for o in obstacles if _overlap(rectangle, o)]
    if not hits:
        return rectangle
    x0, y0, x1, y1 = rectangle
    xs = sorted({x0, x1, *(v for o in hits for v in (o[0], o[2]) if x0 < v < x1)})
    ys = sorted({y0, y1, *(v for o in hits for v in (o[1], o[3]) if y0 < v < y1)})
    candidates = [
        (a, c, b, d)
        for a, b in combinations(xs, 2)
        for c, d in combinations(ys, 2)
        if b - a >= MIN_BLOCK_SIDE_M
        and d - c >= MIN_BLOCK_SIDE_M
        and not any(_overlap((a, c, b, d), o) for o in hits)
    ]
    return max(candidates, key=_area, default=None)


def _area(rectangle: Rectangle) -> float:
    x0, y0, x1, y1 = rectangle
    return (x1 - x0) * (y1 - y0)


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
