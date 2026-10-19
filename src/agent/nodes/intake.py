from __future__ import annotations

import re
from typing import Final

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from loguru import logger

from src.agent._share import language_directive
from src.agent.llm import create_llm
from src.agent.nodes._share import structured
from src.agent.phases import DEPENDS_ON, PHASE_TYPES, Phase, owner, rerun_closure
from src.agent.state import (
    AgentState,
    AgentStateUpdate,
    IntakeOutput,
    IntakePatch,
    PhotoReadingSchema,
    SimContext,
)
from src.agent.storeys import storeys_from_reading

INTAKE_SYSTEM_PROMPT = """You are an EnergyPlus building-simulation intake specialist.
Given a building description in text, and optionally a reading of its
photos or drawings by a vision model, extract structured specifications
for every subsystem.

You MUST invoke the IntakeOutput tool to return the structured JSON.
Do NOT respond with a text/JSON message — always use the tool call.
Fields:
- `building`: EnergyPlus Building object (name, terrain, convergence tolerances)
- `site_location`: EnergyPlus Site:Location object (latitude, longitude,
  time_zone, elevation)
- `zone_plans` and `storeys`: the zones, as floor plans placed on
  storeys. Give each distinct plan ONCE in `zone_plans` (key, corners X, Y
  in meters in order around the zone, and the constructions of its
  exterior walls, roof, ground floor, interior walls and interior floors),
  then list EVERY storey bottom up in `storeys` (name, floor-to-floor
  height, multiplier, and the plan keys on it). Every zone is named
  '<storey name>_<plan key>' (e.g. 'L2_S1'); use exactly these names in
  every other field. For a single-storey building, name the storey ''
  and the zones are named by plan key. Code builds walls, floors and flat
  roofs from this and pairs faces shared by two zones. Zones must not
  overlap.
  When the brief gives no internal layout for a rectangular block of at
  least 12.14 m by 12.14 m (e.g. an office floor), give it ONE plan with
  `zoning` 'perimeter_core': code splits it into the standard 4.57 m
  perimeter zones and a core, named '<storey>_<key>_N', '_E', '_S', '_W'
  and '_Core'; use these names in the other fields. Plans the brief lays
  out, and narrow or non-rectangular blocks, keep `zoning` 'single'.
  Repeated typical floors are modelled once, as the DOE prototypes do:
  the ground storey and the top storey with multiplier 1, and between
  them ONE typical storey with `multiplier` = the number of typical
  floors (e.g. 18 for storeys 3-20). Floor levels are computed from the
  heights and multipliers of the storeys below. A space through several storeys (atrium) is a plan on each of
  the ground, typical and top storeys. A zone taller than its storey
  (e.g. an 8 m lobby through storeys 1-2) sets `height` on its storey
  entry and is left out of the storey it reaches into.
- `*_specs`: one natural-language instruction string per subsystem agent.
  `surface_specs` is only for sloped or pitched roofs and sloped walls;
  leave it empty when every zone has vertical walls and a flat roof.
  `fenestration_specs` gives windows by zone plan key and facing (north,
  east, south, west): a window-to-wall ratio and a window construction
  for each; the fenestration phase applies a ratio to every zone of a
  plan on every storey at once. Give the size or position of single
  windows only when the brief does.

Rules:
1. If latitude/longitude are not given, infer from the city/region mentioned.
2. Use reasonable office-building defaults when a parameter is missing
   (e.g., tolerance 0.04, terrain 'City', solar distribution 'FullExterior').
3. Each `*_specs` field must be concrete: list zone names, material types,
   schedule patterns, etc. Do NOT output placeholders like 'TBD'.
   `equipment_specs` is never empty: when the brief gives no plug loads,
   give typical densities for the space types (e.g. 10 W/m^2 for offices)
   on a fraction schedule.
4. Internal consistency is CRITICAL — the phase agents work from your
   specs. Names referenced across subsystems must MATCH EXACTLY
   (case, underscores, everything):
   - Constructions named in `zone_plans` / `surface_specs` /
     `fenestration_specs` must be defined in `construction_specs` with
     the IDENTICAL name, opaque for walls, roofs and floors. Give the
     interior floor layers from the ceiling below up to the floor above;
     the face on the other side gets the reversed layers automatically.
   - Every window construction in `construction_specs` has a glazing
     layer, e.g. a simple glazing system given by U-factor, SHGC and
     visible transmittance, that `material_specs` defines with the
     IDENTICAL name.
   - Schedules named in `hvac_specs` / `people_specs` / `lights_specs` /
     `equipment_specs`
     must be defined in `schedule_specs` with the IDENTICAL name.
   - Zones named in `surface_specs` / `fenestration_specs` /
     `people_specs` / `lights_specs` / `equipment_specs` / `hvac_specs`
     must be zones of `storeys` and `zone_plans`, named
     '<storey>_<plan key>' exactly. Name all zones of a plan by its key
     instead ('every zone of plan Tower'): the phases apply it to every
     zone made from the plan, on every storey, so list no storeys.
   Pick names once, reuse them verbatim. No synonyms, no pluralization.
5. Name format — EVERY Name field (building.name, site_location.name,
   zone / material / construction / surface / fenestration / schedule /
   thermostat / people / lights / equipment names) MUST use ONLY word characters
   (letters, digits) with `_` as the ONLY word separator. NO spaces,
   NO commas, NO semicolons, NO hyphens, NO slashes, NO parentheses.
   IDF uses `,` and `;` as field delimiters; other punctuation causes
   silent field shifts that crash EnergyPlus.
   Examples:
     ✓ "Shenzhen_CN", "Office_Zone", "ExtWall_Brick_EPS_Gypsum",
       "Schedule_Office_Occupancy_Weekday"
     ✗ "Shenzhen, China"     (comma)
     ✗ "Office Zone 1"       (space)
     ✗ "Wall-Assembly-A"     (hyphen)
     ✗ "Schedule (Weekday)"  (parentheses)
6. `schedule_specs` MUST be complete — every schedule referenced by a
   downstream phase has to be described here, because the schedule
   agent runs FIRST. Checklist of schedule
   types the downstream phases will request:

     Downstream field                              | Schedule type   | Unit
     ----------------------------------------------|-----------------|------
     thermostat.heating_setpoint_schedule_name     | Temperature     | degC
     thermostat.cooling_setpoint_schedule_name     | Temperature     | degC
     ideal_loads.system_availability_schedule_name | Fraction / OnOff| -
     people.number_of_people_schedule_name         | Fraction        | -
     people.activity_level_schedule_name           | ActivityLevel   | W/person
     lights.schedule_name                          | Fraction        | -
     equipment.schedule_name                       | Fraction        | -

   For every row where the downstream phase is non-empty, `schedule_specs`
   must (a) name the schedule, (b) state the schedule type limits it
   uses, and (c) give the value profile for EVERY day type, weekends,
   holidays and design days included (e.g. "weekdays 8-18 at 1.0, all
   other days 0.0"). The activity_level schedule is commonly forgotten —
   default ~120 W/person for seated office work. It is constant at all
   times, never 0: when people are present follows from the occupancy
   fraction schedule. The heating setpoint is at most the cooling
   setpoint at every time of every day type.
"""

PHOTO_READING_INTRO = """Photo reading (a vision model's reading of the attached photos or
drawings; counts and dimensions are estimates), and each block's footprint
computed from it. Build the zones from them:
- give each block ONE rectangular plan whose corners are its footprint
  (x0, y0, x1, y1) exactly; only when the text lays out zones in a block,
  give rectangles that together fill the footprint. The footprints carry
  the setbacks: do not move or resize them;
- set each plan's `block` to its block's `name` exactly; every block gets
  a plan;
- give `storeys` as an empty list: code builds them from the blocks'
  storeys and the reading's storey heights, named 'S<n>' by their first
  storey (a typical storey stands for the ones above it), so zones are
  named like 'S5_Tower_N'. In the other fields name zones by plan, e.g.
  "all zones of plan Tower", not by storey;
- office, podium and tower plans of at least 12.14 m by 12.14 m take
  `zoning` 'perimeter_core'; a service core and narrow blocks take
  'single';
- the facade window types and window-to-wall ratios go into
  `fenestration_specs`, per plan and facing (front south, back north, left
  west, right east). A side the reading does not give was not seen: a back
  takes the front's ratio; an end (west or east) takes the other end's, or
  0.1 when neither end is given, as the ends of a slab are mostly stairs
  and service cores;
- a facade's `space` says what is behind it; stair or service core sides
  keep their small ratio;
- each facing's windows in `fenestration_specs` are a ribbon or punched,
  as the facade's `pattern`; punched windows take the facade's window
  size, else 1.2 m wide by 1.5 m high.
Other values the text gives take precedence over the reading."""

REVISION_PROMPT = """The specifications below were built and checked. Fix
the problems listed after them by returning an IntakePatch:
- set ONLY the fields that must change and omit all others: an omitted
  field keeps its value, and resending an unchanged field only costs time;
- `zone_plans` and `storeys` each replace the whole list, so give every
  plan or storey when you set one;
- a phase whose field you change is rebuilt from scratch together with
  the phases that depend on it; keep other fields unchanged so their
  objects are kept;
- `reason` says which problem each change fixes.
A problem in an object a phase built from an unchanged, correct
specification may need no change here: that phase gets the problem as
feedback and builds again."""


def _brief(state: AgentState) -> HumanMessage:
    """The brief, with the photo reading when images were given.

    Intake sees the reading, not the images: the vision model read them in
    its own call, and revisions resend no image.
    """
    if state.photo_reading is None:
        return HumanMessage(content=state.user_input)
    reading = state.photo_reading.model_dump_json(indent=1)
    footprints = "\n".join(
        f"- {name}: x0 {x0}, y0 {y0}, x1 {x1}, y1 {y1}"
        for name, (x0, y0, x1, y1) in state.photo_reading.footprints().items()
    )
    return HumanMessage(
        content=f"{state.user_input}\n\n{PHOTO_READING_INTRO}\n{reading}\n\n"
        f"Block footprints (metres, x east, y north, fronts facing south):\n"
        f"{footprints}"
    )


PLAN_CONSTRUCTIONS: Final = (
    "exterior_wall_construction",
    "roof_construction",
    "ground_floor_construction",
    "interior_wall_construction",
    "interior_floor_construction",
)


def _with_defined_constructions(output: IntakeOutput) -> IntakeOutput:
    """The output, once every construction its plans name appears in
    `construction_specs`.

    Caught here, intake fixes it in the same call; otherwise every zone
    reports it after a whole pass of the phases.

    Raises:
        ValueError: Naming the constructions `construction_specs` lacks.
    """
    named = {
        getattr(p, field) for p in output.zone_plans for field in PLAN_CONSTRUCTIONS
    }
    missing = sorted(
        n
        for n in named
        if not re.search(rf"\b{re.escape(n)}\b", output.construction_specs)
    )
    if missing:
        raise ValueError(
            f"zone_plans name constructions that construction_specs does not "
            f"define: {missing}; define them there or use defined ones"
        )
    return output


def with_reading_storeys(
    output: IntakeOutput, reading: PhotoReadingSchema | None
) -> IntakeOutput:
    """The output with its storeys built from the photo reading, if any.

    Raises:
        ValueError: If the plans and the reading's blocks do not match (see
            ``storeys_from_reading``); intake gets the reason and corrects it.
    """
    if reading is None:
        return output
    storeys = storeys_from_reading(output.zone_plans, reading)
    return IntakeOutput.model_validate(
        output.model_dump() | {"storeys": [s.model_dump() for s in storeys]}
    )


def _revision(state: AgentState, previous: IntakeOutput) -> HumanMessage:
    problems = [str(e) for e in state.validation_errors]
    if state.review_feedback:
        problems.append(f"Reviewer: {state.review_feedback}")
    listed = "\n".join(f"- {p}" for p in problems) or "- none reported"
    return HumanMessage(
        content=f"{REVISION_PROMPT}\n\nSpecifications:\n"
        f"{previous.model_dump_json(indent=1)}\n\nProblems:\n{listed}"
    )


REFERENCE_INTAKE_RULE: Final = """
7. A reference library of DOE prototype buildings is available to the
   material, construction and schedule phases. Where the brief gives no
   values, do NOT invent material properties, layer thicknesses or
   schedule profiles: describe each material, construction and schedule by
   purpose and building type (e.g. "exterior wall insulation of a medium
   office suited to the site's climate"), still giving the names other
   specs reference, and let those phases take the values from the library.
   Values the brief gives are passed on as given.
"""


def intake_node(state: AgentState, runtime: Runtime[SimContext]) -> AgentStateUpdate:
    """Write the specifications on the first pass, revise them on retries.

    The first pass runs every phase. A revision returns a patch; the phases
    whose input changed, the phases owning the reported problems, and their
    dependants run again, and the rest keep their objects. Building and
    Site:Location go into the model here.
    """
    rules = INTAKE_SYSTEM_PROMPT
    if runtime.context.reference is not None:
        rules += REFERENCE_INTAKE_RULE
    system = SystemMessage(content=rules + language_directive())
    previous = state.intake_output
    if previous is None:
        # The checked result, not the reply: with a reading it has the storeys.
        _, output = structured(
            create_llm(),
            IntakeOutput,
            [system, _brief(state)],
            lambda o: _with_defined_constructions(
                with_reading_storeys(o, state.photo_reading)
            ),
        )
        rerun: set[Phase] = set(PHASE_TYPES)
    else:

        def apply(patch: IntakePatch) -> tuple[IntakeOutput, set[Phase]]:
            if state.photo_reading is not None:
                # Storeys follow from the plans and the reading alone; built
                # first, they are checked and compared like any patch field.
                plans = (
                    previous.zone_plans
                    if patch.zone_plans is None
                    else patch.zone_plans
                )
                storeys = storeys_from_reading(plans, state.photo_reading)
                patch = patch.model_copy(update={"storeys": storeys})
            output, changed = patch.apply(previous)
            return _with_defined_constructions(output), changed

        patch, (output, changed) = structured(
            create_llm(),
            IntakePatch,
            [system, _brief(state), _revision(state, previous)],
            apply,
        )
        logger.info(
            "intake revision sets {}, changing {}: {}",
            sorted(patch.model_fields_set - {"reason"}),
            sorted(changed),
            patch.reason,
        )
        owners = [owner(e) for e in state.validation_errors]
        # Zones are built from the intake output alone: only a changed zone
        # list rebuilds them, which `changed` already says.
        fixing = {p for p in owners if p is not None and p != "zone"}
        rerun = (
            set(PHASE_TYPES)
            if None in owners
            else rerun_closure(changed | fixing) | set(state.unfinished_phases)
        )

    config = state.config_state.model_copy(deep=True)
    for obj in (output.building, output.site_location):
        object_type = type(obj)
        for name in config.idf.all_of_type(object_type):
            config.idf.remove(object_type, name)
        # Copy while unbound so intake_output keeps objects the IDF does not own.
        config.idf.add(obj.model_copy())

    return AgentStateUpdate(
        intake_output=output,
        config_state=config,
        pending_phases=[p for p in DEPENDS_ON if p in rerun],
        validation_errors=[],
        review_feedback="",
        subgroup_retried=False,
    )
