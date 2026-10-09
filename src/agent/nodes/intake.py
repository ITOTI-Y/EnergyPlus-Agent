from __future__ import annotations

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

Rules:
1. If latitude/longitude are not given, infer from the city/region mentioned.
2. Use reasonable office-building defaults when a parameter is missing
   (e.g., tolerance 0.04, terrain 'City', solar distribution 'FullExterior').
3. Each `*_specs` field must be concrete: list zone names, material types,
   schedule patterns, etc. Do NOT output placeholders like 'TBD'.
4. Internal consistency is CRITICAL — the phase agents work from your
   specs. Names referenced across subsystems must MATCH EXACTLY
   (case, underscores, everything):
   - Constructions named in `zone_plans` / `surface_specs` /
     `fenestration_specs` must be defined in `construction_specs` with
     the IDENTICAL name, opaque for walls, roofs and floors. Give the
     interior floor layers from the ceiling below up to the floor above;
     the face on the other side gets the reversed layers automatically.
   - Schedules named in `hvac_specs` / `people_specs` / `lights_specs` /
     `equipment_specs`
     must be defined in `schedule_specs` with the IDENTICAL name.
   - Zones named in `surface_specs` / `fenestration_specs` /
     `people_specs` / `lights_specs` / `equipment_specs` / `hvac_specs`
     must be zones of `storeys` and `zone_plans`, named
     '<storey>_<plan key>' exactly.
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
   default ~120 W/person for seated office work.
"""

PHOTO_READING_INTRO = """Photo reading (a vision model's reading of the attached photos or
drawings; counts and dimensions are estimates). Build the zones from it:
- each block is ONE rectangular plan placed where `position` says, side by
  side or on top of the others, never overlapping another plan on the same
  storey: a core beside the tower is next to the tower's plan, not inside
  it;
- a block's plan is on the storeys `bottom_storey` to `bottom_storey +
  storeys - 1`, with a typical storey and a multiplier for repeated
  floors;
- office, podium and tower plans of at least 12.14 m by 12.14 m take
  `zoning` 'perimeter_core'; a service core and narrow blocks take
  'single';
- the facade window types and window-to-wall ratios go into
  `fenestration_specs`.
Values the text gives take precedence over the reading."""

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
    return HumanMessage(
        content=f"{state.user_input}\n\n{PHOTO_READING_INTRO}\n{reading}"
    )


def storeys_match_reading(
    output: IntakeOutput, reading: PhotoReadingSchema | None
) -> IntakeOutput:
    """The output, if its storeys reach as high as the photo reading's blocks.

    With a reading of 13 storeys, Haiku once modelled 3 and dropped the
    tower; the count is the one fact the reading gives exactly, so it is
    checked in code and a mismatch goes back to the LLM.

    Raises:
        ValueError: If the storeys, counted with their multipliers, differ
            from the top of the reading's highest block.
    """
    if reading is None or not reading.blocks:
        return output
    expected = max(b.bottom_storey + b.storeys - 1 for b in reading.blocks)
    given = sum(s.multiplier for s in output.storeys)
    if given != expected:
        raise ValueError(
            f"the photo reading's highest block ends on storey {expected}, but "
            f"`storeys` stand for {given} (multipliers summed); give every "
            "block its storeys, the repeated ones as one typical storey with a "
            "multiplier"
        )
    return output


def _revision(state: AgentState, previous: IntakeOutput) -> HumanMessage:
    problems = [str(e) for e in state.validation_errors]
    if state.review_feedback:
        problems.append(f"Reviewer: {state.review_feedback}")
    listed = "\n".join(f"- {p}" for p in problems) or "- none reported"
    return HumanMessage(
        content=f"{REVISION_PROMPT}\n\nSpecifications:\n"
        f"{previous.model_dump_json(indent=1)}\n\nProblems:\n{listed}"
    )


REFERENCE_INTAKE_RULE = """
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
        output, _ = structured(
            create_llm(),
            IntakeOutput,
            [system, _brief(state)],
            lambda o: storeys_match_reading(o, state.photo_reading),
        )
        rerun: set[Phase] = set(PHASE_TYPES)
    else:

        def apply(patch: IntakePatch) -> tuple[IntakeOutput, set[Phase]]:
            patched, changed = patch.apply(previous)
            return storeys_match_reading(patched, state.photo_reading), changed

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
