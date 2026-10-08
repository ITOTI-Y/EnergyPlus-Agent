from __future__ import annotations

import base64
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final, Literal, TypedDict, cast

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from loguru import logger
from pydantic import BaseModel

from src.agent._share import language_directive
from src.agent.llm import create_llm
from src.agent.phases import DEPENDS_ON, PHASE_TYPES, Phase, owner, rerun_closure
from src.agent.state import (
    AgentState,
    AgentStateUpdate,
    IntakeOutput,
    IntakePatch,
    SimContext,
)


class TextContentPart(TypedDict):
    """LangChain multimodal text content part."""

    type: Literal["text"]
    text: str


class ImageContentPart(TypedDict):
    """LangChain multimodal image content part (base64-encoded)."""

    type: Literal["image"]
    source_type: Literal["base64"]
    mime_type: str
    data: str


ContentPart = TextContentPart | ImageContentPart

INTAKE_SYSTEM_PROMPT = """You are an EnergyPlus building-simulation intake specialist.
Given a building description (text and optional architectural drawings —
floorplan, elevation, section, axonometric, perspective, etc.), extract
structured specifications for every subsystem.

You MUST invoke the IntakeOutput tool to return the structured JSON.
Do NOT respond with a text/JSON message — always use the tool call.
Fields:
- `building`: EnergyPlus Building object (name, terrain, convergence tolerances)
- `site_location`: EnergyPlus Site:Location object (latitude, longitude,
  time_zone, elevation)
- `zones`: every thermal zone as a prism: name, floor plan corners
  (X, Y in meters, in order around the zone), floor level, height, and
  the constructions of its exterior walls, roof, ground floor, interior
  walls and interior floors. Code builds the walls, floor and flat roof
  from these, and pairs faces shared by two zones. Zones must not
  overlap; zones on upper storeys start at the sum of the heights below.
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
   - Constructions named in `zones` / `surface_specs` /
     `fenestration_specs` must be defined in `construction_specs` with
     the IDENTICAL name, opaque for walls, roofs and floors. Give the
     interior floor layers from the ceiling below up to the floor above;
     the face on the other side gets the reversed layers automatically.
   - Schedules named in `hvac_specs` / `people_specs` / `lights_specs` /
     `equipment_specs`
     must be defined in `schedule_specs` with the IDENTICAL name.
   - Zones named in `surface_specs` / `fenestration_specs` /
     `people_specs` / `lights_specs` / `equipment_specs` / `hvac_specs`
     must appear in `zones` with the IDENTICAL name.
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

_IMAGE_SUFFIX_TO_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def _load_image_part(path: str) -> ImageContentPart:
    """Load an image file and return a multimodal content part."""
    p = Path(path)
    mime = _IMAGE_SUFFIX_TO_MIME.get(p.suffix.lower(), "image/png")
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return ImageContentPart(
        type="image",
        source_type="base64",
        mime_type=mime,
        data=data,
    )


REVISION_PROMPT = """The specifications below were built and checked. Fix
the problems listed after them by returning an IntakePatch:
- set ONLY the fields that must change and omit all others: an omitted
  field keeps its value, and resending an unchanged field only costs time;
- `zones` replaces the whole zone list, so give every zone when you set it;
- a phase whose field you change is rebuilt from scratch together with
  the phases that depend on it; keep other fields unchanged so their
  objects are kept;
- `reason` says which problem each change fixes.
A problem in an object a phase built from an unchanged, correct
specification may need no change here: that phase gets the problem as
feedback and builds again."""

MAX_STRUCTURED_ATTEMPTS: Final = 2
"""A model replying with text instead of the tool call usually complies
when told so once."""


def _structured[T: BaseModel, R](
    schema: type[T], messages: list[BaseMessage], check: Callable[[T], R]
) -> tuple[T, R]:
    """Call the LLM for ``schema``, retrying once on an unusable reply.

    ``check`` turns the parsed reply into the result the caller needs; a
    ValueError from it, like a reply that is not a tool call, is sent back
    to the LLM for one more attempt.

    Raises:
        RuntimeError: If no attempt gives a usable reply.
    """
    llm = create_llm().with_structured_output(schema, include_raw=True)
    problem = ""
    for _ in range(MAX_STRUCTURED_ATTEMPTS):
        result = cast(dict[str, Any], llm.invoke(messages))
        parsed: T | None = result.get("parsed")
        if parsed is None:
            raw: BaseMessage | None = result.get("raw")
            problem = (
                f"no {schema.__name__} tool call (parsing error "
                f"{result.get('parsing_error')!r}; reply "
                f"{repr(raw.content if raw is not None else raw)[:300]})"
            )
        else:
            try:
                return parsed, check(parsed)
            except ValueError as e:
                problem = str(e)
        logger.warning("intake: unusable reply: {}", problem)
        messages = [
            *messages,
            HumanMessage(
                content=f"Your reply was unusable: {problem}. Call the "
                f"{schema.__name__} tool again with a corrected argument."
            ),
        ]
    raise RuntimeError(f"intake gave no usable {schema.__name__}: {problem}")


def _brief(state: AgentState) -> HumanMessage:
    parts: list[ContentPart] = [TextContentPart(type="text", text=state.user_input)]
    parts += [_load_image_part(path) for path in state.image_paths]
    return HumanMessage(content=cast("list[str | dict[str, Any]]", parts))


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
        output, _ = _structured(IntakeOutput, [system, _brief(state)], lambda o: o)
        rerun: set[Phase] = set(PHASE_TYPES)
    else:
        patch, (output, changed) = _structured(
            IntakePatch,
            [system, _brief(state), _revision(state, previous)],
            lambda p: p.apply(previous),
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
