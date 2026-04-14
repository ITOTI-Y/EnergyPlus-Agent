from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from loguru import logger

from src.agent._share import language_directive
from src.agent.llm import create_llm
from src.agent.state import AgentState, AgentStateUpdate, IntakeOutput


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
- `building`: BuildingSchema with name, terrain, convergence tolerances
- `site_location`: SiteLocationSchema with latitude, longitude, time_zone, elevation
- `*_specs`: one natural-language instruction string per subsystem agent

Rules:
1. If latitude/longitude are not given, infer from the city/region mentioned.
2. Use reasonable office-building defaults when a parameter is missing
   (e.g., tolerance 0.04, terrain 'City', solar distribution 'FullExterior').
3. Each `*_specs` field must be concrete: list zone names, material types,
   schedule patterns, etc. Do NOT output placeholders like 'TBD'.
4. Internal consistency is CRITICAL — the phase agents work from your
   specs. Names referenced across subsystems must MATCH EXACTLY
   (case, underscores, everything):
   - Constructions named in `surface_specs` / `fenestration_specs` must
     be defined in `construction_specs` with the IDENTICAL name.
   - Schedules named in `hvac_specs` / `people_specs` / `lights_specs`
     must be defined in `schedule_specs` with the IDENTICAL name.
   - Zones named in `surface_specs` / `people_specs` / `lights_specs` /
     `hvac_specs` must be defined in `zone_specs` with the IDENTICAL name.
   Pick names once, reuse them verbatim. No synonyms, no pluralization.
5. Name format — EVERY Name field (building.name, site_location.name,
   zone / material / construction / surface / fenestration / schedule /
   thermostat / people / lights names) MUST use ONLY word characters
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
   agent runs FIRST and will not be re-invoked. Checklist of schedule
   types the downstream phases will request:

     Downstream field                              | Schedule type   | Unit
     ----------------------------------------------|-----------------|------
     thermostat.heating_setpoint_schedule_name     | Temperature     | degC
     thermostat.cooling_setpoint_schedule_name     | Temperature     | degC
     ideal_loads.system_availability_schedule_name | Fraction / OnOff| -
     people.number_of_people_schedule_name         | Fraction        | -
     people.activity_level_schedule_name           | Activity Level  | W/person
     lights.schedule_name                          | Fraction        | -

   For every row where the downstream phase is non-empty, `schedule_specs`
   must (a) name the schedule, (b) state the schedule type limits it
   uses, and (c) give the value profile (e.g. "weekdays 8-18 at 1.0,
   else 0.0"). The activity_level schedule is commonly forgotten —
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


def intake_node(state: AgentState) -> AgentStateUpdate:
    """Parse user_input + image_path into IntakeOutput and seed config_state.

    The LLM returns nested BuildingSchema and SiteLocationSchema directly,
    which intake_node writes into the shared config_state. Phase agents
    read their own `*_specs` strings from intake_output.
    """
    llm = create_llm().with_structured_output(IntakeOutput, include_raw=True)

    text = state.user_input
    if state.validation_errors:
        errors = "\n".join(f"- {e}" for e in state.validation_errors)
        text += (
            f"\n\nThe previous attempt had these errors. Please address them:\n{errors}"
        )

    content_parts: list[ContentPart] = [TextContentPart(type="text", text=text)]
    for path in state.image_paths:
        content_parts.append(_load_image_part(path))

    result = cast(
        dict[str, Any],
        llm.invoke(
            [
                SystemMessage(content=INTAKE_SYSTEM_PROMPT + language_directive()),
                HumanMessage(content=cast("list[str | dict[str, Any]]", content_parts)),
            ]
        ),
    )

    parsed: IntakeOutput | None = result.get("parsed")
    if parsed is None:
        raw: BaseMessage | None = result.get("raw")
        parsing_error = result.get("parsing_error")
        raw_preview = repr(raw.content if raw is not None else raw)[:500]
        logger.error(
            "intake_node: structured output parse failed. "
            "parsing_error={} raw preview={}",
            parsing_error,
            raw_preview,
        )
        raise RuntimeError(
            "IntakeOutput parsing returned None. The LLM likely replied with "
            "text instead of a tool call — common on retry turns. "
            f"parsing_error={parsing_error!r}; raw preview: {raw_preview}"
        )

    config = state.config_state.model_copy(deep=True)
    config.building = parsed.building
    config.site_location = parsed.site_location

    return AgentStateUpdate(
        intake_output=parsed,
        config_state=config,
        validation_errors=[],
    )
