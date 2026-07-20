from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any, Final, Literal, TypedDict, cast

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from loguru import logger

from src.agent._share import language_directive
from src.agent.llm import create_llm
from src.agent.state import AgentState, AgentStateUpdate, IntakeOutput

INTAKE_MAX_EMPTY_RETRIES: Final[int] = 3
"""Max retries when the LLM returns an empty reply (no content, no tool call).

Some OpenAI-compatible gateways occasionally return a completely empty
AIMessage (content='' and no tool_calls) — most often on transient
gateway hiccups (rate-limit, timeout, upstream blip). The structured-output
wrapper then yields ``parsed=None, parsing_error=None``. Retrying with
backoff almost always recovers, so we don't let a single empty reply
crash the whole case. We deliberately keep this budget small: if the
empty reply is caused by the prompt itself, retrying won't help and the
outer harness will record the failure."""


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

You MUST respond with a single raw JSON object (no markdown, no code fences).
ALL of the following fields are REQUIRED — omitting any field is an error:
- `building`: object with name, terrain, tolerance_for_heating_sizing, tolerance_for_cooling_sizing
- `site_location`: object with name, latitude, longitude, time_zone, elevation
- `zone_specs`: string — zone creation instructions
- `material_specs`: string — material definitions with thermal properties (REQUIRED, do NOT merge into construction_specs)
- `schedule_specs`: string — schedule definitions
- `construction_specs`: string — construction assemblies referencing materials from material_specs
- `surface_specs`: string — surface geometry referencing zones and constructions
- `fenestration_specs`: string — window/door instructions referencing surfaces
- `hvac_specs`: string — HVAC system, thermostat setpoints, schedule references
- `people_specs`: string — occupancy load per zone
- `lights_specs`: string — lighting load per zone

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
    llm = create_llm().with_structured_output(
        IntakeOutput, method="function_calling", include_raw=True
    )

    text = state.user_input
    if state.validation_errors:
        errors = "\n".join(f"- {e}" for e in state.validation_errors)
        text += (
            f"\n\nThe previous attempt had these errors. Please address them:\n{errors}"
        )

    content_parts: list[ContentPart] = [TextContentPart(type="text", text=text)]
    for path in state.image_paths:
        content_parts.append(_load_image_part(path))

    messages = [
        SystemMessage(content=INTAKE_SYSTEM_PROMPT + language_directive()),
        HumanMessage(content=cast("list[str | dict[str, Any]]", content_parts)),
    ]

    # Retry on empty LLM replies. Some gateways transiently return an
    # AIMessage with no content and no tool_calls (parsed=None,
    # parsing_error=None); a short backoff retry usually recovers.
    parsed: IntakeOutput | None = None
    for attempt in range(INTAKE_MAX_EMPTY_RETRIES + 1):
        result = cast(dict[str, Any], llm.invoke(messages))
        parsed = result.get("parsed")
        if parsed is not None:
            break
        # Empty or malformed reply — inspect to decide retry vs. fail.
        raw: BaseMessage | None = result.get("raw")
        parsing_error = result.get("parsing_error")
        raw_preview = repr(raw.content if raw is not None else raw)[:500]
        is_empty = parsing_error is None and raw_preview in ("''", "None")
        if not is_empty or attempt == INTAKE_MAX_EMPTY_RETRIES:
            logger.error(
                "intake_node: structured output parse failed. "
                "parsing_error={} raw preview={}",
                parsing_error,
                raw_preview,
            )
            raise RuntimeError(
                "IntakeOutput parsing returned None. The LLM likely replied "
                "with text instead of a tool call, or returned an empty "
                "reply. parsing_error="
                f"{parsing_error!r}; raw preview: {raw_preview}"
            )
        # Back off (2s, 4s) and retry — transient gateway empty-reply.
        sleep_s = 2 ** (attempt + 1)
        logger.warning(
            "intake_node: empty LLM reply (attempt {}/{}), retrying in {}s",
            attempt + 1,
            INTAKE_MAX_EMPTY_RETRIES + 1,
            sleep_s,
        )
        time.sleep(sleep_s)

    assert parsed is not None
    config = state.config_state.clone()
    config.building = parsed.building
    config.site_location = parsed.site_location

    return AgentStateUpdate(
        intake_output=parsed,
        config_state=config,
        validation_errors=[],
    )
