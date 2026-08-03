"""Revision-mode entry node for multi-turn model editing.

Unlike :mod:`intake` (which builds a model from scratch), :func:`revise_node`
starts from an *existing* ``config_state`` (loaded from a previous IDF by the
UI) and asks the LLM to produce *modification instructions* that downstream
phase agents will apply via ``update_*`` / ``delete_*`` / ``create_*`` tools.

Key differences from intake:
- The config_state is NOT reset — phase agents will see the existing objects
  and (prompted by REVISION_PREFIX) prefer update/delete over create.
- The LLM is given a compact inventory of existing objects so it can refer to
  them by exact name.
- ``building`` / ``site_location`` are preserved from the prior run; only the
  ``*_specs`` fields carry modification instructions.
"""

from __future__ import annotations

import time
from typing import Any, cast

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from src.agent._share import language_directive
from src.agent.llm import create_llm
from src.agent.nodes.intake import INTAKE_MAX_EMPTY_RETRIES
from src.agent.state import AgentState, AgentStateUpdate, IntakeOutput

REVISE_SYSTEM_PROMPT = """You are an EnergyPlus model-revision specialist.
The user has ALREADY built a complete model and run a simulation. Now they
want to MODIFY it based on the results. You receive:

1. An inventory of every object already in the model (zones, materials,
   constructions, surfaces, fenestrations, schedules, HVAC, people, lights).
2. The user's new modification instruction.

Your job: decide which subsystems need changes and write CONCRETE
modification instructions into the ``*_specs`` fields. Respond as a single
raw JSON object (no markdown, no code fences) with the SAME schema as the
initial intake — ALL fields are required, but for subsystems that need no
change, write the single word "no changes needed".

Rules:
1. Refer to existing objects by their EXACT name from the inventory.
2. For each subsystem that DOES need changes, describe the modification
   precisely — name the target object and the new value:
     ✓ "Update material Brick_100mm: change thickness from 0.1 to 0.15"
     ✓ "Update fenestration South_Window: set multiplier to 2 and switch
        construction to LowE_Window_Construction"
     ✓ "Create new material LowE_Glazing: U-factor 1.2, SHGC 0.3, VT 0.7"
     ✗ "make the walls better"  (vague)
3. For subsystems with NO changes, write exactly: no changes needed
4. If the user requests something that needs a NEW object (e.g. a new
   material that doesn't exist yet), include it in the relevant *_specs as a
   "Create" instruction.
5. Preserve name-format rules: ONLY word characters + underscores, no spaces
   or punctuation.
6. building / site_location: echo the values from the inventory unchanged
   (they are passed through as-is).
7. Internal consistency still applies — any new name you introduce must be
   reused verbatim across subsystems.
8. If a "## EnergyPlus simulation errors to fix" section is present, the
   previous simulation FAILED — your PRIMARY job is to fix those errors,
   not to honor stylistic edits. Common geometry fixes:
   - "Outward facing angle of subsurface differs >90 degrees from base
     surface" => the fenestration vertices are wound opposite to their host
     wall. Fix by reversing the vertex order (or correcting the window's
     construction/placement) in fenestration_specs.
   - "X not found" / "does not exist" => a referenced object name is wrong
     or missing; create or rename it in the owning *_specs.
   Give a CONCRETE, named fix in the relevant *_specs; leave unrelated
   subsystems as "no changes needed".
"""


# idfpy object type → (display label, fields to summarize)
# Only name + 1-2 key properties are shown to keep the LLM context compact.
_SUMMARY_SPECS: list[tuple[str, str, tuple[str, ...]]] = [
    ("Zone", "zone", ()),
    ("Material", "material", ("thickness", "conductivity")),
    ("Material:NoMass", "material", ("thermal_resistance",)),
    ("Material:AirGap", "material", ("thermal_resistance",)),
    (
        "WindowMaterial:SimpleGlazingSystem",
        "glazing",
        ("u_factor", "solar_heat_gain_coefficient", "visible_transmittance"),
    ),
    ("Construction", "construction", ()),
    (
        "BuildingSurface:Detailed",
        "surface",
        ("surface_type", "construction_name", "zone_name"),
    ),
    (
        "FenestrationSurface:Detailed",
        "fenestration",
        ("surface_type", "construction_name", "building_surface_name", "multiplier"),
    ),
    ("Schedule:Compact", "schedule", ("schedule_type_limits_name",)),
    ("ScheduleTypeLimits", "schedule_type_limits", ("numeric_type",)),
    (
        "HVACTemplate:Thermostat",
        "thermostat",
        ("heating_setpoint_schedule_name", "cooling_setpoint_schedule_name"),
    ),
    (
        "HVACTemplate:Zone:IdealLoadsAirSystem",
        "ideal_loads",
        ("zone_name", "template_thermostat_name"),
    ),
    ("People", "people", ("zone_or_zonelist_or_space_or_spacelist_name",)),
    ("Lights", "lights", ("zone_or_zonelist_or_space_or_spacelist_name",)),
]


def _summarize_config_for_llm(state: AgentState) -> str:
    """Render a compact inventory of existing objects for the revision LLM.

    Only object names + a few key properties are shown (not full model
    dumps) to stay within token limits on large models.
    """
    idf = state.config_state._idf
    if idf is None:
        raise ValueError("IDF is None")
    lines: list[str] = ["## Existing model inventory (modify these — do NOT rebuild):"]
    for obj_type, _label, fields in _SUMMARY_SPECS:
        items = idf.all_of_type(obj_type)
        if not items:
            continue
        lines.append(f"\n### {obj_type} ({len(items)})")
        for obj in items.values():
            props = {f: getattr(obj, f, None) for f in fields}
            # Filter out None/empty props for compactness
            shown = ", ".join(
                f"{k}={v}" for k, v in props.items() if v not in (None, "", [])
            )
            name = getattr(obj, "name", None) or getattr(obj, "zone_name", obj_type)
            lines.append(f"  - {name}" + (f"  ({shown})" if shown else ""))
    return "\n".join(lines)


def _echo_building_and_site(state: AgentState) -> tuple[Any, Any]:
    """Return the existing building/site_location so the LLM output preserves them."""
    cs = state.config_state
    return cs.building, cs.site_location


def revise_node(state: AgentState) -> AgentStateUpdate:
    """Revision entry: parse the user's modification instruction against the
    existing model inventory and emit ``*_specs`` modification instructions.

    The ``config_state`` is returned UNCHANGED — phase agents will operate on
    it and see existing objects. Only ``intake_output`` is refreshed with the
    new modification specs.

    Note on seed-model recovery: LangGraph's input coercion at the graph
    START strips ConfigState's PrivateAttr ``_idf`` (the loaded seed model)
    before this node runs. The seed IDF is carried as a declared
    ``seed_idf_text`` field on ConfigState (which survives coercion), and
    ``recover_idf_from_seed()`` rebuilds ``_idf`` from it.
    """
    state.config_state.recover_idf_from_seed()

    llm = create_llm().with_structured_output(
        IntakeOutput, method="function_calling", include_raw=True
    )

    inventory = _summarize_config_for_llm(state)

    # If we got here via a simulate failure (not a user edit request),
    # surface the EnergyPlus errors so the LLM writes fix instructions.
    sim_errors = state.simulation_errors
    sections = [inventory]
    if sim_errors:
        err_text = "\n".join(f"  - {e}" for e in sim_errors)
        sections.append(
            "## EnergyPlus simulation errors to fix:\n"
            "The previous simulation FAILED with these errors. Write concrete "
            "fix instructions into the relevant *_specs (e.g. fix surface "
            "vertex winding / window orientation / missing objects):\n"
            f"{err_text}"
        )
    sections.append(f"## User's modification request:\n{state.user_input}")
    text = "\n\n".join(sections)

    messages = [
        SystemMessage(content=REVISE_SYSTEM_PROMPT + language_directive()),
        HumanMessage(content=text),
    ]

    # Retry on empty LLM replies (same gateway transient as intake_node).
    result: dict[str, Any] = {}
    parsed: IntakeOutput | None = None
    for attempt in range(INTAKE_MAX_EMPTY_RETRIES + 1):
        result = cast(dict[str, Any], llm.invoke(messages))
        parsed = result.get("parsed")
        if parsed is not None:
            break
        raw = result.get("raw")
        parsing_error = result.get("parsing_error")
        raw_preview = repr(raw.content if raw is not None else raw)[:500]
        is_empty = parsing_error is None and raw_preview in ("''", "None")
        if not is_empty or attempt == INTAKE_MAX_EMPTY_RETRIES:
            logger.error(
                "revise_node: structured output parse failed. "
                "parsing_error={} raw preview={}",
                parsing_error,
                raw_preview,
            )
            raise RuntimeError(
                "IntakeOutput parsing returned None in revise_node. "
                f"parsing_error={parsing_error!r}; raw preview: {raw_preview}"
            )
        sleep_s = 2 ** (attempt + 1)
        logger.warning(
            "revise_node: empty LLM reply (attempt {}/{}), retrying in {}s",
            attempt + 1,
            INTAKE_MAX_EMPTY_RETRIES + 1,
            sleep_s,
        )
        time.sleep(sleep_s)

    # Preserve building/site_location from the existing model (the LLM may
    # echo them, but we force the authoritative values to avoid drift).
    building, site_location = _echo_building_and_site(state)
    config = state.config_state.clone()
    config.building = building
    config.site_location = site_location

    # Preserve incoming validation_errors: when simulate rolled back here
    # due to an EnergyPlus failure, it pushed the error text into
    # validation_errors so downstream phase agents (surface/fenestration/...)
    # see the fix instructions via invoke_with_self_repair. Clearing them
    # here would drop that signal. We DO clear simulation_errors (consumed
    # above) to avoid re-injecting on the next revise pass.
    return AgentStateUpdate(
        intake_output=parsed,
        config_state=config,
        validation_errors=list(state.validation_errors),
        simulation_errors=[],
        is_revision=True,
    )
