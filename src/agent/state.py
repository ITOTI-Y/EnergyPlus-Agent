from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Final

from idfpy import IDF
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from src.agent._share import DEFAULT_OUTPUT_DIR, MAX_RETRIES, MAX_SIM_RETRIES
from src.mcp.state import ConfigState
from src.validator import (
    BuildingSchema,
    HVACSchema,
    ScheduleCollectionSchema,
    SiteLocationSchema,
)


class IntakeOutput(BaseModel):
    """Structured output from intake LLM call.

    `building` and `site_location` are nested schemas populated directly
    by the LLM's structured output, so intake_node does not need to parse
    free text a second time.

    All `*_specs` fields are natural-language task instructions passed to
    the corresponding phase agent.
    """

    building: BuildingSchema = Field(
        description="Building object (name, orientation, terrain, tolerances)"
    )
    site_location: SiteLocationSchema = Field(
        description="Site location (latitude, longitude, time zone, elevation)"
    )
    zone_specs: str = Field(
        description="Zone creation instructions: count, names, dimensions, positions"
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
        description="Surface geometry instructions referencing zones and constructions"
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


@dataclass(frozen=True)
class SimContext:
    """Immutable runtime context, passed via StateGraph context_schema."""

    epw_path: Path
    output_dir: Path = DEFAULT_OUTPUT_DIR


def _merge_upstream_request(old: dict | None, new: dict | None) -> dict | None:
    """Reducer for ``upstream_request``: a non-None back-hop request wins.

    Parallel branches (zone/material/schedule in phase 1, hvac/people/lights
    in phase 3) all return state updates simultaneously, so LangGraph needs a
    reducer to merge them. Semantics: a real back-hop request (non-None)
    always takes precedence — only one branch ever carries one per step, the
    others return None. If every branch returns None (no hop), the result is
    None. This also makes the field safe to explicitly set to None by a
    target phase to clear a consumed request.
    """
    return new if new is not None else old


def _overwrite_list(old: list[str] | None, new: list[str] | None) -> list[str]:
    """Last-write-wins reducer for full-snapshot list fields.

    ``validation_errors`` and ``simulation_errors`` are always written as a
    COMPLETE freshly-recomputed snapshot (e.g. ``validate_references()`` reruns
    the whole cross-ref check every time), never as an incremental delta. So
    any new value — even an empty list, which legitimately means "no errors" —
    supersedes the previous snapshot. Without this reducer, LangGraph's
    default ``last_value`` channel raises ``InvalidUpdateError`` when parallel
    branches (phase-3 hvac/people/lights, or simulate+revise) each return one
    of these fields in the same superstep, crashing the graph.
    """
    return new if new is not None else (old or [])


def _get_identity(item: Any) -> str:
    """Return the unique identity key for a schema item.

    Most schemas use `.name`; IdealLoadsAirSystem uses `.zone_name`
    (one system per zone).
    """
    if hasattr(item, "name"):
        return item.name
    if hasattr(item, "zone_name"):
        return item.zone_name
    if hasattr(item, "variable_name") and hasattr(item, "key_value"):
        return f"{item.key_value}_{item.variable_name}_{item.reporting_frequency}"
    raise ValueError(f"Cannot determine identity for {type(item).__name__}")


def _merge_named_list(old_items: list, new_items: list) -> list:
    """Union merge by identity key. New wins on conflict."""
    merged = {_get_identity(item): item for item in old_items}
    merged.update({_get_identity(item): item for item in new_items})
    return list(merged.values())


def _is_default(value: Any, field_name: str) -> bool:
    """Check whether `value` equals ConfigState's default for `field_name`."""
    info = ConfigState.model_fields[field_name]
    if info.default_factory is not None:
        return value == info.default_factory()
    return value is info.default or value == info.default


def _merge_schedules(
    old: ScheduleCollectionSchema,
    new: ScheduleCollectionSchema,
) -> ScheduleCollectionSchema:
    return ScheduleCollectionSchema.model_validate(
        {
            "schedule_type_limits": _merge_named_list(
                old.schedule_type_limits, new.schedule_type_limits
            ),
            "schedules": _merge_named_list(old.schedules, new.schedules),
        }
    )


def _merge_hvac(old: HVACSchema, new: HVACSchema) -> HVACSchema:
    return HVACSchema.model_validate(
        {
            "thermostats": _merge_named_list(old.thermostats, new.thermostats),
            "ideal_loads_systems": _merge_named_list(
                old.ideal_loads_systems, new.ideal_loads_systems
            ),
        }
    )


def _idf_has_objects(cs: ConfigState) -> bool:
    """True if the ConfigState's backing IDF contains any typed objects.

    The Pydantic legacy fields can be empty even when ``_idf`` holds the
    real model (e.g. after ``load_idf``), so this is the reliable way to
    tell whether *cs* actually carries a model.
    """
    idf = cs._idf
    if idf is None:
        return False
    try:
        return any(idf.all_of_type(t) for t in (
            "Zone", "Material", "Material:NoMass", "Material:AirGap",
            "WindowMaterial:SimpleGlazingSystem", "Construction",
            "BuildingSurface:Detailed", "FenestrationSurface:Detailed",
            "Schedule:Compact", "ScheduleTypeLimits",
            "HVACTemplate:Thermostat", "HVACTemplate:Zone:IdealLoadsAirSystem",
            "People", "Lights",
        ))
    except Exception:
        return False


def _merge_idf(old: ConfigState, new: ConfigState) -> Any:
    """Merge two ConfigStates' backing IDFs by object name (new wins).

    Returns a fresh idfpy IDF whose objects are the name-keyed union of
    ``old._idf`` and ``new._idf``. Used by :func:`merge_config_state` so
    that the IDF — the real source of truth, especially on revision
    turns where the model was loaded from a saved file — survives the
    parallel-node merge instead of being reset to an empty IDF by
    ``model_post_init``.

    Returns ``None`` if neither side has an IDF (falls back to the
    default empty IDF created by ConfigState's constructor).
    """
    from idfpy import IDF

    old_d = old._idf.to_dict() if _idf_has_objects(old) else {}
    new_d = new._idf.to_dict() if _idf_has_objects(new) else {}
    if not old_d and not new_d:
        return None

    merged: dict[str, dict] = {}
    for obj_type in set(old_d) | set(new_d):
        # Each value is {name: fields}; new wins on name conflict.
        bucket = dict(old_d.get(obj_type, {}))
        bucket.update(new_d.get(obj_type, {}))
        merged[obj_type] = bucket
    return IDF.from_dict(merged)


_NAMED_LIST_FIELDS: Final = (
    "zones",
    "materials",
    "constructions",
    "surfaces",
    "fenestrations",
    "people",
    "lights",
    "output_variable",
)

_SINGLETON_FIELDS: Final = (
    "building",
    "site_location",
    "simulation_control",
    "global_geometry_rules",
    "run_period",
    "output_variable_dictionary",
    "output_diagnostics",
    "output_table_summary_reports",
    "output_control_table_style",
)


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
    """Union merge for parallel-safe state updates; the IDF is authoritative.

    The idfpy IDF containers are merged by (object type, identity) with new
    winning on conflict. The legacy Pydantic fields are still merged
    alongside because intake writes building/site_location into them and
    `save_idf` injects their defaults via `sync_legacy_fields_to_idf`:
    1. Named list fields -> union by identity key; new wins on conflict
    2. Nested containers (schedules, hvac) -> recursive merge
    3. Singleton objects -> non-default wins, new preferred

    IDF merge: the backing idfpy IDF is the authoritative store,
    especially on revision turns where the model was loaded from a
    previously-saved IDF (the Pydantic legacy fields stay empty in that
    case). We union the two IDFs by object name (new wins) and install
    the result as the merged ConfigState's ``_idf`` — otherwise
    ``model_validate`` would reset it to an empty IDF via
    ``model_post_init``, silently dropping the entire model.
    """
    data: dict[str, Any] = {}

    for field_name in _NAMED_LIST_FIELDS:
        data[field_name] = _merge_named_list(
            getattr(old, field_name), getattr(new, field_name)
        )

    data["schedules"] = _merge_schedules(old.schedules, new.schedules)
    data["hvac"] = _merge_hvac(old.hvac, new.hvac)

    for field_name in _SINGLETON_FIELDS:
        new_val = getattr(new, field_name)
        old_val = getattr(old, field_name)
        data[field_name] = new_val if not _is_default(new_val, field_name) else old_val

    merged = ConfigState.model_validate(data)
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

    validation_errors: Annotated[list[str], _overwrite_list] = Field(
        default_factory=list
    )
    retry_count: int = 0
    max_retries: int = MAX_RETRIES

    # --- EnergyPlus simulation failure -> revise rollback loop ---
    # Independent from retry_count (which gates validate's cross-ref
    # rollback) so the two loops don't starve each other's budget.
    simulation_errors: Annotated[list[str], _overwrite_list] = Field(
        default_factory=list
    )
    """Fatal/Severe error lines from eplusout.err of the last simulate run.
    Populated by simulate_node on failure; consumed (and cleared) by
    revise_node so the LLM gets concrete error text to fix."""
    sim_retry_count: int = 0
    """How many times simulate has rolled back to revise for a sim failure."""
    max_sim_retries: int = MAX_SIM_RETRIES
    """Cap on simulate->revise rollback rounds. Once exhausted, simulate
    lets the run fall through to analyze (the test harness records failure)."""

    is_revision: bool = False
    """True for multi-turn model edits: the agent should modify the existing
    config_state (loaded from a previous IDF) rather than rebuild from
    scratch. Drives the revise_node entry and phase-agent prompt prefixes."""

    upstream_request: Annotated[dict | None, _merge_upstream_request] = None
    """Back-hop request set by a phase node when it detects that a needed
    upstream object does not exist (e.g. fenestration needs a window
    construction that was never created). Shape:
    ``{"target": <phase name>, "specs": <instruction string>}``. The target
    phase reads and clears this. ``None`` = no back-hop pending. Uses a
    custom reducer so parallel branches can each return the field safely."""

    hop_count: Annotated[int, lambda o, n: max(o, n)] = 0
    """Back-hop counter to prevent infinite A->B->A loops. Incremented on
    each ``Command(goto=<earlier phase>)`` back-hop; phase nodes refuse to
    hop once it reaches HOP_LIMIT."""


class AgentStateUpdate(TypedDict, total=False):
    """Partial update returned by graph nodes."""

    messages: Sequence[AnyMessage]
    user_input: str
    image_paths: list[str]
    config_state: ConfigState
    intake_output: IntakeOutput | None
    validation_errors: list[str]
    retry_count: int
    simulation_errors: list[str]
    sim_retry_count: int
    is_revision: bool
    upstream_request: dict | None
    hop_count: int
