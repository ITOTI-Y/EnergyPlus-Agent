from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Final, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from src.agent._share import DEFAULT_OUTPUT_DIR, MAX_RETRIES
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


def merge_config_state(old: ConfigState, new: ConfigState) -> ConfigState:
    """Field-level union merge for parallel-safe state updates.

    Three strategies:
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

    # Install the unioned IDF as the source of truth. This MUST happen
    # after model_validate (which would otherwise clobber _idf with a
    # fresh empty IDF via model_post_init).
    merged_idf = _merge_idf(old, new)
    if merged_idf is not None:
        # Assign via the PrivateAttr dict: BaseSchema's validate_assignment
        # + class-level _idf annotation silently break a plain assignment,
        # and object.__setattr__ would break pickling.
        merged._set_idf_private(merged_idf)
    elif not merged._has_idf_objects():
        # Fallback: if neither side carried IDF objects (e.g. revision
        # turn where START-coercion stripped _idf), rebuild from the
        # seed_idf_text that survived as a declared field. This is the
        # authoritative recovery path for revision-turn seed models.
        seed_text = new.seed_idf_text or old.seed_idf_text
        if seed_text:
            merged._set_idf_private(_idf_from_text(seed_text))
            # Propagate so downstream merges keep recovering until a
            # phase node populates _idf with real objects.
            merged.seed_idf_text = seed_text

    return merged


def _idf_from_text(idf_text: str) -> Any:
    """Rebuild an idfpy IDF from saved IDF text (used by merge_config_state
    to recover the seed model on revision turns)."""
    from idfpy import IDF

    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".idf", delete=False
    ) as tf:
        tf.write(idf_text)
        loaded = IDF.load(Path(tf.name))
        Path(tf.name).unlink()
        return IDF.from_dict(loaded.to_dict())


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

    validation_errors: list[str] = Field(default_factory=list)
    retry_count: int = 0
    max_retries: int = MAX_RETRIES

    is_revision: bool = False
    """True for multi-turn model edits: the agent should modify the existing
    config_state (loaded from a previous IDF) rather than rebuild from
    scratch. Drives the revise_node entry and phase-agent prompt prefixes."""


class AgentStateUpdate(TypedDict, total=False):
    """Partial update returned by graph nodes."""

    messages: Sequence[AnyMessage]
    user_input: str
    image_paths: list[str]
    config_state: ConfigState
    intake_output: IntakeOutput | None
    validation_errors: list[str]
    retry_count: int
    is_revision: bool
