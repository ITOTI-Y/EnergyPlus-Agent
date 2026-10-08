from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from idfpy import IDF
from idfpy.models.location import SiteLocation
from idfpy.models.simulation import Building
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from src.agent._share import DEFAULT_OUTPUT_DIR, MAX_RETRIES
from src.modeling.validation import ModelIssue
from src.state.config_state import ConfigState


class IntakeOutput(BaseModel):
    """Structured output from intake LLM call.

    `building` and `site_location` are idfpy models populated directly by
    the LLM's structured output; intake_node adds them to the IDF as is.

    All `*_specs` fields are natural-language task instructions passed to
    the corresponding phase agent.
    """

    building: Building = Field(
        description="Building object (name, orientation, terrain, tolerances)"
    )
    site_location: SiteLocation = Field(
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
        description=(
            "Per zone: floor plan corners (X, Y in meters, in order around the "
            "zone), floor level and height, and the constructions of exterior "
            "walls, roof, ground floor, interior walls and interior floors. "
            "Describe individual surfaces only for sloped roofs or walls."
        )
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
    equipment_specs: str = Field(
        description="Plug loads (ElectricEquipment): zone assignment, power "
        "density, schedule per zone; empty if the brief has none"
    )


@dataclass(frozen=True)
class SimContext:
    """Immutable runtime context, passed via StateGraph context_schema."""

    epw_path: Path
    output_dir: Path = DEFAULT_OUTPUT_DIR


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

    validation_errors: list[ModelIssue] = Field(default_factory=list)
    retry_count: int = 0
    max_retries: int = MAX_RETRIES


class AgentStateUpdate(TypedDict, total=False):
    """Partial update returned by graph nodes."""

    messages: Sequence[AnyMessage]
    user_input: str
    image_paths: list[str]
    config_state: ConfigState
    intake_output: IntakeOutput | None
    validation_errors: list[ModelIssue]
    retry_count: int
