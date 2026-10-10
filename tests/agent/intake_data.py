"""Intake outputs for tests, so each test states only the fields it uses."""

from collections.abc import Sequence
from typing import Any

from src.agent.state import SPEC_PHASES, IntakeOutput


def intake(**fields: Any) -> IntakeOutput:
    """An intake output with no zones and empty specs, plus the given fields."""
    empty = {field: "" for field in SPEC_PHASES if field != "zones"}
    return IntakeOutput.model_validate(
        {
            "building": {"name": "B"},
            "site_location": {"name": "S"},
            "zones": [],
            **empty,
            **fields,
        }
    )


def zone_spec(
    name: str, plan: Sequence[tuple[float, float]], **fields: Any
) -> dict[str, Any]:
    """A zone entry using the 'Wall' and 'Slab' constructions."""
    return {
        "name": name,
        "plan": [{"X": x, "Y": y} for x, y in plan],
        "floor_z": 0.0,
        "height": 3.0,
        "exterior_wall_construction": "Wall",
        "roof_construction": "Wall",
        "ground_floor_construction": "Slab",
        "interior_wall_construction": "Wall",
        "interior_floor_construction": "Slab",
        **fields,
    }
