"""Intake outputs for tests, so each test states only the fields it uses."""

from collections.abc import Sequence
from typing import Any

from src.agent.state import SPEC_PHASES, IntakeOutput

_LAYOUT_FIELDS = ("zone_plans", "storeys")


def layout(*zones: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """``zone_plans`` and ``storeys`` holding the given ground-level zones.

    Each zone gets its own plan keyed by its name, all on one unnamed
    storey, so every zone keeps its name.
    """
    assert all(z["floor_z"] == 0 for z in zones), "zones must be on the ground"
    plans = [
        {k: v for k, v in z.items() if k not in ("name", "floor_z", "height")}
        | {"key": z["name"]}
        for z in zones
    ]
    entries = [{"plan": z["name"], "height": z["height"]} for z in zones]
    storeys = [{"name": "", "height": 3.0, "zones": entries}] if zones else []
    return {"zone_plans": plans, "storeys": storeys}


def intake(zones: Sequence[dict[str, Any]] = (), **fields: Any) -> IntakeOutput:
    """An intake output with the given zones, empty specs and other fields."""
    empty = {field: "" for field in SPEC_PHASES if field not in _LAYOUT_FIELDS}
    return IntakeOutput.model_validate(
        {
            "building": {"name": "B"},
            "site_location": {"name": "S"},
            **layout(*zones),
            **empty,
            **fields,
        }
    )


def zone_spec(
    name: str, plan: Sequence[tuple[float, float]], **fields: Any
) -> dict[str, Any]:
    """A zone on the ground using the 'Wall' and 'Slab' constructions."""
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
