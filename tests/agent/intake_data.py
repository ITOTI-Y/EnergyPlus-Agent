"""Intake outputs for tests, so each test states only the fields it uses."""

from collections.abc import Sequence
from typing import Any

from src.agent.state import SPEC_PHASES, IntakeOutput

_LAYOUT_FIELDS = ("zone_plans", "storeys")


def layout(*zones: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """``zone_plans`` and ``storeys`` holding the given zones.

    Each zone gets its own plan keyed by its name, on an unnamed storey per
    level, so the zone keeps its name.
    """
    plans = [
        {k: v for k, v in z.items() if k not in ("floor_z", "height", "multiplier")}
        | {"key": z["name"]}
        for z in zones
    ]
    storeys: dict[tuple[float, float, int], list[dict[str, str]]] = {}
    for z in zones:
        level = (z["floor_z"], z["height"], z.get("multiplier", 1))
        storeys.setdefault(level, []).append({"plan": z["name"]})
    for plan in plans:
        del plan["name"]
    return {
        "zone_plans": plans,
        "storeys": [
            {"name": "", "floor_z": z, "height": h, "multiplier": m, "zones": entries}
            for (z, h, m), entries in storeys.items()
        ],
    }


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
