"""Internal loads: the one level field each calculation method reads."""

from typing import Final

LEVEL_FIELDS: Final[dict[str, str]] = {
    "People": "number_of_people",
    "People/Area": "people_per_floor_area",
    "Area/Person": "floor_area_per_person",
    "LightingLevel": "lighting_level",
    "EquipmentLevel": "design_level",
    "Watts/Area": "watts_per_floor_area",
    "Watts/Person": "watts_per_person",
}
"""The field each design level calculation method reads."""


def require_level(method: str, **values: float | None) -> None:
    """Check that the field ``method`` reads has a value.

    EnergyPlus ignores the other level fields, and a missing value would
    model no load, so a load without it is refused.

    Raises:
        ValueError: If the field ``method`` reads has no value.
    """
    field = LEVEL_FIELDS[method]
    if values.get(field) is None:
        raise ValueError(f"{method} needs {field}")
