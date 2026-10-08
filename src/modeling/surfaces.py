"""Base surfaces, including pairs of surfaces shared by two zones.

An interzone surface names its twin in the adjacent zone, and the twin names
it back. Neither can be created first if a reference must already exist, so
the partner of a ``Surface`` boundary is allowed to follow; creating the
twin, or naming an existing surface, completes the pair.
"""

from typing import Final

from idfpy import IDF
from idfpy.models.thermal_zones import BuildingSurfaceDetailed

from src.modeling import objects
from src.modeling.envelope import check_construction_fits

AREA_TOLERANCE: Final = 0.01
"""Relative area difference allowed between the two faces of a pair."""


def pair_problem(
    surface: BuildingSurfaceDetailed, partner: BuildingSurfaceDetailed
) -> str | None:
    """Why two surfaces cannot be the two faces of one interzone element."""
    if surface.zone_name == partner.zone_name:
        return f"'{partner.name}' is in the same zone"
    (ax, ay, az), (bx, by, bz) = surface.normal, partner.normal
    if ax * bx + ay * by + az * bz > -0.99:
        return f"'{partner.name}' does not face the opposite way"
    if abs(surface.area - partner.area) > AREA_TOLERANCE * max(surface.area, 1e-9):
        return (
            f"'{partner.name}' has area {partner.area:.2f} m2, "
            f"not {surface.area:.2f} m2"
        )
    return None


def add_surface(
    idf: IDF, surface: BuildingSurfaceDetailed
) -> list[BuildingSurfaceDetailed]:
    """Add a base surface; an existing ``Surface`` partner is linked back.

    Returns:
        The new surface, followed by its partner when that was relinked.

    Raises:
        ValueError: If the construction does not suit the surface type, or
            the existing partner cannot be its twin.
        ModelingError: On a duplicate name or other missing references.
    """
    check_construction_fits(idf, surface.construction_name, surface.surface_type)
    if surface.outside_boundary_condition != "Surface":
        return [objects.create(idf, surface)]

    partner_name = surface.outside_boundary_condition_object or ""
    partner = idf.get(BuildingSurfaceDetailed, partner_name)
    if partner is not None:
        if problem := pair_problem(surface, partner):
            raise ValueError(f"Surface '{surface.name}': {problem}")
        bound_to = partner.outside_boundary_condition_object
        if partner.outside_boundary_condition == "Surface" and bound_to not in (
            None,
            surface.name,
        ):
            raise ValueError(
                f"Surface '{surface.name}': '{partner_name}' is already paired "
                f"with '{bound_to}'"
            )
    # The partner may not exist yet; it is checked when the pair is complete.
    created = objects.create(
        idf, surface.model_copy(update={"outside_boundary_condition_object": None})
    )
    created.outside_boundary_condition_object = partner_name
    if partner is None:
        return [created]
    partner.outside_boundary_condition = "Surface"
    partner.outside_boundary_condition_object = created.name
    partner.sun_exposure = "NoSun"
    partner.wind_exposure = "NoWind"
    return [created, partner]
