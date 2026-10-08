import json
from pathlib import Path

import pytest
from idfpy import IDF
from idfpy.models.thermal_zones import BuildingSurfaceDetailed

from src.modeling.surfaces import add_surface
from src.modeling.validation import interzone_issues

DATA = Path(__file__).parents[2] / "data"
EAST, WEST = "Zone_East_Wall_Internal", "Zone_West_Wall_Internal"


def _model_without_shared_wall() -> tuple[
    IDF, BuildingSurfaceDetailed, BuildingSurfaceDetailed
]:
    """building_schema with its interzone wall pair taken out, unbound."""
    idf = IDF.from_dict(
        json.loads((DATA / "schemas" / "building_schema.epJSON").read_text())
    )
    east, west = (idf.get(BuildingSurfaceDetailed, n) for n in (EAST, WEST))
    assert east is not None and west is not None
    for surface in (east, west):
        surface.outside_boundary_condition_object = None
        surface.outside_boundary_condition = "Adiabatic"
    for name in (EAST, WEST):
        idf.remove(BuildingSurfaceDetailed, name)
    return idf, east, west


def _paired(surface: BuildingSurfaceDetailed, partner: str) -> BuildingSurfaceDetailed:
    return surface.model_copy(
        update={
            "outside_boundary_condition": "Surface",
            "outside_boundary_condition_object": partner,
        }
    )


def _pair_of(idf: IDF, name: str) -> tuple[str, str | None]:
    surface = idf.get(BuildingSurfaceDetailed, name)
    assert surface is not None
    return surface.outside_boundary_condition, surface.outside_boundary_condition_object


def test_pair_can_be_created_one_surface_at_a_time():
    idf, east, west = _model_without_shared_wall()

    add_surface(idf, _paired(east, WEST))
    add_surface(idf, _paired(west, EAST))

    assert _pair_of(idf, EAST) == ("Surface", WEST)
    assert _pair_of(idf, WEST) == ("Surface", EAST)
    assert interzone_issues(idf) == []


def test_existing_surface_is_linked_back():
    idf, east, west = _model_without_shared_wall()
    add_surface(idf, east)

    created = add_surface(idf, _paired(west, EAST))

    assert [s.name for s in created] == [WEST, EAST]
    assert _pair_of(idf, EAST) == ("Surface", WEST)


def test_partner_must_face_the_opposite_way():
    idf, east, west = _model_without_shared_wall()
    add_surface(idf, east)
    south = idf.get(BuildingSurfaceDetailed, "Zone_West_Wall_South")
    assert south is not None

    with pytest.raises(ValueError, match="opposite way"):
        add_surface(
            idf,
            _paired(west, "Zone_West_Wall_South").model_copy(
                update={"zone_name": "Zone_East"}
            ),
        )


def test_one_sided_pair_is_reported():
    idf, east, west = _model_without_shared_wall()
    add_surface(idf, east)
    add_surface(idf, west)
    surface = idf.get(BuildingSurfaceDetailed, WEST)
    assert surface is not None
    surface.outside_boundary_condition = "Surface"
    surface.outside_boundary_condition_object = EAST

    [issue] = interzone_issues(idf)

    assert issue.object_name == WEST
    assert "does not name it back" in issue.message
