import pytest
from idfpy import IDF
from idfpy.models.advanced_construction import SurfacePropertyExposedFoundationPerimeter
from idfpy.models.location import SiteGroundTemperatureBuildingSurface
from idfpy.models.thermal_zones import (
    BuildingSurfaceDetailed,
)

from src.modeling.ground import FOUNDATION_NAME, exposed_perimeter, use_kiva_foundations

type XY = tuple[float, float]


def _surface(
    name: str,
    surface_type: str,
    boundary: str,
    points: list[tuple[float, float, float]],
) -> BuildingSurfaceDetailed:
    return BuildingSurfaceDetailed.model_validate(
        {
            "name": name,
            "surface_type": surface_type,
            "construction_name": "C",
            "zone_name": "Z",
            "outside_boundary_condition": boundary,
            "outside_boundary_condition_object": "Other"
            if boundary == "Surface"
            else None,
            "vertices": [
                {
                    "vertex_x_coordinate": x,
                    "vertex_y_coordinate": y,
                    "vertex_z_coordinate": z,
                }
                for x, y, z in points
            ],
        }
    )


def _wall(name: str, start: XY, end: XY, boundary: str = "Outdoors"):
    (x1, y1), (x2, y2) = start, end
    points = [(x1, y1, 3.0), (x1, y1, 0.0), (x2, y2, 0.0), (x2, y2, 3.0)]
    return _surface(name, "Wall", boundary, points)


def _box(north: str = "Outdoors") -> IDF:
    """10 m x 8 m zone with a ground floor and four walls."""
    idf = IDF()
    floor = [(0.0, 0.0, 0.0), (0.0, 8.0, 0.0), (10.0, 8.0, 0.0), (10.0, 0.0, 0.0)]
    for surface in (
        _surface("Floor", "Floor", "Ground", floor),
        _wall("South", (0, 0), (10, 0)),
        _wall("East", (10, 0), (10, 8)),
        _wall("North", (10, 8), (0, 8), north),
        _wall("West", (0, 8), (0, 0)),
    ):
        idf.add(surface)
    return idf


def _floor(idf: IDF) -> BuildingSurfaceDetailed:
    floor = idf.get(BuildingSurfaceDetailed, "Floor")
    assert floor is not None
    return floor


def test_perimeter_counts_edges_under_outdoor_walls():
    idf = _box()

    assert exposed_perimeter(idf, _floor(idf)) == pytest.approx(36.0)


def test_perimeter_skips_walls_shared_with_another_zone():
    idf = _box(north="Surface")

    assert exposed_perimeter(idf, _floor(idf)) == pytest.approx(26.0)


def test_perimeter_merges_overlapping_partial_walls():
    idf = _box(north="Surface")
    # Two outdoor segments along the north edge, overlapping between x=3..4.
    idf.add(_wall("North_A", (0, 8), (4, 8)))
    idf.add(_wall("North_B", (3, 8), (6, 8)))

    assert exposed_perimeter(idf, _floor(idf)) == pytest.approx(32.0)


def test_ground_floors_move_to_kiva():
    idf = _box()

    assert use_kiva_foundations(idf) == ["Floor"]
    floor = _floor(idf)
    perimeter = idf.all_of_type(SurfacePropertyExposedFoundationPerimeter)
    assert floor.outside_boundary_condition == "Foundation"
    assert floor.outside_boundary_condition_object == FOUNDATION_NAME
    assert [p.total_exposed_perimeter for p in perimeter.values()] == [36.0]
    assert use_kiva_foundations(idf) == []


def test_explicit_ground_temperature_keeps_ground_boundary():
    idf = _box()
    idf.add(SiteGroundTemperatureBuildingSurface())

    assert use_kiva_foundations(idf) == []
    assert _floor(idf).outside_boundary_condition == "Ground"
