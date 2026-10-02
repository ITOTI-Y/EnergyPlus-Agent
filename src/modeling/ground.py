"""Model slab-on-grade floors with Kiva instead of a fixed ground temperature.

A floor with outside boundary ``Ground`` sees ``Site:GroundTemperature:
BuildingSurface``, 18 C all year by default. The undisturbed ground
temperatures in a weather file are no substitute: EnergyPlus documents them
as too extreme for soil under a conditioned building. Kiva instead solves
heat flow in the soil around each floor from the weather file, so it needs
only the length of the floor edge that faces outdoors.
"""

from collections.abc import Iterator
from typing import Final

from idfpy import IDF
from idfpy.models.advanced_construction import (
    FoundationKiva,
    SurfacePropertyExposedFoundationPerimeter,
)
from idfpy.models.location import SiteGroundTemperatureBuildingSurface
from idfpy.models.thermal_zones import BuildingSurfaceDetailed

FOUNDATION_NAME: Final = "Slab_On_Grade"
TOLERANCE_M: Final = 1e-3

type Point = tuple[float, float, float]
type Segment = tuple[Point, Point]


def _points(surface: BuildingSurfaceDetailed) -> list[Point]:
    return [
        (v.vertex_x_coordinate, v.vertex_y_coordinate, v.vertex_z_coordinate)
        for v in surface.vertices or []
    ]


def _edges(points: list[Point]) -> Iterator[Segment]:
    yield from zip(points, points[1:] + points[:1], strict=True)


def _overlap(edge: Segment, other: Segment) -> tuple[float, float] | None:
    """Parameter interval of ``edge`` covered by a collinear ``other``."""
    (ax, ay, _), (bx, by, _) = edge
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    params = []
    for px, py, _ in other:
        # Distance of the point from the edge's line, then its position along it.
        if abs(dx * (py - ay) - dy * (px - ax)) > TOLERANCE_M * length_sq**0.5:
            return None
        params.append(((px - ax) * dx + (py - ay) * dy) / length_sq)
    start, end = max(min(params), 0.0), min(max(params), 1.0)
    return (start, end) if end > start else None


def _covered_fraction(intervals: list[tuple[float, float]]) -> float:
    covered, reach = 0.0, 0.0
    for start, end in sorted(intervals):
        start = max(start, reach)
        if end > start:
            covered += end - start
            reach = end
    return covered


def exposed_perimeter(idf: IDF, floor: BuildingSurfaceDetailed) -> float:
    """Length of floor edges lying under outdoor walls of the same zone, in m."""
    floor_points = _points(floor)
    level = min(z for *_, z in floor_points)
    wall_bottoms = [
        edge
        for wall in idf.all_of_type(BuildingSurfaceDetailed).values()
        if wall.surface_type == "Wall"
        and wall.zone_name == floor.zone_name
        and wall.outside_boundary_condition == "Outdoors"
        for edge in _edges(_points(wall))
        if all(abs(z - level) <= TOLERANCE_M for *_, z in edge)
    ]
    total = 0.0
    for edge in _edges(floor_points):
        (ax, ay, _), (bx, by, _) = edge
        length = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
        if length <= TOLERANCE_M:
            continue
        covered = [o for w in wall_bottoms if (o := _overlap(edge, w)) is not None]
        total += length * _covered_fraction(covered)
    return total


def use_kiva_foundations(idf: IDF) -> list[str]:
    """Move ground-contact floors onto one uninsulated Kiva slab foundation.

    Leaves the model alone when it sets its own ground temperature, since the
    author then chose the fixed-temperature boundary deliberately.

    Returns:
        Names of the floors that now use the foundation.
    """
    if idf.all_of_type(SiteGroundTemperatureBuildingSurface):
        return []
    floors = [
        surface
        for surface in idf.all_of_type(BuildingSurfaceDetailed).values()
        if surface.surface_type == "Floor"
        and surface.outside_boundary_condition == "Ground"
    ]
    if not floors:
        return []
    if not idf.has(FoundationKiva, FOUNDATION_NAME):
        # idfpy writes schema defaults explicitly; EnergyPlus then warns that
        # depths are set for insulation and footings that do not exist.
        idf.add(
            FoundationKiva(
                name=FOUNDATION_NAME,
                interior_horizontal_insulation_depth=None,
                exterior_horizontal_insulation_width=None,
                footing_depth=None,
            )
        )
    for floor in floors:
        perimeter = exposed_perimeter(idf, floor)
        floor.outside_boundary_condition = "Foundation"
        floor.outside_boundary_condition_object = FOUNDATION_NAME
        idf.add(
            SurfacePropertyExposedFoundationPerimeter(
                surface_name=floor.name,
                exposed_perimeter_calculation_method="TotalExposedPerimeter",
                total_exposed_perimeter=perimeter,
                exposed_perimeter_fraction=None,
            )
        )
    return [floor.name for floor in floors]
