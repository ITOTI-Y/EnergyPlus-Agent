"""Place windows and doors on their parent surfaces.

EnergyPlus stops on a window facing opposite to its wall and on an opening
in an interzone wall that lacks a matching opening in the adjacent zone, but
only warns when a window is off its wall's plane or outside its outline, and
then computes heat flows for that wrong geometry.
"""

from collections.abc import Sequence
from typing import Final, Literal

from idfpy import IDF
from idfpy.models.constructions import Construction
from idfpy.models.simulation import Building
from idfpy.models.thermal_zones import (
    BuildingSurfaceDetailed,
    FenestrationSurfaceDetailed,
)

from src.modeling import objects
from src.modeling.envelope import check_construction_fits, reversed_construction
from src.modeling.errors import (
    DuplicateNameError,
    ModelingError,
    ReferencedObjectError,
)

PLANE_TOLERANCE_M: Final = 0.01
PARTNER_SUFFIX: Final = "_Partner"

type Point = tuple[float, float, float]


def _dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _inside(point: Point, polygon: Sequence[Point], normal: Point) -> bool:
    """Point-in-polygon on the coordinate plane the polygon projects onto best."""
    drop = max(range(3), key=lambda i: abs(normal[i]))
    keep = [i for i in range(3) if i != drop]

    def flat(p: Point) -> tuple[float, float]:
        return p[keep[0]], p[keep[1]]

    px, py = flat(point)
    corners = [flat(p) for p in polygon]
    inside = False
    for (x1, y1), (x2, y2) in zip(corners, [*corners[1:], corners[0]], strict=True):
        # On an edge counts as inside: windows often share an edge with the wall.
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if (
            abs(cross) <= PLANE_TOLERANCE_M * ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            and min(x1, x2) - PLANE_TOLERANCE_M <= px <= max(x1, x2) + PLANE_TOLERANCE_M
            and min(y1, y2) - PLANE_TOLERANCE_M <= py <= max(y1, y2) + PLANE_TOLERANCE_M
        ):
            return True
        if (y1 > py) != (y2 > py) and px < x1 + (py - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def faces_away(
    fenestration: FenestrationSurfaceDetailed, parent: BuildingSurfaceDetailed
) -> bool:
    """Whether the vertex order makes the opening face opposite to its surface."""
    return _dot(fenestration.normal, parent.normal) < 0


def placement_problem(
    fenestration: FenestrationSurfaceDetailed, parent: BuildingSurfaceDetailed
) -> str | None:
    """Why the opening does not lie on its parent surface, if it does not."""
    wall = parent.vertices_as_tuples
    normal = parent.normal
    origin = wall[0]
    for point in fenestration.vertices_as_tuples:
        offset = (point[0] - origin[0], point[1] - origin[1], point[2] - origin[2])
        distance = abs(_dot(offset, normal))
        if distance > PLANE_TOLERANCE_M:
            return (
                f"vertex {point} is {distance:.3f} m off the plane of '{parent.name}'"
            )
        if not _inside(point, wall, normal):
            return f"vertex {point} lies outside surface '{parent.name}'"
    return None


def _reversed(fenestration: FenestrationSurfaceDetailed) -> dict[str, float | None]:
    points = fenestration.vertices_as_tuples[::-1]
    fields: dict[str, float | None] = {}
    for i in range(1, 5):
        point = points[i - 1] if i <= len(points) else None
        for axis, j in zip("xyz", range(3), strict=True):
            fields[f"vertex_{i}_{axis}_coordinate"] = point[j] if point else None
    return fields


def add_fenestration(
    idf: IDF, fenestration: FenestrationSurfaceDetailed
) -> tuple[list[FenestrationSurfaceDetailed], bool]:
    """Add an opening, fixing its orientation and pairing interzone openings.

    An opening in a wall whose outside boundary is another surface gets a
    mirrored partner in that surface, the two naming each other, as
    EnergyPlus requires. The partner takes the construction reversed, as
    its layers are seen from the other side (EnergyPlus warns otherwise).
    Both are added or neither.

    Returns:
        The created openings and whether the vertex order was reversed.

    Raises:
        ObjectNotFoundError: If the parent or its partner surface is missing.
        ValueError: If the construction does not suit the opening type, or the
            opening does not lie on its parent surface.
        ModelingError: On duplicate names or missing references.
    """
    parent = objects.get(
        idf, BuildingSurfaceDetailed, fenestration.building_surface_name
    )
    check_construction_fits(
        idf, fenestration.construction_name, fenestration.surface_type
    )
    flipped = faces_away(fenestration, parent)
    if flipped:
        fenestration = fenestration.model_copy(update=_reversed(fenestration))
    if problem := placement_problem(fenestration, parent):
        raise ValueError(f"Fenestration '{fenestration.name}': {problem}")
    if parent.outside_boundary_condition != "Surface":
        return [objects.create(idf, fenestration)], flipped

    partner_wall = objects.get(
        idf, BuildingSurfaceDetailed, parent.outside_boundary_condition_object or ""
    )
    partner = fenestration.model_copy(
        update={
            **_reversed(fenestration),
            "name": fenestration.name + PARTNER_SUFFIX,
            "building_surface_name": partner_wall.name,
            "outside_boundary_condition_object": fenestration.name,
        }
    )
    if problem := placement_problem(partner, partner_wall):
        raise ValueError(f"Fenestration '{partner.name}': {problem}")
    if idf.has(FenestrationSurfaceDetailed, partner.name):
        raise DuplicateNameError(partner.idf_object_type(), partner.name)
    constructions_before = set(idf.all_of_type(Construction))
    partner.construction_name = reversed_construction(
        idf, fenestration.construction_name
    )
    created = None
    try:
        created = objects.create(idf, fenestration)
        objects.create(idf, partner)
    except (ModelingError, ValueError):
        if created is not None:
            idf.remove(FenestrationSurfaceDetailed, created.name)
        if partner.construction_name not in constructions_before:
            idf.remove(Construction, partner.construction_name)
        raise
    created.outside_boundary_condition_object = partner.name
    return [created, partner], flipped


def remove_fenestration(idf: IDF, name: str) -> list[str]:
    """Delete an opening together with its interzone partner.

    Returns:
        Names of the deleted openings.

    Raises:
        ObjectNotFoundError: If no such opening exists.
    """
    opening = objects.get(idf, FenestrationSurfaceDetailed, name)
    partner = idf.get(
        FenestrationSurfaceDetailed, opening.outside_boundary_condition_object or ""
    )
    pair = [opening]
    if partner is not None and partner.outside_boundary_condition_object == name:
        pair.append(partner)
    for item in pair:
        # The two halves reference each other; only outside referrers block.
        others = [r for r in item.referencing() if all(r is not m for m in pair)]
        if others:
            raise ReferencedObjectError(
                item.idf_object_type(), item.name, [objects.label(r) for r in others]
            )
    for item in pair:
        idf.remove(FenestrationSurfaceDetailed, item.name)
    return [item.name for item in pair]


def update_fenestration(
    idf: IDF,
    name: str,
    *,
    new_name: str | None = None,
    construction_name: str | None = None,
    multiplier: float | None = None,
) -> list[FenestrationSurfaceDetailed]:
    """Rename an opening or change its construction or multiplier.

    The multiplier goes to the interzone partner as well, and the
    construction reversed, so both faces of the opening stay alike; a new
    name is also applied to the partner's reference. Geometry and parent changes go through
    ``remove_fenestration`` and ``add_fenestration``.

    Returns:
        The updated openings.

    Raises:
        ObjectNotFoundError: If no such opening exists.
        ValueError: If the construction does not suit the opening type.
        ModelingError: On a taken name or a missing construction.
    """
    opening = objects.get(idf, FenestrationSurfaceDetailed, name)
    if construction_name is not None:
        check_construction_fits(idf, construction_name, opening.surface_type)
    partner = idf.get(
        FenestrationSurfaceDetailed, opening.outside_boundary_condition_object or ""
    )
    shared = {
        k: v
        for k, v in {
            "construction_name": construction_name,
            "multiplier": multiplier,
        }.items()
        if v is not None
    }
    updated = [opening]
    if partner is not None and partner.outside_boundary_condition_object == name:
        partner_changes = dict(shared)
        if construction_name is not None:
            partner_changes["construction_name"] = reversed_construction(
                idf, construction_name
            )
        objects.update(idf, partner, partner_changes)
        updated.append(partner)
    rename = {} if new_name is None else {"name": new_name}
    objects.update(idf, opening, shared | rename)
    return updated


EDGE_OFFSET_M: Final = 0.05
"""Clearance between a ratio-sized window and the edges of its wall."""

type Facing = Literal["North", "East", "South", "West"]


def facing(surface: BuildingSurfaceDetailed, north_axis: float = 0.0) -> Facing:
    """Compass quadrant an outward normal points to.

    ``north_axis`` is the Building's rotation of the model's y axis from
    true north, in degrees clockwise.
    """
    azimuth = (surface.azimuth + north_axis) % 360.0
    return ("North", "East", "South", "West")[int(((azimuth + 45.0) % 360.0) // 90.0)]


def window_for_ratio(
    wall: BuildingSurfaceDetailed, ratio: float, sill_height: float
) -> list[Point]:
    """Corners of a strip window covering ``ratio`` of a vertical wall.

    The window spans the wall's width less EDGE_OFFSET_M at each side and is
    as high as the ratio needs, starting at ``sill_height`` above the wall's
    bottom; a sill too high for that height is lowered.

    Raises:
        ValueError: If the wall is not a vertical rectangle, or the ratio
            does not fit inside it.
    """
    points = wall.vertices_as_tuples
    if abs(wall.tilt - 90.0) > 1.0 or len(points) != 4:
        raise ValueError(f"'{wall.name}' is not a vertical rectangular wall")
    bottom = min(z for *_, z in points)
    top = max(z for *_, z in points)
    low = [p for p in points if abs(p[2] - bottom) <= PLANE_TOLERANCE_M]
    if len(low) != 2:
        raise ValueError(f"'{wall.name}' is not a vertical rectangular wall")
    (ax, ay, _), (bx, by, _) = low
    length = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
    width = length - 2 * EDGE_OFFSET_M
    height = ratio * wall.area / width if width > 0 else float("inf")
    room = top - bottom - 2 * EDGE_OFFSET_M
    if not 0 < ratio < 1 or height > room:
        raise ValueError(
            f"a window-to-wall ratio of {ratio} does not fit on '{wall.name}' "
            f"({length:.2f} m x {top - bottom:.2f} m)"
        )
    sill = min(max(sill_height, EDGE_OFFSET_M), top - bottom - EDGE_OFFSET_M - height)
    ux, uy = (bx - ax) / length, (by - ay) / length
    start = (ax + ux * EDGE_OFFSET_M, ay + uy * EDGE_OFFSET_M)
    end = (bx - ux * EDGE_OFFSET_M, by - uy * EDGE_OFFSET_M)
    z0, z1 = bottom + sill, bottom + sill + height
    # add_fenestration orients the corners to match the wall.
    return [
        (start[0], start[1], z1),
        (start[0], start[1], z0),
        (end[0], end[1], z0),
        (end[0], end[1], z1),
    ]


def add_windows_by_ratio(
    idf: IDF,
    zones: Sequence[str] | None,
    facings: Sequence[Facing] | None,
    ratio: float,
    construction_name: str,
    sill_height: float,
) -> tuple[list[str], list[str]]:
    """One strip window on every matching exterior wall.

    Walls are the outdoor walls of ``zones`` (all zones when None) facing
    one of ``facings`` (all when None). Each wall succeeds or fails on its
    own, e.g. a wall too narrow for the ratio.

    Returns:
        Names of the created windows, and why other walls got none.
    """
    buildings = list(idf.all_of_type(Building).values())
    north = float(buildings[0].north_axis or 0.0) if buildings else 0.0
    created: list[str] = []
    problems: list[str] = []
    for wall in idf.all_of_type(BuildingSurfaceDetailed).values():
        if (
            wall.surface_type != "Wall"
            or wall.outside_boundary_condition != "Outdoors"
            or (zones is not None and wall.zone_name not in zones)
            or (facings is not None and facing(wall, north) not in facings)
        ):
            continue
        name = f"{wall.name}_Window"
        try:
            corners = window_for_ratio(wall, ratio, sill_height)
            window = FenestrationSurfaceDetailed.model_validate(
                {
                    "name": name,
                    "surface_type": "Window",
                    "construction_name": construction_name,
                    "building_surface_name": wall.name,
                    "number_of_vertices": 4,
                    **{
                        f"vertex_{i}_{axis}_coordinate": point[j]
                        for i, point in enumerate(corners, start=1)
                        for j, axis in enumerate("xyz")
                    },
                }
            )
            add_fenestration(idf, window)
        except (ModelingError, ValueError) as e:
            problems.append(f"{wall.name}: {e}")
            continue
        created.append(name)
    return created, problems
