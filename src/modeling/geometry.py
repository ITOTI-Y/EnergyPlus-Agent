"""Zones extruded from a floor plan, with faces shared between zones paired.

Every adjacency, beside zones of equal or different height and storeys whose
plans do not line up, is one operation: two coplanar faces of different
zones that face each other overlap. The overlap becomes an interzone pair
(``Surface`` boundary, each naming the other) and the rest of each face
keeps its own boundary. Existing faces are split as needed, so zones can be
added in any order.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Final, Literal

from idfpy import IDF
from idfpy.ext.geometry.functions import polygon_normal
from idfpy.models.constructions import Construction
from idfpy.models.simulation import Building
from idfpy.models.thermal_zones import BuildingSurfaceDetailed, Zone
from pydantic import BaseModel, ConfigDict, Field
from shapely import GeometryCollection, MultiPolygon, Polygon, ops, union_all
from shapely.geometry import LineString
from shapely.geometry.polygon import orient

from src.modeling import objects
from src.modeling.envelope import check_construction_fits, reversed_construction
from src.modeling.errors import ModelingError, ReferencedObjectError
from src.modeling.surfaces import add_surface

GROUND_Z: Final = 0.0
TOLERANCE_M: Final = 1e-3
MIN_AREA_M2: Final = 1e-4
_DECIMALS: Final = 6

type Point = tuple[float, float, float]
type SurfaceType = Literal["Wall", "Floor", "Roof", "Ceiling"]
type Boundary = Literal["Outdoors", "Ground", "Surface"]


class PlanPointSchema(BaseModel):
    """One corner of a zone's floor plan, in metres."""

    model_config = ConfigDict(populate_by_name=True)

    x: float = Field(alias="X")
    y: float = Field(alias="Y")


@dataclass(frozen=True, slots=True)
class ZoneConstructions:
    """Constructions of the faces an extruded zone can have."""

    exterior_wall: str
    roof: str
    ground_floor: str
    interior_wall: str
    interior_floor: str


@dataclass(slots=True)
class ZoneGeometryResult:
    created: list[str] = field(default_factory=list)
    replaced: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Point, b: Point) -> Point:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


@dataclass(frozen=True, slots=True)
class _Plane:
    """Frame of a face; (u, v) runs counter-clockwise seen from outside."""

    origin: Point
    normal: Point
    u: Point
    v: Point

    @classmethod
    def of(cls, points: list[Point]) -> "_Plane":
        n = polygon_normal(points)
        u = _cross((0.0, 0.0, 1.0), n)
        length = _dot(u, u) ** 0.5
        u = (1.0, 0.0, 0.0) if length < 1e-6 else (u[0] / length, u[1] / length, 0.0)
        return cls(points[0], n, u, _cross(n, u))

    def flat(self, points: Iterable[Point]) -> Polygon:
        return Polygon(
            [
                (_dot(_sub(p, self.origin), self.u), _dot(_sub(p, self.origin), self.v))
                for p in points
            ]
        )

    def lift(self, polygon: Polygon, facing: Literal[1, -1]) -> list[Point]:
        """Vertices counter-clockwise seen from the side ``facing`` the normal.

        Starts at the upper-left corner seen from that side, as
        GlobalGeometryRules UpperLeftCorner asks.
        """
        ring = list(orient(polygon, sign=float(facing)).exterior.coords)[:-1]
        top = max(b for _, b in ring)
        start = min(
            range(len(ring)),
            key=lambda i: (top - ring[i][1] > TOLERANCE_M, facing * ring[i][0]),
        )
        ring = ring[start:] + ring[:start]
        o, u, v = self.origin, self.u, self.v
        return [
            (
                round(o[0] + a * u[0] + b * v[0], _DECIMALS),
                round(o[1] + a * u[1] + b * v[1], _DECIMALS),
                round(o[2] + a * u[2] + b * v[2], _DECIMALS),
            )
            for a, b in ring
        ]

    def faces(self, other: "_Plane") -> bool:
        """Whether ``other`` lies in this plane and faces the opposite way."""
        return (
            _dot(self.normal, other.normal) < -0.999
            and abs(_dot(_sub(other.origin, self.origin), self.normal)) <= TOLERANCE_M
        )


def _cut(polygon: Polygon, x: float | None, y: float | None) -> list[Polygon]:
    """Split by the vertical line at ``x`` or the horizontal line at ``y``."""
    minx, miny, maxx, maxy = polygon.bounds
    line = (
        LineString([(x, miny - 1.0), (x, maxy + 1.0)])
        if x is not None
        else LineString([(minx - 1.0, y), (maxx + 1.0, y)])
    )
    return [g for g in ops.split(polygon, line).geoms if isinstance(g, Polygon)]


def _reflex_vertices(polygon: Polygon) -> list[tuple[float, float]]:
    ring = list(orient(polygon, sign=1.0).exterior.coords)[:-1]
    reflex = []
    for prev, here, nxt in zip(
        ring[-1:] + ring[:-1], ring, ring[1:] + ring[:1], strict=True
    ):
        turn = (here[0] - prev[0]) * (nxt[1] - here[1]) - (here[1] - prev[1]) * (
            nxt[0] - here[0]
        )
        if turn < -1e-9:
            reflex.append(here)
    return reflex


def _convex_parts(polygon: Polygon, depth: int = 0) -> list[Polygon]:
    """Cut a polygon at its reflex corners until every part is convex.

    EnergyPlus warns that shadows on non-convex receiving surfaces may be
    inaccurate. Cuts run along the plane axes through a reflex corner, which
    splits the right-angled outlines of buildings into rectangles; parts
    that a cut cannot improve are kept as they are.
    """
    if depth > 16 or polygon.convex_hull.area - polygon.area <= MIN_AREA_M2:
        return [polygon]
    for x, y in _reflex_vertices(polygon):
        for parts in (_cut(polygon, x, None), _cut(polygon, None, y)):
            if len(parts) > 1:
                return [q for part in parts for q in _convex_parts(part, depth + 1)]
    return [polygon]


def _polygons(geometry: object) -> list[Polygon]:
    """Convex polygonal parts of a shapely result, without slivers.

    EnergyPlus surfaces cannot have holes; convex parts never do. A part with
    a hole is first cut through the hole.
    """
    match geometry:
        case Polygon() if geometry.area < MIN_AREA_M2:
            return []
        case Polygon() if geometry.interiors:
            return _polygons(_cut(geometry, geometry.interiors[0].centroid.x, None))
        case Polygon():
            return [
                p.simplify(1e-9)
                for p in _convex_parts(geometry)
                if p.area >= MIN_AREA_M2
            ]
        case MultiPolygon() | GeometryCollection():
            return [p for part in geometry.geoms for p in _polygons(part)]
        case list():
            return [p for part in geometry for p in _polygons(part)]
        case _:
            return []


@dataclass(slots=True)
class _Planned:
    name: str
    surface_type: SurfaceType
    zone: str
    construction: str
    boundary: str
    points: list[Point]
    partner: str | None = None

    def surface(self) -> BuildingSurfaceDetailed:
        outdoors = self.boundary == "Outdoors"
        return BuildingSurfaceDetailed.model_validate(
            {
                "name": self.name,
                "surface_type": self.surface_type,
                "construction_name": self.construction,
                "zone_name": self.zone,
                "outside_boundary_condition": self.boundary,
                "outside_boundary_condition_object": self.partner,
                "sun_exposure": "SunExposed" if outdoors else "NoSun",
                "wind_exposure": "WindExposed" if outdoors else "NoWind",
                "number_of_vertices": len(self.points),
                "vertices": [
                    {
                        "vertex_x_coordinate": x,
                        "vertex_y_coordinate": y,
                        "vertex_z_coordinate": z,
                    }
                    for x, y, z in self.points
                ],
            }
        )


@dataclass(slots=True)
class _Face:
    """A face of the new zone and the overlaps found on it."""

    role: str
    surface_type: SurfaceType
    plane: _Plane
    remainder: Polygon
    pairs: list[tuple[Polygon, BuildingSurfaceDetailed]] = field(default_factory=list)
    selves: list[Polygon] = field(default_factory=list)
    """Overlaps with zones of another multiplier: they face themselves."""


def _zone_faces(
    footprint: Polygon, z0: float, z1: float
) -> list[tuple[str, SurfaceType, list[Point]]]:
    corners = list(orient(footprint, sign=1.0).exterior.coords)[:-1]
    walls: list[tuple[str, SurfaceType, list[Point]]] = [
        (f"Wall_{i}", "Wall", [(x1, y1, z1), (x1, y1, z0), (x2, y2, z0), (x2, y2, z1)])
        for i, ((x1, y1), (x2, y2)) in enumerate(
            zip(corners, corners[1:] + corners[:1], strict=True), start=1
        )
    ]
    return [
        *walls,
        ("Floor", "Floor", [(x, y, z0) for x, y in reversed(corners)]),
        ("Roof", "Roof", [(x, y, z1) for x, y in corners]),
    ]


def multiplier(idf: IDF, zone_name: str) -> int:
    zone = idf.get(Zone, zone_name)
    return int(zone.multiplier or 1) if zone is not None else 1


def storeys_below_geometry(count: int, floor_z: float) -> int:
    """Storeys of a stack below the one its geometry is modelled at.

    EnergyPlus advises modelling multiplied floors about halfway up, since
    exterior convection depends on height; a stack on the ground stays
    there to keep its ground contact.
    """
    return 0 if floor_z <= GROUND_Z + TOLERANCE_M else (count - 1) // 2


def _zone_extents(idf: IDF) -> dict[str, tuple[Polygon, float, float]]:
    """Plan and the height range each zone stands for.

    A zone with multiplier N models one storey of N identical ones stacked
    above it, so it stands for N times its height.
    """
    floors: dict[str, list[Polygon]] = {}
    heights: dict[str, list[float]] = {}
    for surface in idf.all_of_type(BuildingSurfaceDetailed).values():
        points = surface.vertices_as_tuples
        heights.setdefault(surface.zone_name, []).extend(z for *_, z in points)
        if surface.surface_type == "Floor" and len(points) >= 3:
            floors.setdefault(surface.zone_name, []).append(
                Polygon([(x, y) for x, y, _ in points]).buffer(0)
            )
    extents = {}
    for zone, plans in floors.items():
        low, high = min(heights[zone]), max(heights[zone])
        count = multiplier(idf, zone)
        bottom = low - storeys_below_geometry(count, low) * (high - low)
        extents[zone] = (union_all(plans), bottom, bottom + count * (high - low))
    return extents


def _check_inputs(
    idf: IDF,
    zone_name: str,
    footprint: Polygon,
    z0: float,
    z1: float,
    constructions: ZoneConstructions,
) -> None:
    """Raises on invalid inputs; ``z1`` is the top the zone stands for."""
    objects.get(idf, Zone, zone_name)
    if not footprint.is_valid or footprint.area < MIN_AREA_M2:
        raise ValueError(
            f"Zone '{zone_name}': the plan must be a simple polygon of 3 or more "
            "corners enclosing an area."
        )
    if z1 - z0 <= TOLERANCE_M:
        raise ValueError(f"Zone '{zone_name}': height must be positive.")
    for surface_type, name in (
        ("Wall", constructions.exterior_wall),
        ("Roof", constructions.roof),
        ("Floor", constructions.ground_floor),
        ("Wall", constructions.interior_wall),
        ("Floor", constructions.interior_floor),
    ):
        check_construction_fits(idf, name, surface_type)
    for other, (plan, low, high) in _zone_extents(idf).items():
        if (
            other != zone_name
            and min(z1, high) - max(z0, low) > TOLERANCE_M
            and footprint.intersection(plan).area > MIN_AREA_M2
        ):
            raise ValueError(f"Zone '{zone_name}' would overlap zone '{other}'.")


class _Names:
    """Surface names unused in the model and in this plan."""

    def __init__(self, idf: IDF, freed: set[str]) -> None:
        self._taken = set(idf.all_of_type(BuildingSurfaceDetailed)) - freed

    def take(self, base: str, count: int) -> list[str]:
        names = []
        for k in range(1, count + 1):
            name = base if count == 1 else f"{base}_{k}"
            while name in self._taken:
                name += "_n"
            self._taken.add(name)
            names.append(name)
        return names


def _apply(
    idf: IDF, replaced: list[BuildingSurfaceDetailed], planned: list[_Planned]
) -> None:
    """Swap the replaced surfaces for the planned ones, all or nothing."""
    for surface in replaced:
        idf.remove(BuildingSurfaceDetailed, surface.name)
    added: list[str] = []
    try:
        for item in planned:
            add_surface(idf, item.surface())
            added.append(item.name)
    except (ModelingError, ValueError):
        for name in added:
            idf.remove(BuildingSurfaceDetailed, name)
        for surface in replaced:
            idf.add(surface)
        raise


def _solar_note(idf: IDF) -> str | None:
    """Switch interior solar distribution off, which needs convex geometry.

    EnergyPlus documents FullInteriorAndExterior as valid only for convex
    zones and surfaces, but does not reject other geometry.
    """
    for building in idf.all_of_type(Building).values():
        current = building.solar_distribution or ""
        if current.startswith("FullInteriorAndExterior"):
            exterior = current.replace("FullInteriorAndExterior", "FullExterior")
            objects.update(idf, building, {"solar_distribution": exterior})
            return (
                f"Building solar distribution set to {exterior}: "
                "interior solar distribution needs convex zones and surfaces."
            )
    return None


def _plan(
    idf: IDF,
    zone_name: str,
    z0: float,
    constructions: ZoneConstructions,
    faces: list[_Face],
    replaced: list[BuildingSurfaceDetailed],
    left: dict[str, tuple[_Plane, Polygon]],
    others_selves: dict[str, list[Polygon]],
    names: _Names,
    stacked: bool,
) -> list[_Planned]:
    """Surfaces replacing the split ones, and the faces of the new zone.

    Creates the reversed interior constructions the pairs need.
    """
    planned: list[_Planned] = []
    for surface in replaced:
        plane, shape = left[surface.name]
        for name, piece in zip(
            names.take(
                f"{surface.name}_Self", len(others_selves.get(surface.name, []))
            ),
            others_selves.get(surface.name, []),
            strict=True,
        ):
            kind = "Ceiling" if surface.surface_type == "Roof" else surface.surface_type
            planned.append(
                _self_paired(
                    idf,
                    name,
                    kind,
                    surface.zone_name,
                    constructions,
                    plane.lift(piece, facing=-1),
                )
            )
        pieces = _polygons(shape)
        for name, piece in zip(
            names.take(surface.name, len(pieces)), pieces, strict=True
        ):
            planned.append(
                _Planned(
                    name,
                    surface.surface_type,
                    surface.zone_name,
                    surface.construction_name,
                    surface.outside_boundary_condition,
                    plane.lift(piece, facing=-1),
                )
            )
    for face in faces:
        own_type, boundary, construction = _own_side(
            face.surface_type, z0, constructions
        )
        base = f"{zone_name}_{face.role}"
        # In a stack of identical storeys, the faces between storeys face
        # identical ones: they refer to themselves, as in the DOE prototypes.
        stack_face = stacked and (
            face.surface_type == "Roof"
            or (face.surface_type == "Floor" and boundary != "Ground")
        )
        selves = [*face.selves, *(_polygons(face.remainder) if stack_face else [])]
        kind = "Ceiling" if face.surface_type == "Roof" else face.surface_type
        for name, piece in zip(
            names.take(f"{base}_Self", len(selves)), selves, strict=True
        ):
            planned.append(
                _self_paired(
                    idf,
                    name,
                    kind,
                    zone_name,
                    constructions,
                    face.plane.lift(piece, facing=1),
                )
            )
        pieces = [] if stack_face else _polygons(face.remainder)
        for name, piece in zip(names.take(base, len(pieces)), pieces, strict=True):
            planned.append(
                _Planned(
                    name,
                    own_type,
                    zone_name,
                    construction,
                    boundary,
                    face.plane.lift(piece, facing=1),
                )
            )
        for overlap, other in face.pairs:
            ours = names.take(f"{base}_To_{other.zone_name}", 1)[0]
            theirs = names.take(f"{other.name}_To_{zone_name}", 1)[0]
            shared = (
                constructions.interior_wall
                if face.surface_type == "Wall"
                else constructions.interior_floor
            )
            # The floor, or the wall of the zone being extruded, takes the
            # construction as given; the ceiling or the other wall its reverse.
            reverse = reversed_construction(idf, shared)
            ours_type = "Ceiling" if face.surface_type == "Roof" else face.surface_type
            ours_construction, theirs_construction = (
                (reverse, shared) if ours_type == "Ceiling" else (shared, reverse)
            )
            planned.append(
                _Planned(
                    ours,
                    ours_type,
                    zone_name,
                    ours_construction,
                    "Surface",
                    face.plane.lift(overlap, facing=1),
                    theirs,
                )
            )
            planned.append(
                _Planned(
                    theirs,
                    "Ceiling" if other.surface_type == "Roof" else other.surface_type,
                    other.zone_name,
                    theirs_construction,
                    "Surface",
                    face.plane.lift(overlap, facing=-1),
                    ours,
                )
            )

    return planned


def _self_paired(
    idf: IDF,
    name: str,
    kind: SurfaceType,
    zone: str,
    constructions: ZoneConstructions,
    points: list[Point],
) -> _Planned:
    """A face whose outside sees conditions equal to its inside."""
    if kind == "Wall":
        construction = constructions.interior_wall
    elif kind == "Ceiling":
        construction = reversed_construction(idf, constructions.interior_floor)
    else:
        construction = constructions.interior_floor
    return _Planned(name, kind, zone, construction, "Surface", points, name)


def _stacks(idf: IDF, zone_name: str) -> list[tuple[Polygon, float, float]]:
    """Plan, floor level and represented top of the other multiplied zones."""
    return [
        (plan, low, top)
        for other, (plan, low, top) in _zone_extents(idf).items()
        if other != zone_name and multiplier(idf, other) > 1
    ]


def create_zone_geometry(
    idf: IDF,
    zone_name: str,
    plan: list[PlanPointSchema],
    floor_z: float,
    height: float,
    constructions: ZoneConstructions,
) -> ZoneGeometryResult:
    """Extrude a zone and pair its faces with those of neighbouring zones.

    Where a face lies on a face of another zone facing it, the overlap
    becomes an interzone pair: walls with the interior wall construction;
    floors and ceilings with the interior floor construction, a roof under
    the overlap turning into a ceiling. The ceiling, or the wall of the
    zone already there, gets the construction with its layers reversed.
    Elsewhere walls are outdoors, the floor is on the ground at z <= 0 and
    outdoors above it, and the top is a flat roof.

    A zone whose multiplier is N models one storey of N identical ones, as
    the DOE prototypes model typical floors; ``floor_z`` is the lowest of
    them, and the geometry is placed at the middle one (see
    ``storeys_below_geometry``). Its floor (above ground) and
    top refer to themselves, faces shared with zones of another multiplier
    refer to themselves on both sides, and a floor resting on the top such
    a stack stands for (N storeys up) refers to itself too. Zones with the
    same multiplier pair as usual.

    Raises:
        ObjectNotFoundError: If the zone does not exist.
        ValueError: On an invalid plan or height, constructions of the wrong
            kind, or a zone overlapping another zone.
        ReferencedObjectError: If a face to split carries openings.
    """
    footprint = Polygon([(p.x, p.y) for p in plan])
    count = multiplier(idf, zone_name)
    represented_top = floor_z + count * height
    _check_inputs(idf, zone_name, footprint, floor_z, represented_top, constructions)
    z0 = floor_z + storeys_below_geometry(count, floor_z) * height
    z1 = z0 + height

    others = [
        s
        for s in idf.all_of_type(BuildingSurfaceDetailed).values()
        if s.zone_name != zone_name
        and s.outside_boundary_condition != "Surface"
        and len(s.vertices_as_tuples) >= 3
    ]
    # Shape left of each touched surface, in the plane of the face touching it.
    left: dict[str, tuple[_Plane, Polygon]] = {}
    others_selves: dict[str, list[Polygon]] = {}

    def meet(face: _Face, other: BuildingSurfaceDetailed, emit: bool) -> None:
        """Split ``face`` and ``other`` where they overlap.

        Zones with the same multiplier pair; otherwise both overlaps face
        themselves. A virtual face (``emit`` False) only splits ``other``.
        """
        _, shape = left.get(
            other.name, (face.plane, face.plane.flat(other.vertices_as_tuples))
        )
        for overlap in _polygons(face.remainder.intersection(shape)):
            if multiplier(idf, other.zone_name) == count and emit:
                face.pairs.append((overlap, other))
            else:
                face.selves.append(overlap)
                others_selves.setdefault(other.name, []).append(overlap)
            face.remainder = face.remainder.difference(overlap)
            shape = shape.difference(overlap)
        left[other.name] = (face.plane, shape)

    faces: list[_Face] = []
    for role, surface_type, points in _zone_faces(footprint, z0, z1):
        face = _Face(role, surface_type, _Plane.of(points), Polygon())
        face.remainder = face.plane.flat(points)
        for other in others:
            if face.plane.faces(_Plane.of(other.vertices_as_tuples)):
                meet(face, other, emit=True)
        if surface_type in ("Floor", "Roof"):
            # A floor resting on the top a stack of storeys stands for, or a
            # top under a stack's floor; the stack's own faces already refer
            # to themselves, whichever zone came first.
            level = z0 if surface_type == "Floor" else z1
            for stack_plan, stack_low, stack_top in _stacks(idf, zone_name):
                touching = stack_top if surface_type == "Floor" else stack_low
                if abs(touching - level) <= TOLERANCE_M:
                    for overlap in _polygons(face.remainder.intersection(stack_plan)):
                        face.selves.append(overlap)
                        face.remainder = face.remainder.difference(overlap)
        faces.append(face)
    if count > 1:
        # Faces of other zones at the bottom and top this stack stands for:
        # tops of the storey below, floors of the storey above.
        for level, index in ((floor_z, -2), (represented_top, -1)):
            role, kind, points = _zone_faces(footprint, level, level)[index]
            bound = _Face(f"Stack{role}", kind, _Plane.of(points), Polygon())
            bound.remainder = bound.plane.flat(points)
            for other in others:
                if bound.plane.faces(_Plane.of(other.vertices_as_tuples)):
                    meet(bound, other, emit=False)

    replaced = [
        s
        for s in others
        if s.name in others_selves or any(o is s for f in faces for _, o in f.pairs)
    ]
    for surface in replaced:
        if referrers := surface.referencing():
            raise ReferencedObjectError(
                surface.idf_object_type(),
                surface.name,
                [objects.label(r) for r in referrers],
            )

    names = _Names(idf, {s.name for s in replaced})
    constructions_before = set(idf.all_of_type(Construction))
    try:
        planned = _plan(
            idf,
            zone_name,
            z0,
            constructions,
            faces,
            replaced,
            left,
            others_selves,
            names,
            stacked=count > 1,
        )
        _apply(idf, replaced, planned)
    except (ModelingError, ValueError):
        for name in set(idf.all_of_type(Construction)) - constructions_before:
            idf.remove(Construction, name)
        raise
    result = ZoneGeometryResult(
        created=[p.name for p in planned], replaced=[s.name for s in replaced]
    )
    # Faces are cut into convex parts, but an L-shaped zone stays non-convex.
    concave = footprint.convex_hull.area - footprint.area > MIN_AREA_M2
    if concave and (note := _solar_note(idf)):
        result.notes.append(note)
    return result


def _own_side(
    surface_type: SurfaceType, z0: float, constructions: ZoneConstructions
) -> tuple[SurfaceType, str, str]:
    """Type, boundary and construction of a face where no zone adjoins it."""
    match surface_type:
        case "Wall":
            return "Wall", "Outdoors", constructions.exterior_wall
        case "Floor":
            on_ground = z0 <= GROUND_Z + TOLERANCE_M
            return (
                "Floor",
                "Ground" if on_ground else "Outdoors",
                constructions.ground_floor,
            )
        case _:
            return "Roof", "Outdoors", constructions.roof
