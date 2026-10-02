"""Problems in a model, each tied to the object that has to change.

Three sources feed the same record: idfpy reference checks, geometric checks
that EnergyPlus would only report as warnings or not at all, and the Severe
and Fatal messages of a simulation run. Tying every problem to an object type
lets callers hand it to whoever owns that type.
"""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

from idfpy import IDF, IDFBaseModel
from idfpy.models.thermal_zones import (
    BuildingSurfaceDetailed,
    FenestrationSurfaceDetailed,
    Zone,
)

from src.runner.runner import EnergyPlusMessage

PLANE_TOLERANCE_M: Final = 0.01
_QUOTED: Final = re.compile(r'"([^"]+)"')

type Point = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class ModelIssue:
    """One problem; ``object_type`` is None when no object can be blamed."""

    object_type: str | None
    object_name: str | None
    field: str | None
    message: str

    def __str__(self) -> str:
        if self.object_type is None:
            return self.message
        where = f"{self.object_type} '{self.object_name}'"
        if self.field:
            where += f" field {self.field}"
        return f"{where}: {self.message}"


def _display_name(obj: IDFBaseModel) -> str:
    # Nameless objects such as ideal loads systems are known by their zone.
    return getattr(obj, "name", None) or getattr(obj, "zone_name", None) or "?"


def reference_issues(idf: IDF) -> list[ModelIssue]:
    """References that name no existing object, from ``IDF.validate()``."""
    issues = []
    for error in idf.validate():
        obj = idf.all_of_type(error.object_type)[error.object_name]
        issues.append(
            ModelIssue(
                error.object_type,
                _display_name(obj),
                error.field_name,
                f"references '{error.referenced_name}', which does not exist",
            )
        )
    return issues


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


def fenestration_issues(idf: IDF) -> list[ModelIssue]:
    """Fenestration that faces away from or does not lie on its parent surface.

    A reversed vertex order is fatal in EnergyPlus; a window off its wall's
    plane or outside its outline only draws a warning, after which heat flows
    are computed for that wrong geometry.
    """
    issues = []
    for fenestration in idf.all_of_type(FenestrationSurfaceDetailed).values():
        parent = idf.get(BuildingSurfaceDetailed, fenestration.building_surface_name)
        if parent is None:
            continue  # reported by reference_issues
        wall = parent.vertices_as_tuples
        nx, ny, nz = parent.normal
        ox, oy, oz = wall[0]
        points = fenestration.vertices_as_tuples
        fx, fy, fz = fenestration.normal
        if fx * nx + fy * ny + fz * nz < 0:
            issues.append(
                ModelIssue(
                    fenestration.idf_object_type(),
                    fenestration.name,
                    None,
                    f"vertex order faces opposite to surface '{parent.name}'; "
                    "list the vertices counter-clockwise seen from outside",
                )
            )
            continue
        for point in points:
            x, y, z = point
            distance = abs((x - ox) * nx + (y - oy) * ny + (z - oz) * nz)
            if distance > PLANE_TOLERANCE_M:
                message = (
                    f"vertex {point} is {distance:.3f} m off the plane of "
                    f"'{parent.name}'"
                )
            elif not _inside(point, wall, (nx, ny, nz)):
                message = f"vertex {point} lies outside surface '{parent.name}'"
            else:
                continue
            issues.append(
                ModelIssue(
                    fenestration.idf_object_type(), fenestration.name, None, message
                )
            )
            break
    return issues


def model_issues(idf: IDF) -> list[ModelIssue]:
    """Problems in the objects present, detectable without running EnergyPlus."""
    return reference_issues(idf) + fenestration_issues(idf)


def completeness_issues(idf: IDF) -> list[ModelIssue]:
    """A model without zones or surfaces has nothing to simulate."""
    if not idf.all_of_type(Zone):
        return [ModelIssue(None, None, None, "The model has no zones.")]
    if not idf.all_of_type(BuildingSurfaceDetailed):
        return [ModelIssue(None, None, None, "The model has no surfaces.")]
    return []


def _name_index(idf: IDF) -> dict[str, tuple[str, str]]:
    index: dict[str, tuple[str, str]] = {}
    for obj in idf:
        name = getattr(obj, "name", None)
        if name:
            index.setdefault(name.upper(), (obj.idf_object_type(), name))
    return index


def simulation_issues(
    idf: IDF, messages: Iterable[EnergyPlusMessage]
) -> list[ModelIssue]:
    """Severe and Fatal messages, each tied to the first object they quote.

    EnergyPlus quotes object names in upper case, e.g.
    ``FenestrationSurface:Detailed="WIN_1" has an opaque surface construction``.
    """
    index = _name_index(idf)
    issues = []
    for message in messages:
        if message.severity == "Warning":
            continue
        quoted = (index.get(q.upper()) for q in _QUOTED.findall(message.text))
        found = next((hit for hit in quoted if hit is not None), None)
        object_type, name = found if found else (None, None)
        issues.append(ModelIssue(object_type, name, None, message.text))
    return issues
