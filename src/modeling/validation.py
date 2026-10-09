"""Problems in a model, each tied to the object that has to change.

Three sources feed the same record: idfpy reference checks, geometric checks
that EnergyPlus would only report as warnings or not at all, and the Severe
and Fatal messages of a simulation run. Tying every problem to an object type
lets callers hand it to whoever owns that type.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

from idfpy import IDF, IDFBaseModel
from idfpy.models.thermal_zones import (
    BuildingSurfaceDetailed,
    FenestrationSurfaceDetailed,
    Zone,
)

from src.modeling.fenestration import faces_away, placement_problem
from src.modeling.surfaces import pair_problem
from src.runner.runner import EnergyPlusMessage

# EnergyPlus names objects either quoted, `Subsurface="WIN_1"`, or after a
# keyword, `convergence error ... for window WIN_1`.
_NAMED: Final = re.compile(
    r'"([^"]+)"|\b(?:window|surface|zone|schedule|construction)\s+([^\s,;"]+)',
    re.IGNORECASE,
)

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


def fenestration_issues(idf: IDF) -> list[ModelIssue]:
    """Fenestration that faces away from or does not lie on its parent surface."""
    issues = []
    for fenestration in idf.all_of_type(FenestrationSurfaceDetailed).values():
        parent = idf.get(BuildingSurfaceDetailed, fenestration.building_surface_name)
        if parent is None:
            continue  # reported by reference_issues
        if faces_away(fenestration, parent):
            problem = (
                f"vertex order faces opposite to surface '{parent.name}'; "
                "list the vertices counter-clockwise seen from outside"
            )
        else:
            problem = placement_problem(fenestration, parent)
        if problem:
            issues.append(
                ModelIssue(
                    fenestration.idf_object_type(), fenestration.name, None, problem
                )
            )
    return issues


def interzone_issues(idf: IDF) -> list[ModelIssue]:
    """Interzone surfaces whose partner does not name them back or does not match.

    EnergyPlus needs both faces of an interzone element; a one-sided pair
    leaves the other face adiabatic or outdoors and fails any opening in it.
    """
    issues = []
    for surface in idf.all_of_type(BuildingSurfaceDetailed).values():
        if surface.outside_boundary_condition != "Surface":
            continue
        partner = idf.get(
            BuildingSurfaceDetailed, surface.outside_boundary_condition_object or ""
        )
        if partner is None:
            continue  # reported by reference_issues
        if partner.outside_boundary_condition_object != surface.name:
            problem = (
                f"partner '{partner.name}' does not name it back "
                f"(its boundary is {partner.outside_boundary_condition})"
            )
        else:
            problem = pair_problem(surface, partner)
        if problem:
            issues.append(
                ModelIssue(surface.idf_object_type(), surface.name, None, problem)
            )
    return issues


def model_issues(idf: IDF) -> list[ModelIssue]:
    """Problems in the objects present, detectable without running EnergyPlus."""
    return reference_issues(idf) + fenestration_issues(idf) + interzone_issues(idf)


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
    """Severe and Fatal messages, each tied to the first object they name.

    EnergyPlus quotes object names in upper case, e.g.
    ``FenestrationSurface:Detailed="WIN_1" has an opaque surface construction``.
    """
    index = _name_index(idf)
    issues = []
    for message in messages:
        if message.severity == "Warning":
            continue
        named = (
            index.get((quoted or bare).upper())
            for quoted, bare in _NAMED.findall(message.text)
        )
        found = next((hit for hit in named if hit is not None), None)
        object_type, name = found if found else (None, None)
        issues.append(ModelIssue(object_type, name, None, message.text))
    return issues
