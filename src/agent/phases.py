"""Which phase owns which object types, so problems reach the phase that can fix them."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal

from idfpy import IDF, IDFBaseModel
from idfpy.models.constructions import Construction
from idfpy.models.hvac_templates import (
    HVACTemplateThermostat,
    HVACTemplateZoneIdealLoadsAirSystem,
)
from idfpy.models.internal_gains import ElectricEquipment, Lights, People
from idfpy.models.schedules import ScheduleCompact, ScheduleTypeLimits
from idfpy.models.thermal_zones import (
    BuildingSurfaceDetailed,
    FenestrationSurfaceDetailed,
    Zone,
)

from src.modeling.envelope import MATERIAL_TYPES, construction_kind
from src.modeling.validation import ModelIssue

if TYPE_CHECKING:
    from src.agent.state import IntakeOutput

type Phase = Literal[
    "zone",
    "material",
    "schedule",
    "construction",
    "surface",
    "fenestration",
    "hvac",
    "people",
    "lights",
    "equipment",
]

PHASE_TYPES: Final[dict[Phase, tuple[type[IDFBaseModel], ...]]] = {
    "zone": (Zone,),
    "material": MATERIAL_TYPES,
    "schedule": (ScheduleTypeLimits, ScheduleCompact),
    "construction": (Construction,),
    "surface": (BuildingSurfaceDetailed,),
    "fenestration": (FenestrationSurfaceDetailed,),
    "hvac": (HVACTemplateThermostat, HVACTemplateZoneIdealLoadsAirSystem),
    "people": (People,),
    "lights": (Lights,),
    "equipment": (ElectricEquipment,),
}

FOUNDATION_PHASES: Final[tuple[Phase, ...]] = ("zone", "material", "schedule")

DEPENDS_ON: Final[dict[Phase, tuple[Phase, ...]]] = {
    "zone": (),
    "material": (),
    "schedule": (),
    "construction": ("material",),
    "surface": ("zone", "construction"),
    "fenestration": ("surface", "construction"),
    "hvac": ("zone", "schedule"),
    "people": ("zone", "schedule"),
    "lights": ("zone", "schedule"),
    "equipment": ("zone", "schedule"),
}
"""Phases whose objects a phase references; keys are in dependency order."""

_OWNERS: Final[dict[str, Phase]] = {
    object_type.idf_object_type(): phase
    for phase, object_types in PHASE_TYPES.items()
    for object_type in object_types
}


def owner(issue: ModelIssue) -> Phase | None:
    """Phase that creates the blamed object; None for model-wide problems."""
    return None if issue.object_type is None else _OWNERS.get(issue.object_type)


def missing_output_issues(
    idf: IDF, intake: IntakeOutput, phases: tuple[Phase, ...]
) -> list[ModelIssue]:
    """Phases given a task that ended without creating any object.

    A phase stopped by the failure-loop guard, or one that only reported a
    problem, leaves nothing behind and would otherwise go unnoticed until
    EnergyPlus runs.
    """
    issues = []
    for phase in phases:
        object_types = PHASE_TYPES[phase]
        if has_task(intake, phase) and not any(
            idf.all_of_type(t) for t in object_types
        ):
            issues.append(
                ModelIssue(
                    object_types[0].idf_object_type(),
                    None,
                    None,
                    f"The {phase} phase created no objects although its "
                    "specification asks for some.",
                )
            )
    return issues


def window_construction_issues(idf: IDF, intake: IntakeOutput) -> list[ModelIssue]:
    """Windows asked for, none built, and no window construction to build them.

    The fenestration phase cannot make constructions; blaming the
    construction phase reruns it, then fenestration after it.
    """
    glazed = any(
        f.surface_type in ("Window", "GlassDoor")
        for f in idf.all_of_type(FenestrationSurfaceDetailed).values()
    )
    windows = any(
        construction_kind(c) == "window" for c in idf.all_of_type(Construction).values()
    )
    if not has_task(intake, "fenestration") or glazed or windows:
        return []
    return [
        ModelIssue(
            Construction.idf_object_type(),
            None,
            None,
            "The fenestration specification asks for windows, but no "
            "construction is a window construction (glazing layers only); "
            "create the window constructions it needs.",
        )
    ]


def rerun_closure(phases: set[Phase]) -> set[Phase]:
    """The given phases and every phase that references their objects."""
    closure = set(phases)
    for phase, upstream in DEPENDS_ON.items():  # dependency order
        if closure.intersection(upstream):
            closure.add(phase)
    return closure


def remove_phase_objects(idf: IDF, phases: set[Phase]) -> list[str]:
    """Delete every object the given phases own, dependants first.

    ``phases`` must be closed under ``rerun_closure``, so no object left in
    the model references a removed one.

    Returns:
        Labels of the removed objects.
    """
    removed = []
    for phase in reversed(DEPENDS_ON):
        if phase not in phases:
            continue
        for object_type in PHASE_TYPES[phase]:
            for name in list(idf.all_of_type(object_type)):
                idf.remove(object_type, name)
                removed.append(f"{object_type.idf_object_type()} '{name}'")
    return removed


def has_task(intake: IntakeOutput, phase: Phase) -> bool:
    """Whether the intake output asks the phase to build anything."""
    if phase in ("zone", "surface"):
        # Surfaces come from the zone prisms; surface_specs only adds to them.
        return bool(intake.zones)
    specs: str = getattr(intake, f"{phase}_specs")
    return bool(specs.strip())
