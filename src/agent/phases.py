"""Which phase owns which object types, so problems reach the phase that can fix them."""

from typing import Final, Literal

from idfpy import IDF, IDFBaseModel
from idfpy.models.constructions import Construction
from idfpy.models.hvac_templates import (
    HVACTemplateThermostat,
    HVACTemplateZoneIdealLoadsAirSystem,
)
from idfpy.models.internal_gains import Lights, People
from idfpy.models.schedules import ScheduleCompact, ScheduleTypeLimits
from idfpy.models.thermal_zones import (
    BuildingSurfaceDetailed,
    FenestrationSurfaceDetailed,
    Zone,
)

from src.agent.state import IntakeOutput
from src.modeling.envelope import MATERIAL_TYPES
from src.modeling.validation import ModelIssue

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
}

FOUNDATION_PHASES: Final[tuple[Phase, ...]] = ("zone", "material", "schedule")

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
        specs: str = getattr(intake, f"{phase}_specs")
        if specs.strip() and not any(idf.all_of_type(t) for t in object_types):
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
