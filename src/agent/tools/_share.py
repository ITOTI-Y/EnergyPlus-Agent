"""Glue between LangChain tools and ``src.modeling``."""

import functools
import json
from collections.abc import Callable
from typing import Any, Literal, cast

from idfpy import IDF, IDFBaseModel
from idfpy.models import Building, BuildingSurfaceDetailed, Construction, Zone
from langchain_core.tools import BaseTool, ToolException, tool
from pydantic import BaseModel, ConfigDict, ValidationError
from pydantic.v1 import ValidationError as ValidationErrorV1

from src.modeling import objects
from src.modeling.envelope import ConstructionKind, construction_kind
from src.modeling.errors import ModelingError, describe_error
from src.modeling.fenestration import facing
from src.modeling.objects import dumps


def ok(message: str, data: Any = None) -> str:
    return json.dumps({"success": True, "message": message, "data": data})


def model_tool(func: Callable[..., str]) -> BaseTool:
    """``@tool`` whose rejected operations come back as error tool messages.

    The error status lets the trace and the failure-loop guard tell failed
    calls apart; the content tells the model which field to fix.
    """

    @functools.wraps(func)
    def run(*args: Any, **kwargs: Any) -> str:
        try:
            return func(*args, **kwargs)
        except (ModelingError, ValueError) as e:
            message, data = describe_error(e)
            raise ToolException(
                json.dumps({"success": False, "message": message, "data": data})
            ) from e

    wrapped = tool(run)
    wrapped.handle_tool_error = True
    # A misspelt argument, e.g. 'plans-', would otherwise be dropped
    # silently and the call would run without it.
    schema = cast(type[BaseModel], wrapped.args_schema)
    wrapped.args_schema = type(
        schema.__name__,
        (schema,),
        {"model_config": ConfigDict(**schema.model_config, extra="forbid")},
    )
    wrapped.handle_validation_error = _invalid_arguments
    return wrapped


def _invalid_arguments(error: ValidationError | ValidationErrorV1) -> str:
    problems = [
        f"{'.'.join(map(str, e['loc'])) or 'arguments'}: {e['msg']}"
        for e in error.errors()
    ]
    return json.dumps(
        {"success": False, "message": "invalid arguments: " + "; ".join(problems)}
    )


def list_tool(
    idf: IDF,
    name: str,
    object_type: type[IDFBaseModel],
    description: str,
    fields: tuple[str, ...] | None = None,
) -> BaseTool:
    """Read-only tool listing every object of ``object_type``.

    ``fields`` limits each entry to those keys: every later call of the
    phase resends the result, so a large building's full records add up.
    """

    def list_objects() -> str:
        items = dumps(idf.all_of_type(object_type))
        if fields is not None:
            items = [{k: v for k, v in item.items() if k in fields} for item in items]
        return ok(f"Listed {len(items)} {object_type.idf_object_type()}.", items)

    list_objects.__name__ = name
    list_objects.__doc__ = description
    return tool(list_objects)


def list_zone_names_tool(idf: IDF, description: str) -> BaseTool:
    """Read-only tool listing zone names only."""

    def list_zones() -> str:
        names = list(idf.all_of_type(Zone))
        return ok(f"Listed {len(names)} zones.", names)

    list_zones.__doc__ = description
    return tool(list_zones)


def zones_of(
    plans: list[str] | None,
    zone_names: list[str] | None,
    plan_zones: dict[str, list[str]],
) -> list[str]:
    """The zones made from ``plans``, on every storey, then ``zone_names``.

    A model asked for a load per plan otherwise expands each plan into
    storey zones itself and names storeys a multiplied typical storey
    stands for, which have no zone.

    Raises:
        ValueError: If a plan is unknown, or neither argument names a zone.
    """
    if unknown := sorted(set(plans or []) - set(plan_zones)):
        raise ValueError(f"unknown plans {unknown}; the plans are {sorted(plan_zones)}")
    # A zone named both by plan and by name gets one object, not two.
    zones = list(
        dict.fromkeys(
            [z for key in plans or [] for z in plan_zones[key]] + list(zone_names or [])
        )
    )
    if not zones:
        raise ValueError("give the zones by `plans`, `zone_names` or both")
    return zones


def create_in_zones(
    idf: IDF, label: str, zone_names: list[str], build: Callable[[str], IDFBaseModel]
) -> str:
    """Create ``build(zone)`` for each zone; each zone succeeds or fails alone.

    The reply names only the failures: the created objects follow from the
    zones asked for, and every later call of the phase resends the reply.

    Raises:
        ModelingError: If no zone got its object, with each zone's reason.
    """
    failed = []
    for zone in zone_names:
        try:
            objects.create(idf, build(zone))
        except (ModelingError, ValueError) as e:
            message, data = describe_error(e)
            failed.append(
                f"{zone}: {message}" + (f" {json.dumps(data)}" if data else "")
            )
    created = len(zone_names) - len(failed)
    if created == 0:
        raise ModelingError(f"No {label} created.", {"failed": failed})
    message = f"Created {label} in {created} of {len(zone_names)} zones."
    return ok(message, {"failed": failed} if failed else None)


def list_constructions_tool(
    idf: IDF, description: str, kinds: tuple[ConstructionKind, ...]
) -> BaseTool:
    """Read-only tool listing constructions of the given kinds, kind included."""

    def list_constructions() -> str:
        items = [
            {"kind": kind, **construction.model_dump(exclude_none=True)}
            for construction in idf.all_of_type(Construction).values()
            if (kind := construction_kind(construction)) in kinds
        ]
        return ok(f"Listed {len(items)} constructions.", items)

    list_constructions.__doc__ = description
    return tool(list_constructions)


def list_surfaces_tool(idf: IDF, description: str) -> BaseTool:
    """Read-only, filtered surface list with compact entries.

    Full records of a large building's surfaces run to hundreds of thousands
    of characters, which every later call of the phase would resend.
    """

    def list_surfaces(
        zone_names: list[str] | None = None,
        surface_type: Literal["Wall", "Floor", "Ceiling", "Roof"] | None = None,
        outside_boundary_condition: Literal["Outdoors", "Surface", "Ground"]
        | None = None,
    ) -> str:
        """Entries give name, zone, type, boundary, construction, facing,
        area and corner coordinates.

        Args:
            zone_names: Only these zones.
            surface_type: Only this surface type.
            outside_boundary_condition: Only surfaces with this boundary.
        """
        buildings = list(idf.all_of_type(Building).values())
        north = float(buildings[0].north_axis or 0.0) if buildings else 0.0
        items = [
            {
                "name": s.name,
                "zone": s.zone_name,
                "type": s.surface_type,
                "boundary": s.outside_boundary_condition,
                "construction": s.construction_name,
                "facing": facing(s, north) if s.surface_type == "Wall" else None,
                "area": round(s.area, 2),
                "corners": [[round(c, 3) for c in p] for p in s.vertices_as_tuples],
            }
            for s in idf.all_of_type(BuildingSurfaceDetailed).values()
            if (zone_names is None or s.zone_name in zone_names)
            and (surface_type is None or s.surface_type == surface_type)
            and (
                outside_boundary_condition is None
                or s.outside_boundary_condition == outside_boundary_condition
            )
        ]
        return ok(f"Listed {len(items)} surfaces.", items)

    list_surfaces.__doc__ = f"{description}\n\n{list_surfaces.__doc__}"
    return tool(list_surfaces)
