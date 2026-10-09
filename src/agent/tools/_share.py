"""Glue between LangChain tools and ``src.modeling``."""

import functools
import json
from collections.abc import Callable
from typing import Any

from idfpy import IDF, IDFBaseModel
from idfpy.models.constructions import Construction
from idfpy.models.thermal_zones import Zone
from langchain_core.tools import BaseTool, ToolException, tool

from src.modeling import objects
from src.modeling.envelope import ConstructionKind, construction_kind
from src.modeling.errors import ModelingError, describe_error
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
    return wrapped


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
