"""Glue between LangChain tools and ``src.modeling``."""

import functools
import json
from collections.abc import Callable
from typing import Any

from idfpy import IDF, IDFBaseModel
from langchain_core.tools import BaseTool, ToolException, tool

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
    idf: IDF, name: str, object_type: type[IDFBaseModel], description: str
) -> BaseTool:
    """Read-only tool listing every object of ``object_type``."""

    def list_objects() -> str:
        items = dumps(idf.all_of_type(object_type))
        return ok(f"Listed {len(items)} {object_type.idf_object_type()}.", items)

    list_objects.__name__ = name
    list_objects.__doc__ = description
    return tool(list_objects)
