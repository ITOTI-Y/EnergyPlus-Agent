import functools
import inspect
from collections.abc import Callable
from typing import Any

from fastmcp import FastMCP
from idfpy import IDFBaseModel

from src.mcp.interface import ToolResponse
from src.modeling.errors import ModelingError, describe_error

type Outcome = tuple[str, Any]


def model_tool(mcp: FastMCP) -> Callable[[Callable[..., Outcome]], None]:
    """Register a function returning ``(message, data)`` as an MCP tool.

    Rejected model operations become failure responses with field-level
    details instead of exceptions.
    """

    def register(func: Callable[..., Outcome]) -> None:
        @functools.wraps(func)
        def run(*args: Any, **kwargs: Any) -> dict[str, Any]:
            try:
                message, data = func(*args, **kwargs)
            except (ModelingError, ValueError) as e:
                message, data = describe_error(e)
                response = ToolResponse(success=False, message=message, data=data)
            else:
                response = ToolResponse(success=True, message=message, data=data)
            return response.to_mcp_response()

        run.__signature__ = inspect.signature(func).replace(  # ty: ignore[unresolved-attribute]
            return_annotation=dict[str, Any]
        )
        mcp.tool(run)

    return register


def dump(obj: IDFBaseModel) -> dict[str, Any]:
    return obj.model_dump(exclude_none=True)


def given(**fields: Any) -> dict[str, Any]:
    """Fields the caller actually passed; None means "leave unchanged"."""
    return {k: v for k, v in fields.items() if v is not None}
