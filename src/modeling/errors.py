"""Failures of model operations that tool adapters report back to the caller."""

from typing import Any

from pydantic import ValidationError


class ModelingError(Exception):
    """An operation on the model was rejected; ``data`` carries the details."""

    def __init__(self, message: str, data: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.data = data


class ObjectNotFoundError(ModelingError, LookupError):
    def __init__(self, object_type: str, name: str) -> None:
        super().__init__(f"{object_type} '{name}' not found.")


class DuplicateNameError(ModelingError, ValueError):
    def __init__(self, object_type: str, name: str) -> None:
        super().__init__(f"{object_type} '{name}' already exists.")


class MissingReferenceError(ModelingError, ValueError):
    def __init__(self, object_type: str, name: str, missing: list[str]) -> None:
        super().__init__(
            f"{object_type} '{name}' references objects that do not exist.",
            {"missing_references": missing},
        )


class ReferencedObjectError(ModelingError, ValueError):
    def __init__(self, object_type: str, name: str, referrers: list[str]) -> None:
        super().__init__(
            f"{object_type} '{name}' is referenced by other objects.",
            {"references": referrers},
        )


def describe_error(
    error: ModelingError | ValueError,
) -> tuple[str, dict[str, Any] | None]:
    """Message and details for a rejected operation, field by field when known."""
    match error:
        case ValidationError():
            return f"Invalid {error.title} input.", {
                "errors": [
                    {"field": ".".join(map(str, e["loc"])), "message": e["msg"]}
                    for e in error.errors()
                ]
            }
        case ModelingError():
            return str(error), error.data
        case _:
            return str(error), None
