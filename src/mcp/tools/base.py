from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

from pydantic import ValidationError

from src.mcp.interface import SchemaValidationError, ToolResponse
from src.mcp.state import ConfigState
from src.utils.logging import get_logger

logger = get_logger(__name__)


def payload_key_to_field(key: str) -> str:
    """Convert EnergyPlus/MCP alias keys to idfpy constructor field names."""
    return re.sub(r"_+", "_", re.sub(r"[^0-9A-Za-z]+", "_", key)).strip("_").lower()


def normalize_payload(data: dict[str, Any]) -> dict[str, Any]:
    return {payload_key_to_field(k): v for k, v in data.items() if v is not None}


def dump_obj(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_none=True)
    return dict(getattr(obj, "__dict__", {}))


class BaseTool(ABC):
    """Base class for MCP tools backed directly by an ``idfpy.IDF`` instance."""

    def __init__(self, state: ConfigState, component_name: str):
        self.state = state
        self.component_name = component_name

    @property
    @abstractmethod
    def object_types(self) -> tuple[str, ...]: ...

    @abstractmethod
    def _create_model(self, data: dict[str, Any]) -> Any: ...

    @abstractmethod
    def _get_name(self, instance: Any) -> str: ...

    @abstractmethod
    def _check_references(self, name: str) -> list[str]: ...

    def _iter_entries(self):
        for object_type in self.object_types:
            try:
                items = self.state.idf.all_of_type(object_type)
            except Exception:
                continue
            for key, obj in items.items():
                yield object_type, key, obj

    @property
    def storage(self) -> dict[str, Any]:
        return {self._get_name(obj): obj for _, _, obj in self._iter_entries()}

    def _find_entry(self, name: str) -> tuple[str, Any, Any] | None:
        for object_type, key, obj in self._iter_entries():
            if self._get_name(obj) == name:
                return object_type, key, obj
        return None

    def _add_to_idf(self, instance: Any) -> None:
        self.state.idf.add(instance)

    def _remove_from_idf(self, name: str) -> bool:
        entry = self._find_entry(name)
        if entry is None:
            return False
        object_type, key, _obj = entry
        self.state.idf.remove(object_type, key)
        return True

    def create(self, data: dict[str, Any]) -> ToolResponse:
        name = data.get("Name", data.get("name", "<unknown>"))
        try:
            instance = self._create_model(data)
            name = self._get_name(instance)
            if name in self.storage:
                return ToolResponse(
                    success=False,
                    message=f"Component '{self.component_name}':'{name}' already exists.",
                )

            self._add_to_idf(instance)
            logger.info(
                "Component '{}':'{}' created with idfpy.",
                self.component_name,
                name,
            )
            return ToolResponse(
                success=True,
                message=f"Component '{self.component_name}':'{name}' created successfully.",
                data=dump_obj(instance),
            )
        except ValidationError as e:
            errors = [
                SchemaValidationError(
                    field=".".join(str(loc) for loc in err["loc"]),
                    message=err["msg"],
                )
                for err in e.errors()
            ]
            return ToolResponse(
                success=False,
                message=f"Validation error for component '{self.component_name}':'{name}'.",
                data={"errors": [err.model_dump() for err in errors]},
            )
        except Exception as e:
            logger.exception(
                "Error creating component '{}':'{}'.", self.component_name, name
            )
            return ToolResponse(
                success=False,
                message=f"Error creating component '{self.component_name}':'{name}': {e!s}",
            )

    def read(self, name: str) -> ToolResponse:
        obj = self.storage.get(name)
        if obj is None:
            return ToolResponse(
                success=False,
                message=f"Component '{self.component_name}':'{name}' not found.",
            )
        return ToolResponse(
            success=True,
            message=f"Component '{self.component_name}':'{name}' read successfully.",
            data=dump_obj(obj),
        )

    def update(self, name: str, data: dict[str, Any]) -> ToolResponse:
        obj = self.storage.get(name)
        if obj is None:
            return ToolResponse(
                success=False,
                message=f"Component '{self.component_name}':'{name}' not found.",
            )

        try:
            existing_data = dump_obj(obj)
            existing_data.update(normalize_payload(data))
            updated = self._create_model(existing_data)
            new_name = self._get_name(updated)
            if new_name != name and new_name in self.storage:
                return ToolResponse(
                    success=False,
                    message=f"Component '{self.component_name}':'{new_name}' already exists.",
                )

            self._remove_from_idf(name)
            self._add_to_idf(updated)
            return ToolResponse(
                success=True,
                message=f"Component '{self.component_name}':'{name}' updated successfully.",
                data=dump_obj(updated),
            )
        except ValidationError as e:
            errors = [
                SchemaValidationError(
                    field=".".join(str(loc) for loc in err["loc"]),
                    message=err["msg"],
                )
                for err in e.errors()
            ]
            return ToolResponse(
                success=False,
                message=f"Validation error for component '{self.component_name}':'{name}'.",
                data={"errors": [err.model_dump() for err in errors]},
            )
        except Exception as e:
            logger.exception(
                "Error updating component '{}':'{}'.", self.component_name, name
            )
            return ToolResponse(
                success=False,
                message=f"Error updating component '{self.component_name}':'{name}': {e!s}",
            )

    def delete(self, name: str) -> ToolResponse:
        if name not in self.storage:
            return ToolResponse(
                success=False,
                message=f"Component '{self.component_name}':'{name}' not found.",
            )

        refs = self._check_references(name)
        if refs:
            return ToolResponse(
                success=False,
                message=f"Component '{self.component_name}':'{name}' is referenced by other components.",
                data={"references": refs},
            )

        self._remove_from_idf(name)
        return ToolResponse(
            success=True,
            message=f"Component '{self.component_name}':'{name}' deleted successfully.",
        )

    def list_all(self) -> ToolResponse:
        items = [dump_obj(item) for item in self.storage.values()]
        return ToolResponse(
            success=True,
            message=f"Listed {len(items)} {self.component_name}s.",
            data=items,
        )
