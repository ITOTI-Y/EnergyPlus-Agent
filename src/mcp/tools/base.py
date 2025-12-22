from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ValidationError

from src.mcp.interface import SchemaValidationError, ToolResponse
from src.mcp.state import ConfigState
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BaseTool(ABC):
    def __init__(self, state: ConfigState, component_name: str):
        self.state = state
        self.component_name = component_name

    @property
    @abstractmethod
    def storage(self) -> dict[str, BaseModel]: ...

    @abstractmethod
    def _validate_and_create(self, data: dict[str, Any]) -> BaseModel: ...

    @abstractmethod
    def _get_name(self, instance: Any) -> str: ...

    @abstractmethod
    def _check_references(self, name: str) -> list[str]: ...

    @abstractmethod
    def _add_to_storage(self, instance: Any) -> None: ...

    @abstractmethod
    def _remove_from_storage(self, name: str) -> None: ...

    @abstractmethod
    def _update_storage(self, name: str, instance: Any) -> None: ...

    def create(self, data: dict[str, Any]) -> ToolResponse:
        try:
            instance = self._validate_and_create(data)
            name = self._get_name(instance)

            if name in self.storage:
                return ToolResponse(
                    success=False,
                    message=f"Component '{self.component_name}':'{name}' already exists.",
                )

            self._add_to_storage(instance)
            logger.info(
                f"Component '{self.component_name}':'{name}' created successfully."
            )

            return ToolResponse(
                success=True,
                message=f"Component '{self.component_name}':'{name}' created successfully.",
                data=instance.model_dump(by_alias=True),
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
                f"Error creating component '{self.component_name}':'{name}'."
            )
            return ToolResponse(
                success=False,
                message=f"Error creating component '{self.component_name}':'{name}': {e!s}",
            )

    def read(self, name: str) -> ToolResponse:
        if name not in self.storage:
            return ToolResponse(
                success=False,
                message=f"Component '{self.component_name}':'{name}' not found.",
            )

        instance = self.storage[name]
        return ToolResponse(
            success=True,
            message=f"Component '{self.component_name}':'{name}' read successfully.",
            data=instance.model_dump(by_alias=True),
        )

    def update(self, name: str, data: dict[str, Any]) -> ToolResponse:
        if name not in self.storage:
            return ToolResponse(
                success=False,
                message=f"Component '{self.component_name}':'{name}' not found.",
            )

        try:
            existing = self.storage[name]
            updated = existing.model_validate({k: v for k, v in data.items() if v})

            new_name = self._get_name(updated)
            if new_name != name:
                self._remove_from_storage(name)
                self._add_to_storage(updated)
                logger.info(f"Updated {self.component_name}: {name} -> {new_name}")
            else:
                self._update_storage(name, updated)
                logger.info(f"Updated {self.component_name}: {name}")

            return ToolResponse(
                success=True,
                message=f"Component '{self.component_name}':'{name}' updated successfully.",
                data=updated.model_dump(by_alias=True),
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
                f"Error updating component '{self.component_name}':'{name}'."
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

        self._remove_from_storage(name)
        logger.info(f"Deleted {self.component_name}':'{name}'")
        return ToolResponse(
            success=True,
            message=f"Component '{self.component_name}':'{name}' deleted successfully.",
        )

    def list_all(self) -> ToolResponse:
        items = [item.model_dump(by_alias=True) for item in self.storage.values()]

        return ToolResponse(
            success=True,
            message=f"Listed {len(items)} {self.component_name}s.",
            data=items,
        )
