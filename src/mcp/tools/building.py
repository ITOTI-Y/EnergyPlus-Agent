from typing import Any

from src.mcp.state import ConfigState
from src.mcp.tools.base import BaseTool
from src.validator.data_model import BuildingSchema


class BuildingTool(BaseTool):
    """Tool for managing EnergyPlus Building objects.

    Handles CRUD operations for the building configuration, which is a
    singleton object (only one building per configuration).
    """

    def __init__(self, state: ConfigState):
        super().__init__(state, "Building")

    @property
    def storage(self) -> dict[str, BuildingSchema]:
        if self.state.building:
            return {self.state.building.name: self.state.building}
        return {}

    def _add_to_storage(self, instance: BuildingSchema) -> None:
        self.state.building = instance

    def _remove_from_storage(self, name: str) -> None:
        if self.state.building and self.state.building.name == name:
            self.state.building = None
        else:
            raise ValueError(f"Building with name {name} not found.")

    def _update_storage(self, name: str, instance: BuildingSchema) -> None:
        if self.state.building and self.state.building.name == name:
            self.state.building = instance
        else:
            raise ValueError(f"Building with name {name} not found.")

    def _validate_and_create(self, data: dict[str, Any]) -> BuildingSchema:
        return BuildingSchema.model_validate(data)

    def _get_name(self, instance: BuildingSchema) -> str:
        return instance.name

    def _check_references(self, name: str) -> list[str]:
        return []
