from typing import Any

from src.mcp.state import ConfigState
from src.mcp.tools.base import BaseTool
from src.validator.data_model import BuildingSchema

class BuildingTool(BaseTool):
    def __init__(self, state: ConfigState):
        super().__init__(state, "Building")

    @property
    def storage(self) -> dict[str, BuildingSchema]:
        if self.state.building:
            return {"Building": self.state.building}
        return {}

    def _validate_and_create(self, data: dict[str, Any]) -> BuildingSchema:
        return BuildingSchema.model_validate(data)

    def _get_name(self, instance: Any) -> str:
        return "Building"

    def _check_references(self, name: str) -> list[str]:
        return []

    def _add_to_storage(self, instance: Any) -> None:
        self.state.building = instance

    def _remove_from_storage(self, name: str) -> None:
        if name == "Building":
            self.state.building = None

    def _update_storage(self, name: str, instance: Any) -> None:
        if name == "Building":
            self.state.building = instance
