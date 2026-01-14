from typing import Any

from src.mcp.state import ConfigState
from src.mcp.tools.base import BaseTool
from src.validator.data_model import SurfaceSchema


class SurfaceTool(BaseTool):
    def __init__(self, state: ConfigState):
        super().__init__(state, "Surface")

    @property
    def storage(self) -> dict[str, SurfaceSchema]:
        return {surface.name: surface for surface in self.state.surfaces}

    def _add_to_storage(self, instance: SurfaceSchema) -> None:
        self.state.surfaces.append(instance)

    def _remove_from_storage(self, name: str) -> None:
        self.state.surfaces = [s for s in self.state.surfaces if s.name != name]

    def _update_storage(self, name: str, instance: SurfaceSchema) -> None:
        self.state.surfaces = [s for s in self.state.surfaces if s.name != name]
        self.state.surfaces.append(instance)

    def _validate_and_create(self, data: dict[str, Any]) -> SurfaceSchema:
        return SurfaceSchema.model_validate(data)

    def _get_name(self, instance: SurfaceSchema) -> str:
        return instance.name

    def _check_references(self, name: str) -> list[str]:
        refs = []

        # Add logic to check for references to this surface in other components
        # For example:
        for fen in self.state.fenestrations:
            if fen.building_surface_name and name == fen.building_surface_name:
                refs.append(f"FenestrationSurface:{fen.name}")

        return refs
