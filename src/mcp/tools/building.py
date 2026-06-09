from typing import Any

from idfpy.models.simulation import Building

from src.mcp.state import ConfigState
from src.mcp.tools.base import BaseTool, normalize_payload


class BuildingTool(BaseTool):
    def __init__(self, state: ConfigState):
        super().__init__(state, "Building")

    @property
    def object_types(self) -> tuple[str, ...]:
        return ("Building",)

    def _create_model(self, data: dict[str, Any]) -> Building:
        payload = normalize_payload(data)
        payload.setdefault("loads_convergence_tolerance_value", 0.04)
        payload.setdefault("temperature_convergence_tolerance_value", 0.4)
        payload.setdefault("solar_distribution", "FullExterior")
        payload.setdefault("maximum_number_of_warmup_days", 25)
        payload.setdefault("minimum_number_of_warmup_days", 1)
        return Building(**payload)

    def _get_name(self, instance: Building) -> str:
        return instance.name

    def _check_references(self, name: str) -> list[str]:
        return []
