from typing import Any

from idfpy.models.location import SiteLocation

from src.mcp.state import ConfigState
from src.mcp.tools.base import BaseTool, normalize_payload


class LocationTool(BaseTool):
    def __init__(self, state: ConfigState):
        super().__init__(state, "Location")

    @property
    def object_types(self) -> tuple[str, ...]:
        return ("Site:Location",)

    def _create_model(self, data: dict[str, Any]) -> SiteLocation:
        return SiteLocation(**normalize_payload(data))

    def _get_name(self, instance: SiteLocation) -> str:
        return instance.name

    def _check_references(self, name: str) -> list[str]:
        return []
