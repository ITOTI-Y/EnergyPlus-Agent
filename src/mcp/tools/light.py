from typing import Any

from idfpy.models import Lights

from src.mcp.state import ConfigState
from src.mcp.tools.base import BaseTool, normalize_payload


class LightTool(BaseTool):
    def __init__(self, state: ConfigState):
        super().__init__(state, "Light")

    @property
    def object_types(self) -> tuple[str, ...]:
        return ("Lights", "Light")

    def _create_model(self, data: dict[str, Any]) -> Lights:
        return Lights(**normalize_payload(data))

    def _get_name(self, instance: Lights) -> str:
        return instance.name

    def _check_references(self, name: str) -> list[str]:
        return []
