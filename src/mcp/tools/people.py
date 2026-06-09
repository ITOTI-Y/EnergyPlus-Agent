from typing import Any

from idfpy.models.internal_gains import People

from src.mcp.state import ConfigState
from src.mcp.tools.base import BaseTool, normalize_payload


class PeopleTool(BaseTool):
    def __init__(self, state: ConfigState):
        super().__init__(state, "People")

    @property
    def object_types(self) -> tuple[str, ...]:
        return ("People",)

    def _create_model(self, data: dict[str, Any]) -> People:
        return People(**normalize_payload(data))

    def _get_name(self, instance: People) -> str:
        return instance.name

    def _check_references(self, name: str) -> list[str]:
        return []
