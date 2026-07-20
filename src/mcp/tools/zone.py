from typing import Any

from idfpy.models import (
    BuildingSurfaceDetailed,
    HVACTemplateZoneIdealLoadsAirSystem,
    Zone,
)

from src.mcp.state import ConfigState
from src.mcp.tools.base import BaseTool, normalize_payload


class ZoneTool(BaseTool):
    def __init__(self, state: ConfigState):
        super().__init__(state, "Zone")

    @property
    def object_types(self) -> tuple[str, ...]:
        return ("Zone",)

    def _create_model(self, data: dict[str, Any]) -> Zone:
        payload = normalize_payload(data)
        payload.setdefault("direction_of_relative_north", 0.0)
        payload.setdefault("x_origin", 0.0)
        payload.setdefault("y_origin", 0.0)
        payload.setdefault("z_origin", 0.0)
        payload.setdefault("multiplier", 1)
        return Zone(**payload)

    def _get_name(self, instance: Zone) -> str:
        return instance.name

    def _check_references(self, name: str) -> list[str]:
        refs = []
        for surface in self.state.idf.all_of_type(BuildingSurfaceDetailed).values():
            if surface.zone_name == name:
                refs.append(f"Surface:{surface.name}")
        for ils in self.state.idf.all_of_type(
            HVACTemplateZoneIdealLoadsAirSystem
        ).values():
            if ils.zone_name == name:
                refs.append(f"IdealLoadsSystem:{ils.zone_name}")
        return refs
