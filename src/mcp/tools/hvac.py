from typing import Any

from idfpy.models import (
    HVACTemplateThermostat,
    HVACTemplateZoneIdealLoadsAirSystem,
)

from src.mcp.state import ConfigState
from src.mcp.tools.base import BaseTool, normalize_payload


class ThermostatTool(BaseTool):
    def __init__(self, state: ConfigState):
        super().__init__(state, "HVACTemplate:Thermostat")

    @property
    def object_types(self) -> tuple[str, ...]:
        return ("HVACTemplate:Thermostat",)

    def _create_model(self, data: dict[str, Any]) -> HVACTemplateThermostat:
        return HVACTemplateThermostat(**normalize_payload(data))

    def _get_name(self, instance: HVACTemplateThermostat) -> str:
        return instance.name

    def _check_references(self, name: str) -> list[str]:
        refs = []
        for ils in self.state.idf.all_of_type(
            HVACTemplateZoneIdealLoadsAirSystem
        ).values():
            if ils.template_thermostat_name == name:
                refs.append(f"IdealLoadsSystem:{ils.zone_name}")
        return refs


class IdealLoadsSystemTool(BaseTool):
    def __init__(self, state: ConfigState):
        super().__init__(state, "HVACTemplate:Zone:IdealLoadsAirSystem")

    @property
    def object_types(self) -> tuple[str, ...]:
        return ("HVACTemplate:Zone:IdealLoadsAirSystem",)

    def _create_model(
        self, data: dict[str, Any]
    ) -> HVACTemplateZoneIdealLoadsAirSystem:
        return HVACTemplateZoneIdealLoadsAirSystem(**normalize_payload(data))

    def _get_name(self, instance: HVACTemplateZoneIdealLoadsAirSystem) -> str:
        return instance.zone_name

    def _check_references(self, name: str) -> list[str]:
        return []
