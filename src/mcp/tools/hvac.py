from typing import Any

from src.mcp.state import ConfigState
from src.mcp.tools.base import BaseTool
from src.validator.data_model import (
    HVACTemplateThermostatSchema,
    HVACTemplateZoneIdealLoadsAirSystemSchema
)

class ThermostatTool(BaseTool):
    def __init__(self, state: ConfigState):
        super().__init__(state, "HVACTemplate:Thermostat")

    @property
    def storage(self) -> dict[str, HVACTemplateThermostatSchema]:
        if not self.state.hvac or not self.state.hvac.thermostats:
            return {}
        return {t.name: t for t in self.state.hvac.thermostats}

    def _validate_and_create(self, data: dict[str, Any]) -> HVACTemplateThermostatSchema:
        return HVACTemplateThermostatSchema.model_validate(data)

    def _get_name(self, instance: Any) -> str:
        return instance.name

    def _check_references(self, name: str) -> list[str]:
        refs = []
        if self.state.hvac and self.state.hvac.ideal_loads_systems:
            for ils in self.state.hvac.ideal_loads_systems:
                if ils.template_thermostat_name == name:
                    refs.append(f"IdealLoadsSystem_Zone:{ils.zone_name}")
        return refs

    def _add_to_storage(self, instance: Any) -> None:
        self.state.hvac.thermostats.append(instance)

    def _remove_from_storage(self, name: str) -> None:
        self.state.hvac.thermostats = [
            t for t in self.state.hvac.thermostats if t.name != name
        ]

    def _update_storage(self, name: str, instance: Any) -> None:
        self._remove_from_storage(name)
        self._add_to_storage(instance)


class IdealLoadsSystemTool(BaseTool):
    def __init__(self, state: ConfigState):
        super().__init__(state, "HVACTemplate:Zone:IdealLoadsAirSystem")

    @property
    def storage(self) -> dict[str, HVACTemplateZoneIdealLoadsAirSystemSchema]:
        if not self.state.hvac or not self.state.hvac.ideal_loads_systems:
            return {}
        return {ils.zone_name: ils for ils in self.state.hvac.ideal_loads_systems}

    def _validate_and_create(self, data: dict[str, Any]) -> HVACTemplateZoneIdealLoadsAirSystemSchema:
        return HVACTemplateZoneIdealLoadsAirSystemSchema.model_validate(data)

    def _get_name(self, instance: Any) -> str:
        return instance.zone_name

    def _check_references(self, name: str) -> list[str]:
        return []

    def _add_to_storage(self, instance: Any) -> None:
        self.state.hvac.ideal_loads_systems.append(instance)

    def _remove_from_storage(self, name: str) -> None:
        self.state.hvac.ideal_loads_systems = [
            ils for ils in self.state.hvac.ideal_loads_systems if ils.zone_name != name
        ]

    def _update_storage(self, name: str, instance: Any) -> None:
        self._remove_from_storage(name)
        self._add_to_storage(instance)
