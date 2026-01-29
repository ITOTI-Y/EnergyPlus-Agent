from typing import Any

from src.mcp.state import ConfigState
from src.mcp.tools.base import BaseTool
from src.validator.data_model import (
    ScheduleCompactSchema,
    ScheduleTypeLimitsSchema
)

class ScheduleTool(BaseTool):
    def __init__(self, state: ConfigState):
        super().__init__(state, "Schedule:Compact")

    @property
    def storage(self) -> dict[str, ScheduleCompactSchema]:
        if not self.state.schedules or not self.state.schedules.schedules:
            return {}
        return {s.name: s for s in self.state.schedules.schedules}

    def _validate_and_create(self, data: dict[str, Any]) -> ScheduleCompactSchema:
        return ScheduleCompactSchema.model_validate(data)

    def _get_name(self, instance: Any) -> str:
        return instance.name

    def _check_references(self, name: str) -> list[str]:
        refs = []
        if self.state.hvac and self.state.hvac.thermostats:
            for t in self.state.hvac.thermostats:
                if t.heating_setpoint_schedule_name == name:
                    refs.append(f"Thermostat:{t.name}")
                if t.cooling_setpoint_schedule_name == name:
                    refs.append(f"Thermostat:{t.name}")
        if self.state.hvac and self.state.hvac.ideal_loads_systems:
            for ils in self.state.hvac.ideal_loads_systems:
                if ils.system_availability_schedule_name == name:
                    refs.append(f"IdealLoadsSystem_Zone:{ils.zone_name}")
        if hasattr(self.state, "people") and self.state.people:
            for p in self.state.people:
                if p.number_of_people_schedule_name == name:
                    refs.append(f"People:{p.name}")
                if p.activity_level_schedule_name == name:
                    refs.append(f"People:{p.name}")

        if hasattr(self.state, "lights") and self.state.lights:
            for l in self.state.lights:
                if l.schedule_name == name:
                    refs.append(f"Lights:{l.name}")

        return refs

    def _add_to_storage(self, instance: Any) -> None:
        self.state.schedules.schedules.append(instance)

    def _remove_from_storage(self, name: str) -> None:
        self.state.schedules.schedules = [
            s for s in self.state.schedules.schedules if s.name != name
        ]

    def _update_storage(self, name: str, instance: Any) -> None:
        self._remove_from_storage(name)
        self._add_to_storage(instance)


class ScheduleTypeLimitsTool(BaseTool):
    def __init__(self, state: ConfigState):
        super().__init__(state, "ScheduleTypeLimits")

    @property
    def storage(self) -> dict[str, ScheduleTypeLimitsSchema]:
        if not self.state.schedules or not self.state.schedules.schedule_type_limits:
            return {}
        return {s.name: s for s in self.state.schedules.schedule_type_limits}

    def _validate_and_create(self, data: dict[str, Any]) -> ScheduleTypeLimitsSchema:
        return ScheduleTypeLimitsSchema.model_validate(data)

    def _get_name(self, instance: Any) -> str:
        return instance.name

    def _check_references(self, name: str) -> list[str]:
        refs = []
        if self.state.schedules and self.state.schedules.schedules:
            for s in self.state.schedules.schedules:
                if s.schedule_type_limits_name == name:
                    refs.append(f"Schedule:Compact:{s.name}")
        return refs

    def _add_to_storage(self, instance: Any) -> None:
        self.state.schedules.schedule_type_limits.append(instance)

    def _remove_from_storage(self, name: str) -> None:
        self.state.schedules.schedule_type_limits = [
            s for s in self.state.schedules.schedule_type_limits if s.name != name
        ]

    def _update_storage(self, name: str, instance: Any) -> None:
        self._remove_from_storage(name)
        self._add_to_storage(instance)
