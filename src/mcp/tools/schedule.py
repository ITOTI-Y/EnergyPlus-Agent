from typing import Any

from src.mcp.state import ConfigState
from src.mcp.tools.base import BaseTool
from src.validator.data_model import (
    ScheduleCompactSchema,
    ScheduleTypeLimitsSchema,
)


class ScheduleTool(BaseTool):
    def __init__(self, state: ConfigState):
        super().__init__(state, "Schedule")

    @property
    def storage(
        self,
    ) -> dict[str, ScheduleTypeLimitsSchema | ScheduleCompactSchema]:
        if not self.state.schedules:
            return {}

        storage: dict[str, ScheduleTypeLimitsSchema |
                      ScheduleCompactSchema] = {}
        for limit in self.state.schedules.schedule_type_limits:
            storage[f"ScheduleTypeLimits:{limit.name}"] = limit
        for schedule in self.state.schedules.schedules:
            storage[f"Schedule:Compact:{schedule.name}"] = schedule
        return storage

    def _add_to_storage(
        self, instance: ScheduleTypeLimitsSchema | ScheduleCompactSchema
    ) -> None:
        if isinstance(instance, ScheduleTypeLimitsSchema):
            self.component_name = "ScheduleTypeLimits"
            self.state.schedules.schedule_type_limits.append(instance)
        elif isinstance(instance, ScheduleCompactSchema):
            self.component_name = "Schedule:Compact"
            self.state.schedules.schedules.append(instance)
        else:
            raise ValueError(f"Invalid schedule type: {type(instance)}")

    def _remove_from_storage(self, name: str) -> None:
        found_in_type_limits = any(
            limit.name == name for limit in self.state.schedules.schedule_type_limits
        )
        found_in_schedules = any(
            schedule.name == name for schedule in self.state.schedules.schedules
        )

        if not found_in_type_limits and not found_in_schedules:
            raise ValueError(f"Schedule not found: {name}")

        self.state.schedules.schedule_type_limits = [
            limit
            for limit in self.state.schedules.schedule_type_limits
            if limit.name != name
        ]
        self.state.schedules.schedules = [
            schedule
            for schedule in self.state.schedules.schedules
            if schedule.name != name
        ]

    def _update_storage(
        self, name: str, instance: ScheduleTypeLimitsSchema | ScheduleCompactSchema
    ) -> None:
        if isinstance(instance, ScheduleTypeLimitsSchema):
            self.component_name = "ScheduleTypeLimits"
            self.state.schedules.schedule_type_limits = [
                limit
                for limit in self.state.schedules.schedule_type_limits
                if limit.name != name
            ]
            self.state.schedules.schedule_type_limits.append(instance)
        elif isinstance(instance, ScheduleCompactSchema):
            self.component_name = "Schedule:Compact"
            self.state.schedules.schedules = [
                schedule
                for schedule in self.state.schedules.schedules
                if schedule.name != name
            ]
            self.state.schedules.schedules.append(instance)
        else:
            raise ValueError(f"Invalid schedule type: {type(instance)}")

    def _validate_and_create(
        self, data: dict[str, Any]
    ) -> ScheduleTypeLimitsSchema | ScheduleCompactSchema:
        if "ScheduleTypeLimits" in data:
            self.component_name = "ScheduleTypeLimits"
            return ScheduleTypeLimitsSchema.model_validate(data["ScheduleTypeLimits"])
        elif "Schedule:Compact" in data:
            self.component_name = "Schedule:Compact"
            return ScheduleCompactSchema.model_validate(data["Schedule:Compact"])
        else:
            raise ValueError(f"Invalid schedule type: {type(data)}")

    def _get_name(
        self, instance: ScheduleTypeLimitsSchema | ScheduleCompactSchema
    ) -> str:
        return instance.name

    def _check_references(self, name: str) -> list[str]:
        refs = []
        for schedule in self.state.schedules.schedules:
            if schedule.schedule_type_limits_name == name:
                refs.append(
                    f"ScheduleTypeLimits:{schedule.schedule_type_limits_name}")

        for thermostat in self.state.hvac.thermostats:
            if thermostat.heating_setpoint_schedule_name == name:
                refs.append(f"Thermostat:{thermostat.name}")
            if thermostat.cooling_setpoint_schedule_name == name:
                refs.append(f"Thermostat:{thermostat.name}")

        for ils in self.state.hvac.ideal_loads_systems:
            if ils.system_availability_schedule_name == name:
                refs.append(
                    f"IdealLoadsSystem:{ils.system_availability_schedule_name}")

        for people in self.state.people:
            if people.number_of_people_schedule_name == name:
                refs.append(f"People:{people.name}")
            if people.activity_level_schedule_name == name:
                refs.append(f"People:{people.name}")

        for light in self.state.lights:
            if light.schedule_name == name:
                refs.append(f"Light:{light.name}")
        return refs
