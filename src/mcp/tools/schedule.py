from typing import Any

from idfpy.models import (
    HVACTemplateThermostat,
    HVACTemplateZoneIdealLoadsAirSystem,
    Lights,
    People,
    ScheduleCompact,
    ScheduleCompactDataItem,
    ScheduleTypeLimits,
)

from src.mcp.tools.base import BaseTool, normalize_payload
from src.state.config_state import ConfigState
from src.state.schedules import flatten_schedule_data


def _schedule_data_items(data: list[dict[str, Any]]) -> list[ScheduleCompactDataItem]:
    if data and all("field" in item for item in data):
        return [ScheduleCompactDataItem(field=item["field"]) for item in data]
    return [ScheduleCompactDataItem(field=v) for v in flatten_schedule_data(data)]


class ScheduleTypeLimitsTool(BaseTool):
    def __init__(self, state: ConfigState):
        super().__init__(state, "ScheduleTypeLimits")

    @property
    def object_types(self) -> tuple[str, ...]:
        return ("ScheduleTypeLimits",)

    def _create_model(
        self,
        data: dict[str, Any],
        *,
        existing_object_type: str | None = None,
    ) -> ScheduleTypeLimits:
        payload = normalize_payload(data)
        for key in ("lower_limit_value", "upper_limit_value"):
            if payload.get(key) == "":
                payload[key] = None
        return ScheduleTypeLimits(**payload)

    def _get_name(self, instance: ScheduleTypeLimits) -> str:
        return instance.name

    def _check_references(self, name: str) -> list[str]:
        refs = []
        for schedule in self.state.idf.all_of_type(ScheduleCompact).values():
            if schedule.schedule_type_limits_name == name:
                refs.append(f"ScheduleCompact:{schedule.name}")
        return refs


class ScheduleCompactTool(BaseTool):
    def __init__(self, state: ConfigState):
        super().__init__(state, "ScheduleCompact")

    @property
    def object_types(self) -> tuple[str, ...]:
        return ("Schedule:Compact", "ScheduleCompact")

    def _create_model(
        self,
        data: dict[str, Any],
        *,
        existing_object_type: str | None = None,
    ) -> ScheduleCompact:
        payload = normalize_payload(data)
        raw_data = payload.pop("data", payload.pop("times", None))
        if raw_data is not None:
            payload["data"] = _schedule_data_items(raw_data)
        return ScheduleCompact(**payload)

    def _get_name(self, instance: ScheduleCompact) -> str:
        return instance.name

    def _check_references(self, name: str) -> list[str]:
        refs = []
        for thermostat in self.state.idf.all_of_type(HVACTemplateThermostat).values():
            if thermostat.heating_setpoint_schedule_name == name:
                refs.append(f"Thermostat:{thermostat.name}")
            if thermostat.cooling_setpoint_schedule_name == name:
                refs.append(f"Thermostat:{thermostat.name}")
        for ils in self.state.idf.all_of_type(
            HVACTemplateZoneIdealLoadsAirSystem
        ).values():
            if getattr(ils, "system_availability_schedule_name", None) == name:
                refs.append(f"IdealLoadsSystem:{ils.zone_name}")
        for people in self.state.idf.all_of_type(People).values():
            if people.number_of_people_schedule_name == name:
                refs.append(f"People:{people.name}")
            if people.activity_level_schedule_name == name:
                refs.append(f"People:{people.name}")
        for light in self.state.idf.all_of_type(Lights).values():
            if light.schedule_name == name:
                refs.append(f"Lights:{light.name}")
        return refs
