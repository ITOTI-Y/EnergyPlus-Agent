from typing import Any

from idfpy.models.schedules import (
    ScheduleCompact,
    ScheduleCompactDataItem,
    ScheduleTypeLimits,
)

from src.mcp.state import ConfigState
from src.mcp.tools.base import BaseTool, normalize_payload


def _flatten_schedule_data(data: Any) -> list[str]:
    if not data:
        return []
    if isinstance(data, list) and all(isinstance(item, str) for item in data):
        return data
    result: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            result.append(str(item))
            continue
        if "field" in item:
            result.append(str(item["field"]))
            continue
        through = item["Through"] if "Through" in item else item.get("through")
        if through:
            result.append(f"Through: {through}")
        for day in item.get("Days", item.get("days", [])) or []:
            day_type = day["For"] if "For" in day else day.get("for")
            if day_type:
                result.append(f"For: {day_type}")
            for time_row in day.get("Times", day.get("times", [])) or []:
                until = time_row.get("Until") or time_row.get("until") or {}
                time = until["Time"] if "Time" in until else until.get("time")
                value = until["Value"] if "Value" in until else until.get("value")
                if time is not None and value is not None:
                    result.append(f"Until: {time}, {value}")
    return result


class ScheduleTypeLimitsTool(BaseTool):
    def __init__(self, state: ConfigState):
        super().__init__(state, "ScheduleTypeLimits")

    @property
    def object_types(self) -> tuple[str, ...]:
        return ("ScheduleTypeLimits",)

    def _create_model(self, data: dict[str, Any]) -> ScheduleTypeLimits:
        payload = normalize_payload(data)
        for key in ("lower_limit_value", "upper_limit_value"):
            if payload.get(key) == "":
                payload[key] = None
        return ScheduleTypeLimits(**payload)

    def _get_name(self, instance: ScheduleTypeLimits) -> str:
        return instance.name

    def _check_references(self, name: str) -> list[str]:
        refs = []
        for schedule in self.state.idf.all_of_type("Schedule:Compact").values():
            if schedule.schedule_type_limits_name == name:
                refs.append(f"ScheduleCompact:{schedule.name}")
        return refs


class ScheduleCompactTool(BaseTool):
    def __init__(self, state: ConfigState):
        super().__init__(state, "ScheduleCompact")

    @property
    def object_types(self) -> tuple[str, ...]:
        return ("Schedule:Compact", "ScheduleCompact")

    def _create_model(self, data: dict[str, Any]) -> ScheduleCompact:
        payload = normalize_payload(data)
        raw_data = payload.pop("data", payload.pop("times", None))
        if raw_data is not None:
            payload["data"] = [
                ScheduleCompactDataItem(field=item)
                for item in _flatten_schedule_data(raw_data)
            ]
        return ScheduleCompact(**payload)

    def _get_name(self, instance: ScheduleCompact) -> str:
        return instance.name

    def _check_references(self, name: str) -> list[str]:
        refs = []
        for thermostat in self.state.idf.all_of_type("HVACTemplate:Thermostat").values():
            if thermostat.heating_setpoint_schedule_name == name:
                refs.append(f"Thermostat:{thermostat.name}")
            if thermostat.cooling_setpoint_schedule_name == name:
                refs.append(f"Thermostat:{thermostat.name}")
        for ils in self.state.idf.all_of_type(
            "HVACTemplate:Zone:IdealLoadsAirSystem"
        ).values():
            if getattr(ils, "system_availability_schedule_name", None) == name:
                refs.append(f"IdealLoadsSystem:{ils.zone_name}")
        for people in self.state.idf.all_of_type("People").values():
            if people.number_of_people_schedule_name == name:
                refs.append(f"People:{people.name}")
            if people.activity_level_schedule_name == name:
                refs.append(f"People:{people.name}")
        for light in self.state.idf.all_of_type("Lights").values():
            if light.schedule_name == name:
                refs.append(f"Lights:{light.name}")
        return refs
