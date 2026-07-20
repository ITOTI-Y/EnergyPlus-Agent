import json
from typing import Any, Literal

from idfpy.models import (
    HVACTemplateThermostat,
    HVACTemplateZoneIdealLoadsAirSystem,
    Lights,
    People,
    ScheduleCompact,
    ScheduleCompactDataItem,
    ScheduleTypeLimits,
)
from langchain_core.tools import BaseTool, tool

from src.mcp.state import ConfigState
from src.validator.data_model import ScheduleCompactSchema


def _ok(msg: str, data=None) -> str:
    return json.dumps({"success": True, "message": msg, "data": data})


def _err(msg: str, data=None) -> str:
    return json.dumps({"success": False, "message": msg, "data": data})


def make_schedule_tools(config: ConfigState, rag=None) -> list[BaseTool]:
    idf = config._idf

    @tool
    def create_schedule_type_limits(
        name: str,
        lower_limit_value: float | None = None,
        upper_limit_value: float | None = None,
        numeric_type: Literal["Continuous", "Discrete"] = "Continuous",
        unit_type: Literal[
            "",
            "ActivityLevel",
            "Angle",
            "Availability",
            "Capacity",
            "Control",
            "ConvectionCoefficient",
            "DeltaTemperature",
            "Dimensionless",
            "Mode",
            "Percent",
            "Power",
            "PrecipitationRate",
            "Temperature",
            "Velocity",
        ]
        | None = "Dimensionless",
    ) -> str:
        """Create a ScheduleTypeLimits.

        Args:
            name: Unique name (e.g., 'Fraction', 'Temperature', 'OnOff').
            lower_limit_value: Minimum allowed value (None = unbounded).
            upper_limit_value: Maximum allowed value (None = unbounded).
            numeric_type: CONTINUOUS or DISCRETE.
            unit_type: EnergyPlus unit category (Dimensionless / Temperature / Power / ...).
        """
        if idf is None:
            raise ValueError("IDF is None")
        if idf.has(ScheduleTypeLimits, name):
            return _err(f"ScheduleTypeLimits '{name}' already exists.")
        try:
            idf.add(
                ScheduleTypeLimits(
                    name=name,
                    lower_limit_value=lower_limit_value,
                    upper_limit_value=upper_limit_value,
                    numeric_type=numeric_type,
                    unit_type=unit_type,
                )
            )
            data = idf.get(ScheduleTypeLimits, name)
            if data is None:
                raise ValueError("ScheduleTypeLimits not found")
            return _ok(
                f"ScheduleTypeLimits '{name}' created successfully.",
                data.model_dump(),
            )
        except Exception as e:
            return _err(f"Error creating ScheduleTypeLimits '{name}': {e}")

    @tool
    def create_schedule_compact(
        name: str,
        schedule_type_limits_name: str,
        data: list[dict[str, Any]],
    ) -> str:
        """Create a Schedule:Compact.

        Args:
            name: Unique schedule name.
            schedule_type_limits_name: Existing ScheduleTypeLimits name.
            data: Nested schedule structure. Each element is one "Through"-block:

                {
                  "Through": "MM/DD",               # last block must be "12/31"
                  "Days": [
                    {
                      "For": "<DayType>",           # Weekdays / Weekends / Saturday /
                                                    # Sunday / AllDays / AllOtherDays /
                                                    # SummerDesignDay / WinterDesignDay /
                                                    # Monday...Friday / Holidays /
                                                    # CustomDay1 / CustomDay2
                      "Times": [
                        {"Until": {"Time": "HH:MM", "Value": <float>}},
                        ...
                        {"Until": {"Time": "24:00", "Value": <float>}},  # last must be 24:00
                      ],
                    },
                    ...  # additional day-type blocks under the same Through
                  ],
                }

                Example (office fraction schedule, weekdays 8-18 at 1.0, else 0.0):

                [
                  {
                    "Through": "12/31",
                    "Days": [
                      {"For": "Weekdays", "Times": [
                        {"Until": {"Time": "08:00", "Value": 0.0}},
                        {"Until": {"Time": "18:00", "Value": 1.0}},
                        {"Until": {"Time": "24:00", "Value": 0.0}},
                      ]},
                      {"For": "AllOtherDays", "Times": [
                        {"Until": {"Time": "24:00", "Value": 0.0}},
                      ]},
                    ],
                  },
                ]
        """
        if idf is None:
            raise ValueError("IDF is None")
        if idf.has(ScheduleCompact, name):
            return _err(f"Schedule:Compact '{name}' already exists.")
        try:
            # Validate and flatten the nested data structure
            validated = ScheduleCompactSchema.model_validate(
                {
                    "Name": name,
                    "Schedule Type Limits Name": schedule_type_limits_name,
                    "Data": data,
                }
            )
            idf.add(
                ScheduleCompact(
                    name=validated.name,
                    schedule_type_limits_name=validated.schedule_type_limits_name,
                    data=[ScheduleCompactDataItem(field=v) for v in validated.data],
                )
            )
            obj = idf.get("Schedule:Compact", name)
            return _ok(
                f"Schedule:Compact '{name}' created successfully.",
                obj.model_dump() if obj else None,
            )
        except Exception as e:
            return _err(f"Error creating Schedule:Compact '{name}': {e}")

    @tool
    def list_schedules() -> str:
        """List all Schedule:Compact objects."""
        if idf is None:
            raise ValueError("IDF is None")
        items = [s.model_dump() for s in idf.all_of_type(ScheduleCompact).values()]
        return _ok(f"Listed {len(items)} Schedule:Compact objects.", items)

    @tool
    def list_schedule_type_limits() -> str:
        """List all ScheduleTypeLimits objects."""
        if idf is None:
            raise ValueError("IDF is None")
        items = [s.model_dump() for s in idf.all_of_type(ScheduleTypeLimits).values()]
        return _ok(f"Listed {len(items)} ScheduleTypeLimits objects.", items)

    @tool
    def get_schedule(name: str) -> str:
        """Read a Schedule:Compact by name."""
        if idf is None:
            raise ValueError("IDF is None")
        obj = idf.get(ScheduleCompact, name)
        if obj is None:
            return _err(f"Schedule:Compact '{name}' not found.")
        return _ok(f"Schedule:Compact '{name}' read successfully.", obj.model_dump())

    @tool
    def update_schedule_compact(
        name: str,
        schedule_type_limits_name: str | None = None,
        data: list[dict[str, Any]] | None = None,
    ) -> str:
        """Update an existing Schedule:Compact's type-limits and/or data.

        Only non-None fields are rewritten. Pass ``data`` to replace the
        entire schedule time-value structure (same nested format as
        create_schedule_compact).

        Args:
            name: Existing Schedule:Compact name.
            schedule_type_limits_name: New ScheduleTypeLimits name.
            data: New nested Through/Days/Times structure (replaces all).
        """
        if idf is None:
            raise ValueError("IDF is None")
        obj = idf.get(ScheduleCompact, name)
        if obj is None:
            return _err(f"Schedule:Compact '{name}' not found.")
        try:
            if schedule_type_limits_name is not None:
                obj.schedule_type_limits_name = schedule_type_limits_name
            if data is not None:
                # Validate the new data via the schema, then rebuild items
                validated = ScheduleCompactSchema.model_validate(
                    {
                        "Name": name,
                        "Schedule Type Limits Name": obj.schedule_type_limits_name,
                        "Data": data,
                    }
                )
                obj.data = [ScheduleCompactDataItem(field=v) for v in validated.data]
            return _ok(
                f"Schedule:Compact '{name}' updated successfully.", obj.model_dump()
            )
        except Exception as e:
            return _err(f"Error updating Schedule:Compact '{name}': {e}")

    @tool
    def delete_schedule(name: str) -> str:
        """Delete a Schedule:Compact. Fails if referenced."""
        if idf is None:
            raise ValueError("IDF is None")
        if not idf.has(ScheduleCompact, name):
            return _err(f"Schedule:Compact '{name}' not found.")
        refs = []
        for t in idf.all_of_type(HVACTemplateThermostat).values():
            if t.heating_setpoint_schedule_name == name:
                refs.append(f"Thermostat:{t.name}")
            if t.cooling_setpoint_schedule_name == name:
                refs.append(f"Thermostat:{t.name}")
        for ils in idf.all_of_type(HVACTemplateZoneIdealLoadsAirSystem).values():
            if ils.system_availability_schedule_name == name:
                refs.append(f"IdealLoadsSystem:{ils.zone_name}")
        for p in idf.all_of_type(People).values():
            if p.number_of_people_schedule_name == name:
                refs.append(f"People:{p.name}")
            if p.activity_level_schedule_name == name:
                refs.append(f"People:{p.name}")
        for lt in idf.all_of_type(Lights).values():
            if lt.schedule_name == name:
                refs.append(f"Lights:{lt.name}")
        if refs:
            return _err(
                f"Schedule:Compact '{name}' is referenced by other components.",
                {"references": refs},
            )
        idf.remove("Schedule:Compact", name)
        return _ok(f"Schedule:Compact '{name}' deleted successfully.")

    tools = [
        create_schedule_type_limits,
        create_schedule_compact,
        list_schedules,
        list_schedule_type_limits,
        get_schedule,
        update_schedule_compact,
        delete_schedule,
    ]
    if rag is not None:
        from src.agent.tools.rag_tools import (
            TABLE_SCHEDULE_COMPACT,
            TABLE_SCHEDULE_TYPE_LIMITS,
            make_rag_tool,
        )

        tools.append(
            make_rag_tool([TABLE_SCHEDULE_TYPE_LIMITS, TABLE_SCHEDULE_COMPACT], rag=rag)
        )
    return tools
