from idfpy.models.schedules import ScheduleCompact, ScheduleTypeLimits
from langchain_core.tools import BaseTool

from src.agent.tools._share import list_tool, model_tool, ok
from src.modeling import objects
from src.modeling.schedules import ThroughSchema, compact_schedule
from src.state.config_state import ConfigState


def make_schedule_tools(config: ConfigState) -> list[BaseTool]:
    idf = config.idf

    @model_tool
    def create_schedule_type_limits(
        name: str,
        lower_limit_value: float | None = None,
        upper_limit_value: float | None = None,
        numeric_type: str = "Continuous",
        unit_type: str = "Dimensionless",
    ) -> str:
        """Create a ScheduleTypeLimits.

        Args:
            name: Unique name (e.g., 'Fraction', 'Temperature', 'OnOff').
            lower_limit_value: Minimum allowed value (None = unbounded).
            upper_limit_value: Maximum allowed value (None = unbounded).
            numeric_type: Continuous or Discrete.
            unit_type: EnergyPlus unit category (Dimensionless / Temperature / ActivityLevel / ...).
        """
        limits = objects.create(
            idf,
            ScheduleTypeLimits.model_validate(
                {
                    "name": name,
                    "lower_limit_value": lower_limit_value,
                    "upper_limit_value": upper_limit_value,
                    "numeric_type": numeric_type,
                    "unit_type": unit_type,
                }
            ),
        )
        return ok(
            f"ScheduleTypeLimits '{name}' created.",
            limits.model_dump(exclude_none=True),
        )

    @model_tool
    def create_schedule_compact(
        name: str,
        schedule_type_limits_name: str,
        data: list[ThroughSchema],
    ) -> str:
        """Create a Schedule:Compact from Through / For / Until blocks.

        Args:
            name: Unique schedule name.
            schedule_type_limits_name: Existing ScheduleTypeLimits name.
            data: Periods in date order; the last "Through" is "12/31" and
                every day's last "Until" time is "24:00". Example (weekdays
                8-18 at 1.0, else 0.0):
                  [{"Through": "12/31", "Days": [
                     {"For": "Weekdays", "Times": [
                        {"Until": {"Time": "08:00", "Value": 0.0}},
                        {"Until": {"Time": "18:00", "Value": 1.0}},
                        {"Until": {"Time": "24:00", "Value": 0.0}}]},
                     {"For": "AllOtherDays", "Times": [
                        {"Until": {"Time": "24:00", "Value": 0.0}}]}]}]
        """
        schedule = objects.create(
            idf, compact_schedule(name, schedule_type_limits_name, data)
        )
        return ok(
            f"Schedule:Compact '{name}' created.",
            schedule.model_dump(exclude_none=True),
        )

    @model_tool
    def get_schedule(name: str) -> str:
        """Read a Schedule:Compact by name."""
        schedule = objects.get(idf, ScheduleCompact, name)
        return ok(
            f"Schedule:Compact '{name}' read.", schedule.model_dump(exclude_none=True)
        )

    @model_tool
    def delete_schedule(name: str) -> str:
        """Delete a Schedule:Compact; refused while another object uses it."""
        objects.delete(idf, objects.get(idf, ScheduleCompact, name), name)
        return ok(f"Schedule:Compact '{name}' deleted.")

    return [
        create_schedule_type_limits,
        create_schedule_compact,
        list_tool(idf, "list_schedules", ScheduleCompact, "List all Schedule:Compact."),
        list_tool(
            idf,
            "list_schedule_type_limits",
            ScheduleTypeLimits,
            "List all ScheduleTypeLimits.",
        ),
        get_schedule,
        delete_schedule,
    ]
