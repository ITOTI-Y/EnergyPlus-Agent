from fastmcp import FastMCP
from idfpy.models.schedules import ScheduleCompact, ScheduleTypeLimits

from src.mcp.api.common import Outcome, dump, given, model_tool
from src.modeling import objects
from src.modeling.schedules import ThroughSchema, compact_schedule, schedule_fields
from src.state.config_state import ConfigState


def register_schedule_tools(mcp: FastMCP, state: ConfigState) -> None:
    """Register ScheduleTypeLimits and Schedule:Compact tools."""
    idf = state.idf
    tool = model_tool(mcp)

    @tool
    def create_schedule_type_limits(
        name: str,
        lower_limit_value: float | None = None,
        upper_limit_value: float | None = None,
        numeric_type: str = "Continuous",
        unit_type: str = "Dimensionless",
    ) -> Outcome:
        """Create a ScheduleTypeLimits.

        Args:
            name: Unique name, e.g. Fraction or Temperature.
            lower_limit_value: Minimum value; omit for unbounded.
            upper_limit_value: Maximum value; omit for unbounded.
            numeric_type: Continuous or Discrete.
            unit_type: Dimensionless, Temperature, ActivityLevel, ...
        """
        limits = ScheduleTypeLimits.model_validate(
            {
                "name": name,
                "lower_limit_value": lower_limit_value,
                "upper_limit_value": upper_limit_value,
                "numeric_type": numeric_type,
                "unit_type": unit_type,
            }
        )
        return (
            f"ScheduleTypeLimits '{name}' created.",
            dump(objects.create(idf, limits)),
        )

    @tool
    def get_schedule_type_limits(name: str) -> Outcome:
        """Read a ScheduleTypeLimits by name."""
        return (
            f"ScheduleTypeLimits '{name}' read.",
            dump(objects.get(idf, ScheduleTypeLimits, name)),
        )

    @tool
    def update_schedule_type_limits(
        name: str,
        new_name: str | None = None,
        lower_limit_value: float | None = None,
        upper_limit_value: float | None = None,
        numeric_type: str | None = None,
        unit_type: str | None = None,
    ) -> Outcome:
        """Update a ScheduleTypeLimits; a new name is applied to its schedules."""
        limits = objects.update(
            idf,
            objects.get(idf, ScheduleTypeLimits, name),
            given(
                name=new_name,
                lower_limit_value=lower_limit_value,
                upper_limit_value=upper_limit_value,
                numeric_type=numeric_type,
                unit_type=unit_type,
            ),
        )
        return f"ScheduleTypeLimits '{name}' updated.", dump(limits)

    @tool
    def delete_schedule_type_limits(name: str) -> Outcome:
        """Delete a ScheduleTypeLimits; refused while a schedule uses it."""
        objects.delete(idf, objects.get(idf, ScheduleTypeLimits, name), name)
        return f"ScheduleTypeLimits '{name}' deleted.", None

    @tool
    def list_schedule_type_limits() -> Outcome:
        """List all ScheduleTypeLimits."""
        return "Listed ScheduleTypeLimits.", objects.dumps(
            idf.all_of_type(ScheduleTypeLimits)
        )

    @tool
    def create_schedule_compact(
        name: str, schedule_type_limits_name: str, times: list[ThroughSchema]
    ) -> Outcome:
        """Create a Schedule:Compact from Through / For / Until blocks.

        Args:
            name: Unique schedule name.
            schedule_type_limits_name: Existing ScheduleTypeLimits.
            times: Periods in date order; the last Through is 12/31 and every
                day's last Until time is 24:00.
        """
        schedule = compact_schedule(name, schedule_type_limits_name, times)
        return (
            f"Schedule:Compact '{name}' created.",
            dump(objects.create(idf, schedule)),
        )

    @tool
    def get_schedule_compact(name: str) -> Outcome:
        """Read a Schedule:Compact by name."""
        return (
            f"Schedule:Compact '{name}' read.",
            dump(objects.get(idf, ScheduleCompact, name)),
        )

    @tool
    def update_schedule_compact(
        name: str,
        new_name: str | None = None,
        schedule_type_limits_name: str | None = None,
        times: list[ThroughSchema] | None = None,
    ) -> Outcome:
        """Update a Schedule:Compact; a new name is applied to every user."""
        changes = given(
            name=new_name, schedule_type_limits_name=schedule_type_limits_name
        )
        if times is not None:
            changes["data"] = [{"field": f} for f in schedule_fields(times)]
        schedule = objects.update(idf, objects.get(idf, ScheduleCompact, name), changes)
        return f"Schedule:Compact '{name}' updated.", dump(schedule)

    @tool
    def delete_schedule_compact(name: str) -> Outcome:
        """Delete a Schedule:Compact; refused while another object uses it."""
        objects.delete(idf, objects.get(idf, ScheduleCompact, name), name)
        return f"Schedule:Compact '{name}' deleted.", None

    @tool
    def list_schedule_compacts() -> Outcome:
        """List all Schedule:Compact."""
        return "Listed Schedule:Compact.", objects.dumps(
            idf.all_of_type(ScheduleCompact)
        )
