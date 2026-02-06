from fastmcp import FastMCP
from pydantic import Field

from src.mcp.api.common import ToolInput, to_payload
from src.mcp.tools import LightTool, PeopleTool


class PeopleCreateInput(ToolInput):
    name: str = Field(alias="Name")
    zone_or_zonelist_or_space_or_spacelist_name: str = Field(
        alias="Zone or ZoneList or Space or SpaceList Name"
    )
    number_of_people_schedule_name: str = Field(alias="Number of People Schedule Name")
    activity_level_schedule_name: str = Field(alias="Activity Level Schedule Name")
    number_of_people_calculation_method: str = Field(
        default="People",
        alias="Number of People Calculation Method",
    )
    number_of_people: float | None = Field(default=None, alias="Number of People")
    people_per_floor_area: float | None = Field(default=None, alias="People per Floor Area")
    floor_area_per_person: float | None = Field(default=None, alias="Floor Area per Person")
    fraction_radiant: float = Field(default=0.3, alias="Fraction Radiant")
    sensible_heat_fraction: float | str = Field(
        default="Autocalculate",
        alias="Sensible Heat Fraction",
    )


class PeopleUpdateInput(ToolInput):
    name: str = Field(alias="Name")
    zone_or_zonelist_or_space_or_spacelist_name: str | None = Field(
        default=None,
        alias="Zone or ZoneList or Space or SpaceList Name",
    )
    number_of_people_schedule_name: str | None = Field(
        default=None,
        alias="Number of People Schedule Name",
    )
    activity_level_schedule_name: str | None = Field(
        default=None,
        alias="Activity Level Schedule Name",
    )
    number_of_people_calculation_method: str | None = Field(
        default=None,
        alias="Number of People Calculation Method",
    )
    number_of_people: float | None = Field(default=None, alias="Number of People")
    people_per_floor_area: float | None = Field(default=None, alias="People per Floor Area")
    floor_area_per_person: float | None = Field(default=None, alias="Floor Area per Person")
    fraction_radiant: float | None = Field(default=None, alias="Fraction Radiant")
    sensible_heat_fraction: float | str | None = Field(
        default=None,
        alias="Sensible Heat Fraction",
    )


class LightCreateInput(ToolInput):
    name: str = Field(alias="Name")
    zone_or_zone_list_or_space_or_space_list_name: str = Field(
        alias="Zone or ZoneList or Space or SpaceList Name"
    )
    schedule_name: str = Field(alias="Schedule Name")
    design_level_calculation_method: str = Field(
        default="LightingLevel",
        alias="Design Level Calculation Method",
    )
    lighting_level: float | None = Field(default=None, alias="Lighting Level")
    watts_per_floor_area: float | None = Field(default=None, alias="Watts per Floor Area")
    watts_per_person: float | None = Field(default=None, alias="Watts per Person")
    return_air_fraction: float = Field(default=0.0, alias="Return Air Fraction")
    fraction_radiant: float = Field(default=0.0, alias="Fraction Radiant")
    fraction_visible: float = Field(default=0.0, alias="Fraction Visible")
    fraction_replaceable: float = Field(default=1.0, alias="Fraction Replaceable")
    end_use_subcategory: str = Field(default="General", alias="End Use Subcategory")


class LightUpdateInput(ToolInput):
    name: str = Field(alias="Name")
    zone_or_zone_list_or_space_or_space_list_name: str | None = Field(
        default=None,
        alias="Zone or ZoneList or Space or SpaceList Name",
    )
    schedule_name: str | None = Field(default=None, alias="Schedule Name")
    design_level_calculation_method: str | None = Field(
        default=None,
        alias="Design Level Calculation Method",
    )
    lighting_level: float | None = Field(default=None, alias="Lighting Level")
    watts_per_floor_area: float | None = Field(default=None, alias="Watts per Floor Area")
    watts_per_person: float | None = Field(default=None, alias="Watts per Person")
    return_air_fraction: float | None = Field(default=None, alias="Return Air Fraction")
    fraction_radiant: float | None = Field(default=None, alias="Fraction Radiant")
    fraction_visible: float | None = Field(default=None, alias="Fraction Visible")
    fraction_replaceable: float | None = Field(default=None, alias="Fraction Replaceable")
    end_use_subcategory: str | None = Field(default=None, alias="End Use Subcategory")


def register_load_tools(
    mcp: FastMCP,
    people_tool: PeopleTool,
    light_tool: LightTool,
) -> None:
    @mcp.tool
    def create_people(
        name: str,
        zone_or_zonelist_or_space_or_spacelist_name: str,
        number_of_people_schedule_name: str,
        activity_level_schedule_name: str,
        number_of_people_calculation_method: str = "People",
        number_of_people: float | None = None,
        people_per_floor_area: float | None = None,
        floor_area_per_person: float | None = None,
        fraction_radiant: float = 0.3,
        sensible_heat_fraction: float | str = "Autocalculate",
    ) -> dict:
        payload = to_payload(
            PeopleCreateInput.model_validate(
                {
                    "name": name,
                    "zone_or_zonelist_or_space_or_spacelist_name": zone_or_zonelist_or_space_or_spacelist_name,
                    "number_of_people_schedule_name": number_of_people_schedule_name,
                    "activity_level_schedule_name": activity_level_schedule_name,
                    "number_of_people_calculation_method": number_of_people_calculation_method,
                    "number_of_people": number_of_people,
                    "people_per_floor_area": people_per_floor_area,
                    "floor_area_per_person": floor_area_per_person,
                    "fraction_radiant": fraction_radiant,
                    "sensible_heat_fraction": sensible_heat_fraction,
                }
            )
        )
        return people_tool.create(payload).to_mcp_response()

    @mcp.tool
    def get_people(name: str) -> dict:
        return people_tool.read(name).to_mcp_response()

    @mcp.tool
    def update_people(
        name: str,
        zone_or_zonelist_or_space_or_spacelist_name: str | None = None,
        number_of_people_schedule_name: str | None = None,
        activity_level_schedule_name: str | None = None,
        number_of_people_calculation_method: str | None = None,
        number_of_people: float | None = None,
        people_per_floor_area: float | None = None,
        floor_area_per_person: float | None = None,
        fraction_radiant: float | None = None,
        sensible_heat_fraction: float | str | None = None,
    ) -> dict:
        payload = to_payload(
            PeopleUpdateInput.model_validate(
                {
                    "name": name,
                    "zone_or_zonelist_or_space_or_spacelist_name": zone_or_zonelist_or_space_or_spacelist_name,
                    "number_of_people_schedule_name": number_of_people_schedule_name,
                    "activity_level_schedule_name": activity_level_schedule_name,
                    "number_of_people_calculation_method": number_of_people_calculation_method,
                    "number_of_people": number_of_people,
                    "people_per_floor_area": people_per_floor_area,
                    "floor_area_per_person": floor_area_per_person,
                    "fraction_radiant": fraction_radiant,
                    "sensible_heat_fraction": sensible_heat_fraction,
                }
            )
        )
        return people_tool.update(name, payload).to_mcp_response()

    @mcp.tool
    def delete_people(name: str) -> dict:
        return people_tool.delete(name).to_mcp_response()

    @mcp.tool
    def list_people() -> dict:
        return people_tool.list_all().to_mcp_response()

    @mcp.tool
    def create_light(
        name: str,
        zone_or_zone_list_or_space_or_space_list_name: str,
        schedule_name: str,
        design_level_calculation_method: str = "LightingLevel",
        lighting_level: float | None = None,
        watts_per_floor_area: float | None = None,
        watts_per_person: float | None = None,
        return_air_fraction: float = 0.0,
        fraction_radiant: float = 0.0,
        fraction_visible: float = 0.0,
        fraction_replaceable: float = 1.0,
        end_use_subcategory: str = "General",
    ) -> dict:
        payload = to_payload(
            LightCreateInput.model_validate(
                {
                    "name": name,
                    "zone_or_zone_list_or_space_or_space_list_name": zone_or_zone_list_or_space_or_space_list_name,
                    "schedule_name": schedule_name,
                    "design_level_calculation_method": design_level_calculation_method,
                    "lighting_level": lighting_level,
                    "watts_per_floor_area": watts_per_floor_area,
                    "watts_per_person": watts_per_person,
                    "return_air_fraction": return_air_fraction,
                    "fraction_radiant": fraction_radiant,
                    "fraction_visible": fraction_visible,
                    "fraction_replaceable": fraction_replaceable,
                    "end_use_subcategory": end_use_subcategory,
                }
            )
        )
        return light_tool.create(payload).to_mcp_response()

    @mcp.tool
    def get_light(name: str) -> dict:
        return light_tool.read(name).to_mcp_response()

    @mcp.tool
    def update_light(
        name: str,
        zone_or_zone_list_or_space_or_space_list_name: str | None = None,
        schedule_name: str | None = None,
        design_level_calculation_method: str | None = None,
        lighting_level: float | None = None,
        watts_per_floor_area: float | None = None,
        watts_per_person: float | None = None,
        return_air_fraction: float | None = None,
        fraction_radiant: float | None = None,
        fraction_visible: float | None = None,
        fraction_replaceable: float | None = None,
        end_use_subcategory: str | None = None,
    ) -> dict:
        payload = to_payload(
            LightUpdateInput.model_validate(
                {
                    "name": name,
                    "zone_or_zone_list_or_space_or_space_list_name": zone_or_zone_list_or_space_or_space_list_name,
                    "schedule_name": schedule_name,
                    "design_level_calculation_method": design_level_calculation_method,
                    "lighting_level": lighting_level,
                    "watts_per_floor_area": watts_per_floor_area,
                    "watts_per_person": watts_per_person,
                    "return_air_fraction": return_air_fraction,
                    "fraction_radiant": fraction_radiant,
                    "fraction_visible": fraction_visible,
                    "fraction_replaceable": fraction_replaceable,
                    "end_use_subcategory": end_use_subcategory,
                }
            )
        )
        return light_tool.update(name, payload).to_mcp_response()

    @mcp.tool
    def delete_light(name: str) -> dict:
        return light_tool.delete(name).to_mcp_response()

    @mcp.tool
    def list_lights() -> dict:
        return light_tool.list_all().to_mcp_response()
