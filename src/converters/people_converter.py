from idfpy import IDF
from idfpy.models import People

from src.converters.base_converter import BaseConverter
from src.validator.data_model import PeopleSchema


class PeopleConverter(BaseConverter):
    def __init__(self, idf: IDF):
        super().__init__(idf)

    def convert(self, data: dict):
        self.logger.info("Converting people data...")
        for people in data.get("People", []):
            try:
                validated_people = self.validate(people)
                self._add_to_idf(validated_people)
            except Exception as e:
                self.state["failed"] += 1
                self.logger.error("Error converting people data: {}", e)
                continue

    def _add_to_idf(self, val_data: PeopleSchema):
        if self.idf.has("People", val_data.name):
            self.logger.warning(
                "People with name {} already exists in IDF. Skipping addition.",
                val_data.name,
            )
            self.state["skipped"] += 1
            return
        self.idf.add(
            People(
                name=val_data.name,
                zone_or_zonelist_or_space_or_spacelist_name=val_data.zone_or_zonelist_or_space_or_spacelist_name,
                number_of_people_schedule_name=val_data.number_of_people_schedule_name,
                number_of_people_calculation_method=val_data.number_of_people_calculation_method,
                number_of_people=val_data.number_of_people,
                people_per_floor_area=val_data.people_per_floor_area,
                floor_area_per_person=val_data.floor_area_per_person,
                fraction_radiant=val_data.fraction_radiant,
                sensible_heat_fraction=val_data.sensible_heat_fraction,
                activity_level_schedule_name=val_data.activity_level_schedule_name,
                carbon_dioxide_generation_rate=val_data.carbon_dioxide_generation_rate,
                enable_ashrae_55_comfort_warnings=val_data.enable_ashrae_55_comfort_warnings,
                mean_radiant_temperature_calculation_type=val_data.mean_radiant_temperature_calculation_type,
                surface_name_angle_factor_list_name=val_data.surface_name_angle_factor_list_name
                or None,
                work_efficiency_schedule_name=val_data.work_efficiency_schedule_name
                or None,
                clothing_insulation_calculation_method=val_data.clothing_insulation_calculation_method,
                clothing_insulation_calculation_method_schedule_name=val_data.clothing_insulation_calculation_method_schedule_name
                or None,
                clothing_insulation_schedule_name=val_data.clothing_insulation_schedule_name
                or None,
                air_velocity_schedule_name=val_data.air_velocity_schedule_name or None,
                thermal_comfort_model_1_type=val_data.thermal_comfort_model_1_type
                or None,
                thermal_comfort_model_2_type=val_data.thermal_comfort_model_2_type
                or None,
                thermal_comfort_model_3_type=val_data.thermal_comfort_model_3_type
                or None,
                thermal_comfort_model_4_type=val_data.thermal_comfort_model_4_type
                or None,
                thermal_comfort_model_5_type=val_data.thermal_comfort_model_5_type
                or None,
                thermal_comfort_model_6_type=val_data.thermal_comfort_model_6_type
                or None,
                thermal_comfort_model_7_type=val_data.thermal_comfort_model_7_type
                or None,
                ankle_level_air_velocity_schedule_name=val_data.ankle_level_air_velocity_schedule_name
                or None,
                cold_stress_temperature_threshold=val_data.cold_stress_temperature_threshold,
                heat_stress_temperature_threshold=val_data.heat_stress_temperature_threshold,
            )
        )
        self.state["success"] += 1
        self.logger.success("People with name {} added to IDF.", val_data.name)

    def validate(self, data: dict) -> PeopleSchema:
        return PeopleSchema.model_validate(data)
