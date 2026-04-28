from idfpy import IDF
from idfpy.models.internal_gains import Lights

from src.converters.base_converter import BaseConverter
from src.validator.data_model import LightSchema


class LightConverter(BaseConverter):
    def __init__(self, idf: IDF):
        super().__init__(idf)

    def convert(self, data: dict):
        self.logger.info("Converting light data...")
        for light in data.get("Light", []):
            try:
                validated_light = self.validate(light)
                self._add_to_idf(validated_light)
            except Exception as e:
                self.state["failed"] += 1
                self.logger.error("Error converting light data: {}", e)
                continue

    def _add_to_idf(self, val_data: LightSchema):
        if self.idf.has("Lights", val_data.name):
            self.logger.warning(
                "Light with name {} already exists in IDF. Skipping addition.",
                val_data.name,
            )
            self.state["skipped"] += 1
            return
        self.idf.add(Lights(
            name=val_data.name,
            zone_or_zonelist_or_space_or_spacelist_name=val_data.zone_or_zone_list_or_space_or_space_list_name,
            schedule_name=val_data.schedule_name,
            design_level_calculation_method=val_data.design_level_calculation_method,
            lighting_level=val_data.lighting_level,
            watts_per_floor_area=val_data.watts_per_floor_area,
            watts_per_person=val_data.watts_per_person,
            return_air_fraction=val_data.return_air_fraction,
            fraction_radiant=val_data.fraction_radiant,
            fraction_visible=val_data.fraction_visible,
            fraction_replaceable=val_data.fraction_replaceable,
            end_use_subcategory=val_data.end_use_subcategory,
            return_air_fraction_calculated_from_plenum_temperature=val_data.return_air_fraction_calculated_from_plenum_temperature,
            return_air_fraction_function_of_plenum_temperature_coefficient_1=val_data.return_air_fraction_function_of_plenum_temperature_coefficient_1,
            return_air_fraction_function_of_plenum_temperature_coefficient_2=val_data.return_air_fraction_function_of_plenum_temperature_coefficient_2,
            return_air_heat_gain_node_name=val_data.return_air_heat_gain_node_name or None,
            exhaust_air_heat_gain_node_name=val_data.exhaust_air_heat_gain_node_name or None,
        ))
        self.state["success"] += 1
        self.logger.success("Light with name {} added to IDF.", val_data.name)

    def validate(self, data: dict) -> LightSchema:
        return LightSchema.model_validate(data)
