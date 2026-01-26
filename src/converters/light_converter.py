from typing import Any

from eppy.modeleditor import IDF

from src.converters.base_converter import BaseConverter
from src.utils.logging import get_logger
from src.validator.data_model import LightsSchema

class LightConverter(BaseConverter):
    """
    Converts Lights definitions from YAML data into IDF objects.
    Refactored to match HVACConverter style and use direct field assignment.
    """
    def __init__(self, idf: IDF):
        super().__init__(idf)
        self.logger = get_logger(__name__)

    def convert(self, data: dict[str, Any]) -> None:
        """
        Processes the Lights list from YAML, validating and adding each component.
        """
        self.logger.info("Light Converter Starting...")
        lights_list = data.get("Lights") or data.get("lights", [])
        if not lights_list:
            self.logger.info("No Lights data found in YAML.")
            return
        for light_data in lights_list:
            try:
                validated_light = self.validate(light_data)
                self._add_to_idf(validated_light)
            except Exception as e:
                self.state["failed"] += 1
                self.logger.error(
                    f"Failed to convert Lights '{light_data.get('Name', 'N/A')}': {e}",
                    exc_info=True,
                )
                continue

    def validate(self, data: dict[str, Any]) -> LightsSchema:
        """Validates a single Light entry against the LightsSchema."""
        return LightsSchema.model_validate(data)

    def _add_to_idf(self, val_data: LightsSchema) -> None:
        """
        Adds the validated Lights object to the IDF file.
        Includes existence check and dependency check.
        """
        try:
            if self.idf.getobject("LIGHTS", val_data.name):
                self.logger.warning(
                    f"Lights object with name '{val_data.name}' already exists. Skipping addition."
                )
                self.state["skipped"] += 1
                return

            self.logger.debug(f"Adding Lights '{val_data.name}' to IDF.")

            if not self.idf.getobject("ZONE", val_data.zone_name):
                 raise ValueError(
                    f"Zone '{val_data.zone_name}' referenced in Lights '{val_data.name}' "
                    f"does not exist in IDF. Please add the Zone first."
                )

            self.idf.newidfobject(
                "LIGHTS",
                Name=val_data.name,
                Zone_or_ZoneList_or_Space_or_SpaceList_Name=val_data.zone_name,
                Schedule_Name=val_data.schedule_name,
                Lighting_Level=val_data.lighting_level,
                Watts_per_Floor_Area=val_data.watts_per_floor_area,
                Watts_per_Person=val_data.watts_per_person,
                Return_Air_Fraction=val_data.return_air_fraction,
                Fraction_Radiant=val_data.fraction_radiant,
                Fraction_Visible=val_data.fraction_visible,
                Fraction_Replaceable=val_data.fraction_replaceable,
                EndUse_Subcategory=val_data.end_use_subcategory
            )

            self.state["success"] += 1
            self.logger.success(
                f"Successfully added Lights '{val_data.name}' to Zone '{val_data.zone_name}'."
            )

        except Exception as e:
            self.state["failed"] += 1
            self.logger.error(f"Failed to add Lights object '{val_data.name}': {e}")
