from typing import Any

from eppy.modeleditor import IDF

from src.converters.base_converter import BaseConverter
from src.utils.logging import get_logger
from src.validator.data_model import LightsSchema


class LightConverter(BaseConverter):
    def __init__(self, idf: IDF):
        super().__init__(idf)
        self.logger = get_logger(__name__)

    def convert(self, data: dict[str, Any]) -> None:
        self.logger.info("Converting Lights data...")
        lights_list = data.get("Lights") or data.get("lights", [])
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

    def validate(self, data: dict) -> LightsSchema:
        return LightsSchema.model_validate(data)

    def _add_to_idf(self, val_data: LightsSchema) -> None:
        if self.idf.getobject("LIGHTS", val_data.name):
            self.logger.warning(
                f"Lights object with name '{val_data.name}' already exists. Skipping addition."
            )
            self.state["skipped"] += 1
            return

        try:
            self.logger.debug(f"Adding Lights '{val_data.name}' to IDF.")

            if not self.idf.getobject("ZONE", val_data.zone_name):
                raise ValueError(
                    f"Zone '{val_data.zone_name}' referenced in Lights '{val_data.name}' "
                    f"does not exist in IDF. Please add the Zone first."
                )

            schedule_exists = False
            for key in self.idf.idfobjects:
                if key.upper().startswith("SCHEDULE") and self.idf.getobject(key, val_data.schedule_name):
                    schedule_exists = True
                    break
            if not schedule_exists:
                raise ValueError(
                    f"Schedule '{val_data.schedule_name}' referenced in Lights '{val_data.name}' "
                    f"does not exist in IDF. Please add the Schedule first."
                )

            light_obj = self.idf.newidfobject("LIGHTS")
            light_obj.Name = val_data.name

            zone_field = self._get_idd_field_name(light_obj, ["Zone", "Name"])
            if zone_field:
                setattr(light_obj, zone_field, val_data.zone_name)
            else:
                raise AttributeError(
                    f"IDD Error: Could not find 'Zone Name' field in IDD definition for Lights. "
                    f"Available fields: {light_obj.fieldnames}"
                )

            light_obj.Schedule_Name = val_data.schedule_name
            light_obj.Design_Level_Calculation_Method = val_data.design_level_calc_method

            if val_data.design_level_calc_method == "LightingLevel":
                light_obj.Lighting_Level = val_data.lighting_level

            elif val_data.design_level_calc_method == "Watts/Area":
                wa_field = self._get_idd_field_name(light_obj, ["Watts", "Area"])
                if wa_field:
                    setattr(light_obj, wa_field, val_data.watts_per_zone_floor_area)
                else:
                    raise AttributeError(
                        f"IDD Error: Could not find a field matching 'Watts' and 'Area' "
                        f"in LIGHTS object. Available fields: {light_obj.fieldnames}"
                    )

            elif val_data.design_level_calc_method == "Watts/Person":
                wp_field = self._get_idd_field_name(light_obj, ["Watts", "Person"])
                if wp_field:
                    setattr(light_obj, wp_field, val_data.watts_per_person)
                else:
                    raise AttributeError(
                        f"IDD Error: Could not find a field matching 'Watts' and 'Person' "
                        f"in LIGHTS object. Available fields: {light_obj.fieldnames}"
                    )

            light_obj.Return_Air_Fraction = val_data.return_air_fraction
            light_obj.Fraction_Radiant = val_data.fraction_radiant
            light_obj.Fraction_Visible = val_data.fraction_visible

            rep_field = self._get_idd_field_name(light_obj, ["Fraction", "Replaceable"])
            if rep_field:
                setattr(light_obj, rep_field, val_data.fraction_replaceable)
            end_use_field = self._get_idd_field_name(light_obj, ["End", "Use", "Subcategory"])
            if end_use_field:
                setattr(light_obj, end_use_field, val_data.end_use_subcategory)

            self.state["success"] += 1
            self.logger.success(
                f"Lights '{val_data.name}' added successfully to Zone '{val_data.zone_name}'."
            )

        except ValueError as e:
            self.state["failed"] += 1
            self.logger.error(f"Validation Error adding Lights '{val_data.name}': {e}")
        except AttributeError as e:
            self.state["failed"] += 1
            self.logger.error(f"IDD/Schema Error adding Lights '{val_data.name}': {e}")
        except Exception:
            self.state["failed"] += 1
            self.logger.exception(
                f"An unexpected error occurred while adding Lights '{val_data.name}'"
            )

    def _get_idd_field_name(self, idf_obj, keywords: list[str]) -> str | None:
        """
        Helper Function: Resolves inconsistencies in IDD version field names
For example: Searches for fields containing both “Zone” and “Name”
        """
        for field in idf_obj.fieldnames:
            if all(k.lower() in field.lower() for k in keywords):
                return field
        return None
