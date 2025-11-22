from typing import Any

from eppy.modeleditor import IDF

from src.converters.base_converter import BaseConverter
from src.utils.logging import get_logger
from src.validator.data_model import (
    HVACSchema,
    HVACTemplateThermostatSchema,
    HVACTemplateZoneIdealLoadsAirSystemSchema,
)


class HVACConverter(BaseConverter):
    """
    Converts HVAC component definitions from YAML data into IDF objects.
    This version follows the explicit handling pattern of MaterialConverter.
    """
    def __init__(self, idf: IDF):
        super().__init__(idf)
        self.logger = get_logger(__name__)

    def convert(self, data: dict[str, Any]) -> None:
        """
        Processes the HVAC block from YAML, validating and adding each component.
        """
        self.logger.info("HVAC Converter Starting...")
        hvac_data = data.get("HVAC", {})
        if not hvac_data:
            self.logger.info("No HVAC data found in YAML.")
            return

        try:
            validated_hvac_schema = self.validate(hvac_data)
            self._add_to_idf(validated_hvac_schema)
        except Exception as e:
            self.state["failed"] += 1
            self.logger.error(f"Failed to process the entire HVAC block: {e}", exc_info=True)

    def validate(self, data: dict[str, Any]) -> HVACSchema:
        """Validates the entire HVAC data block against the HVACSchema."""
        return HVACSchema.model_validate(data)

    def _add_to_idf(self, hvac_schema: HVACSchema) -> None:
        """
        Iterates through the lists of different HVAC components within the schema
        and adds them to the IDF using specific handlers.
        """
        if hvac_schema.thermostats:
            for thermostat_model in hvac_schema.thermostats:
                try:
                    self._add_thermostat(thermostat_model)
                except Exception as e:
                    self.state["failed"] += 1
                    self.logger.error(
                        f"Failed to add HVACTemplate:Thermostat '{thermostat_model.name}': {e}",
                        exc_info=True
                    )

        if hvac_schema.ideal_loads_systems:
            for ideal_loads_model in hvac_schema.ideal_loads_systems:
                try:
                    self._add_ideal_loads(ideal_loads_model)
                except Exception as e:
                    self.state["failed"] += 1
                    self.logger.error(
                        f"Failed to add IdealLoadsAirSystem for zone '{ideal_loads_model.zone_name}': {e}",
                        exc_info=True
                    )


    def _add_thermostat(self, model: HVACTemplateThermostatSchema) -> None:
        """
        Adds a single HVACTemplate:Thermostat object to the IDF.
        Uses explicit, hard-coded keyword arguments for reliability.
        """
        idf_key = "HVACTemplate:Thermostat"
        if self.idf.getobject(idf_key.upper(), model.name):
            self.logger.warning(f"{idf_key} '{model.name}' already exists. Skipping.")
            self.state["skipped"] += 1
            return
        self.idf.newidfobject(
            idf_key.upper(),
            Name=model.name,
            Heating_Setpoint_Schedule_Name=model.heating_setpoint_schedule_name,
            Cooling_Setpoint_Schedule_Name=model.cooling_setpoint_schedule_name,
        )
        self.state["success"] += 1
        self.logger.success(f"Successfully added {idf_key} '{model.name}'.")

    def _add_ideal_loads(self, model: HVACTemplateZoneIdealLoadsAirSystemSchema) -> None:
        """
        Adds a single HVACTemplate:Zone:IdealLoadsAirSystem object to the IDF,
        using the correct, discovered field names.
        """
        idf_key = "HVACTemplate:Zone:IdealLoadsAirSystem"
        try:
            new_obj = self.idf.newidfobject(idf_key.upper())

            new_obj.Zone_Name = model.zone_name
            new_obj.Template_Thermostat_Name = model.template_thermostat_name
            if model.system_availability_schedule_name:
                new_obj.System_Availability_Schedule_Name = model.system_availability_schedule_name

            self.state["success"] += 1
            self.logger.success(f"Successfully added and configured {idf_key} for zone '{model.zone_name}'.")

        except Exception as e:
            self.state["failed"] += 1
            self.logger.error(f"Failed to set attributes for {idf_key} for zone '{model.zone_name}': {e}", exc_info=True)
