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
        except Exception as e:
            self.state["failed"] += 1
            self.logger.error(f"Failed to process the entire HVAC block: {e}")
            return

        for thermostat in validated_hvac_schema.thermostats:
            self._add_to_idf(thermostat)

        for ideal_loads in validated_hvac_schema.ideal_loads_systems:
            self._add_to_idf(ideal_loads)

    def validate(self, data: dict[str, Any]) -> HVACSchema:
        """Validates the entire HVAC data block against the HVACSchema."""
        return HVACSchema.model_validate(data)

    def _add_to_idf(
        self,
        data: HVACTemplateThermostatSchema | HVACTemplateZoneIdealLoadsAirSystemSchema,
    ) -> None:
        try:
            if isinstance(data, HVACTemplateThermostatSchema):
                if not self.idf.getobject("HVACTemplate:Thermostat", data.name):
                    self.idf.newidfobject(
                        "HVACTemplate:Thermostat",
                        Name=data.name,
                        Heating_Setpoint_Schedule_Name=data.heating_setpoint_schedule_name,
                        Cooling_Setpoint_Schedule_Name=data.cooling_setpoint_schedule_name,
                    )
                    self.state["success"] += 1
                    self.logger.success(
                        f"Successfully added HVACTemplate:Thermostat '{data.name}'."
                    )
                else:
                    self.logger.warning(
                        f"HVACTemplate:Thermostat '{data.name}' already exists. Skipping."
                    )
                    self.state["skipped"] += 1
            elif isinstance(data, HVACTemplateZoneIdealLoadsAirSystemSchema):
                if not self.idf.getobject(
                    "HVACTemplate:Zone:IdealLoadsAirSystem", data.zone_name
                ):
                    self.idf.newidfobject(
                        "HVACTemplate:Zone:IdealLoadsAirSystem",
                        Zone_Name=data.zone_name,
                        Template_Thermostat_Name=data.template_thermostat_name,
                        System_Availability_Schedule_Name=data.system_availability_schedule_name
                        or "",
                    )
                    self.state["success"] += 1
                    self.logger.success(
                        f"Successfully added HVACTemplate:Zone:IdealLoadsAirSystem for zone '{data.zone_name}'."
                    )
                else:
                    self.logger.warning(
                        f"HVACTemplate:Zone:IdealLoadsAirSystem for zone '{data.zone_name}' already exists. Skipping."
                    )
                    self.state["skipped"] += 1
        except Exception as e:
            self.state["failed"] += 1
            self.logger.error(f"Failed to add HVAC object: {e}")
