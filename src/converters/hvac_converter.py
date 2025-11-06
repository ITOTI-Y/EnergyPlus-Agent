from typing import Any

from eppy.modeleditor import IDF

from src.converters.base_converter import BaseConverter
from src.utils.logging import get_logger
from src.validator.data_model import HVACSchema


class HVACConverter(BaseConverter):
    """
    Converts HVAC component definitions from YAML data into IDF objects.
    This is a simplified version to establish the workflow. It creates
    individual components but does not handle their interconnections.
    """

    def __init__(self, idf: IDF):
        super().__init__(idf)
        self.logger = get_logger(__name__)

    def convert(self, data: dict[str, Any]) -> None:
        self.logger.info("HVAC Converter Starting...")
        hvac_data = data.get("HVAC", {})
        if not hvac_data:
            self.logger.info("No HVAC data found in YAML.")
            return

        try:
            validated_hvac = self.validate(hvac_data)
            self._add_to_idf(validated_hvac)
        except Exception as e:
            self.logger.error(f"Failed to process HVAC block: {e}", exc_info=True)

    def validate(self, data: dict[str, Any]) -> HVACSchema:
        """Validates the entire HVAC data block."""
        return HVACSchema.model_validate(data)

    def _add_to_idf(self, hvac_schema: HVACSchema) -> None:
        """Iterates through validated HVAC components and adds them to the IDF."""
        hvac_definitions = hvac_schema.model_dump(by_alias=True, exclude_none=True)

        for idf_key, object_list in hvac_definitions.items():
            self.logger.info(f"Processing IDF object type: {idf_key}...")
            for obj_data in object_list:
                self._add_single_object(idf_key, obj_data)

    def _add_single_object(self, idf_key: str, data: dict[str, Any]) -> None:
        """
        Generic function to add a single IDF object from a dictionary.
        This function handles object creation and logging.
        """
        object_name = data.get("Name", "Unknown Object")
        try:
            params = {
                key.replace(" ", "_").replace("-", "_"): value
                for key, value in data.items()
            }

            if idf_key == "Schedule:Compact" and "Data" in params:
                schedule_data = params.pop("Data")
                idf_object = self.idf.newidfobject(idf_key, **params)
                for i, datum in enumerate(schedule_data, 1):
                    setattr(idf_object, f"Field_{i}", datum)
            else:
                self.idf.newidfobject(idf_key, **params)

            self.state["success"] += 1
            self.logger.debug(f"Successfully added '{object_name}' as {idf_key}.")
        except Exception as e:
            self.state["failed"] += 1
            self.logger.error(f"Failed to add '{object_name}' as {idf_key}: {e}")
