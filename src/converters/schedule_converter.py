from typing import Any

from eppy.modeleditor import IDF

from src.converters.base_converter import BaseConverter
from src.utils.logging import get_logger
from src.validator.data_model import ScheduleCollectionSchema


class ScheduleConverter(BaseConverter):
    """
    Converts Schedule component definitions from YAML data into IDF objects.
    Handles ScheduleTypeLimits and Schedule:Compact.
    """
    def __init__(self, idf: IDF):
        super().__init__(idf)
        self.logger = get_logger(__name__)

    def convert(self, data: dict[str, Any]) -> None:
        self.logger.info("Schedule Converter Starting...")
        schedule_data = data.get("Schedule", {})
        if not schedule_data:
            self.logger.info("No Schedule data found in YAML.")
            return

        try:
            validated_schedules = self.validate(schedule_data)
            self._add_to_idf(validated_schedules)
        except Exception as e:
            self.logger.error(f"Failed to process Schedule block: {e}", exc_info=True)

    def validate(self, data: dict[str, Any]) -> ScheduleCollectionSchema:
        """Validates the entire Schedule data block using the container schema."""
        return ScheduleCollectionSchema.model_validate(data)

    # --- MODIFIED: 更新类型注解 ---
    def _add_to_idf(self, schedule_schema: ScheduleCollectionSchema) -> None:
        """Iterates through validated Schedule components and adds them to the IDF."""

        schedule_definitions = schedule_schema.model_dump(by_alias=True, exclude_none=True)

        for idf_key, object_list in schedule_definitions.items():
            if not object_list:
                continue
            self.logger.info(f"Processing IDF object type: {idf_key}...")
            for obj_data in object_list:
                self._add_single_object(idf_key, obj_data)

    def _add_single_object(self, idf_key: str, data: dict[str, Any]) -> None:
        """Generic function to add a single IDF object from a dictionary."""
        object_name = data.get("Name", "Unknown Schedule Object")
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
