from typing import Any

from idfpy import IDF
from idfpy.models.schedules import ScheduleCompact, ScheduleCompactDataItem, ScheduleTypeLimits

from src.converters.base_converter import BaseConverter
from src.utils.logging import get_logger
from src.validator.data_model import (
    ScheduleCollectionSchema,
    ScheduleCompactSchema,
    ScheduleTypeLimitsSchema,
)


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
            validated_data = self.validate(schedule_data)
        except Exception as e:
            self.state["failed"] += 1
            self.logger.error("Failed to validate Schedule data: {}", e)
            return

        for schedule_type_limits in validated_data.schedule_type_limits:
            self._add_to_idf(schedule_type_limits)

        for schedule_compact in validated_data.schedules:
            self._add_to_idf(schedule_compact)

    def _add_to_idf(self, val_data: Any) -> None:
        try:
            if isinstance(val_data, ScheduleTypeLimitsSchema):
                if not self.idf.has("ScheduleTypeLimits", val_data.name):
                    lower = val_data.lower_limit_value if val_data.lower_limit_value != "" else None
                    upper = val_data.upper_limit_value if val_data.upper_limit_value != "" else None
                    self.idf.add(ScheduleTypeLimits(
                        name=val_data.name,
                        lower_limit_value=lower,
                        upper_limit_value=upper,
                        numeric_type=val_data.numeric_type,
                        unit_type=val_data.unit_type,
                    ))
                    self.state["success"] += 1
                    self.logger.success(
                        "ScheduleTypeLimits with name {} added to IDF.",
                        val_data.name,
                    )
                else:
                    self.logger.warning(
                        "ScheduleTypeLimits with name {} already exists in IDF. "
                        "Skipping addition.",
                        val_data.name,
                    )
                    self.state["skipped"] += 1
            elif isinstance(val_data, ScheduleCompactSchema):
                if not self.idf.has("ScheduleCompact", val_data.name):
                    self.idf.add(ScheduleCompact(
                        name=val_data.name,
                        schedule_type_limits_name=val_data.schedule_type_limits_name,
                        data=[ScheduleCompactDataItem(field=v) for v in val_data.data],
                    ))
                    self.state["success"] += 1
                    self.logger.success(
                        "Schedule:Compact with name {} added to IDF.", val_data.name
                    )
                else:
                    self.logger.warning(
                        "Schedule:Compact with name {} already exists in IDF. "
                        "Skipping addition.",
                        val_data.name,
                    )
                    self.state["skipped"] += 1
            else:
                self.state["failed"] += 1
                raise ValueError(f"Unknown Schedule object type: {type(val_data)}")
        except Exception as e:
            self.state["failed"] += 1
            self.logger.error("Failed to add Schedule object: {}", e)

    def validate(self, data: dict[str, Any]) -> Any:
        return ScheduleCollectionSchema.model_validate(data)
