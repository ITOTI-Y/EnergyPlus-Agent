from typing import Any

from idfpy import IDF
from idfpy.models.thermal_zones import Zone

from src.converters.base_converter import BaseConverter
from src.validator.data_model import ZoneSchema


class ZoneConverter(BaseConverter):
    def __init__(self, idf: IDF):
        super().__init__(idf)

    def convert(self, data: dict) -> None:
        self.logger.info("Converting zone data...")
        for zd in data.get("Zone", []):
            try:
                val_data = self.validate(zd)
                self._add_to_idf(val_data)
            except Exception:
                self.state["failed"] += 1
                self.logger.exception("Error processing Zone")
                continue

    def _add_to_idf(self, val_data: Any) -> None:
        if self.idf.has("Zone", val_data.name):
            self.logger.warning(
                "Zone with name {} already exists in IDF. Skipping addition.",
                val_data.name,
            )
            self.state["skipped"] += 1
            return
        try:
            self.idf.add(Zone(
                name=val_data.name,
                direction_of_relative_north=val_data.direction_of_relative_north,
                x_origin=val_data.x_origin,
                y_origin=val_data.y_origin,
                z_origin=val_data.z_origin,
                type=val_data.type,
                multiplier=val_data.multiplier,
                ceiling_height=val_data.ceiling_height,
                volume=val_data.volume,
                floor_area=val_data.floor_area,
                zone_inside_convection_algorithm=val_data.zone_inside_convection_algorithm,
                zone_outside_convection_algorithm=val_data.zone_outside_convection_algorithm,
                part_of_total_floor_area=val_data.part_of_total_floor_area,
            ))
            self.state["success"] += 1
            self.logger.success("Zone with name {} added to IDF.", val_data.name)
        except Exception:
            self.state["failed"] += 1
            self.logger.exception("Error Adding Zone Data to IDF")

    def validate(self, data: dict) -> Any:
        val_data = ZoneSchema.model_validate(data)
        return val_data
