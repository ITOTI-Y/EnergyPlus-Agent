from idfpy import IDF
from idfpy.models.simulation import Building

from src.converters.base_converter import BaseConverter
from src.utils.logging import get_logger
from src.validator.data_model import BuildingSchema

logger = get_logger(__name__)


class BuildingConverter(BaseConverter):
    def __init__(self, idf: IDF):
        super().__init__(idf)

    def convert(self, data: dict) -> None:
        self.logger.info("Building Converter Starting...")

        building_data: dict = data.get("Building", {})

        try:
            validated_data = self.validate(building_data)
            self._add_to_idf(validated_data)
        except Exception as e:
            self.state["failed"] += 1
            self.logger.error("Error Convert Building Data: {}", e)

    def _add_to_idf(self, val_data: dict) -> None:
        building_data: BuildingSchema = val_data["building_data"]

        try:
            if not self.idf.has("Building", building_data.name):
                self.idf.add(Building(
                    name=building_data.name,
                    north_axis=building_data.north_axis,
                    terrain=building_data.terrain,
                    loads_convergence_tolerance_value=building_data.loads_convergence_tolerance_value,
                    temperature_convergence_tolerance_value=building_data.temperature_convergence_tolerance_value,
                    solar_distribution=building_data.solar_distribution,
                    maximum_number_of_warmup_days=building_data.maximum_number_of_warmup_days,
                    minimum_number_of_warmup_days=building_data.minimum_number_of_warmup_days,
                ))
                self.state["success"] += 1
                self.logger.success(
                    "Building object with name {} added to IDF.", building_data.name
                )
            else:
                self.logger.warning(
                    "Building object with name {} already exists in IDF. "
                    "Skipping addition.",
                    building_data.name,
                )
                self.state["skipped"] += 1
        except Exception as e:
            self.state["failed"] += 1
            self.logger.error("Error Adding Building to IDF: {}", e)

    def validate(self, data: dict) -> dict:
        val_building_data = BuildingSchema.model_validate(data)
        return {
            "building_data": val_building_data,
        }
