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
        self.idf.add(
            self._to_idf_model(
                Lights,
                val_data,
                zone_or_zonelist_or_space_or_spacelist_name=val_data.zone_or_zone_list_or_space_or_space_list_name,
            )
        )
        self.state["success"] += 1
        self.logger.success("Light with name {} added to IDF.", val_data.name)

    def validate(self, data: dict) -> LightSchema:
        return LightSchema.model_validate(data)
