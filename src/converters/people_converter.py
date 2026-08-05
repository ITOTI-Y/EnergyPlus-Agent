from idfpy import IDF
from idfpy.models.internal_gains import People

from src.converters.base_converter import BaseConverter
from src.validator.data_model import PeopleSchema


class PeopleConverter(BaseConverter):
    def __init__(self, idf: IDF):
        super().__init__(idf)

    def convert(self, data: dict):
        self.logger.info("Converting people data...")
        for people in data.get("People", []):
            try:
                validated_people = self.validate(people)
                self._add_to_idf(validated_people)
            except Exception as e:
                self.state["failed"] += 1
                self.logger.error("Error converting people data: {}", e)
                continue

    def _add_to_idf(self, val_data: PeopleSchema):
        if self.idf.has("People", val_data.name):
            self.logger.warning(
                "People with name {} already exists in IDF. Skipping addition.",
                val_data.name,
            )
            self.state["skipped"] += 1
            return
        self.idf.add(self._to_idf_model(People, val_data))
        self.state["success"] += 1
        self.logger.success("People with name {} added to IDF.", val_data.name)

    def validate(self, data: dict) -> PeopleSchema:
        return PeopleSchema.model_validate(data)
