from copy import deepcopy
from pathlib import Path

import yaml
from idfpy import IDF

from src.converters import (
    BuildingConverter,
    ConstructionConverter,
    FenestrationConverter,
    HVACConverter,
    LightConverter,
    MaterialConverter,
    PeopleConverter,
    ScheduleConverter,
    SettingsConverter,
    SurfaceConverter,
    ZoneConverter,
)
from src.utils.logging import get_logger
from src.validator.data_model import BaseSchema


class ConverterManager:
    def __init__(self, source: Path | dict):
        self.logger = get_logger(__name__)
        self._idf = BaseSchema.get_idf()
        if isinstance(source, dict):
            self.yaml_data: dict = source
        else:
            self.yaml_data = self._load_yaml(source)
        self.converters = {
            "settings": SettingsConverter(self._idf),
            "building": BuildingConverter(self._idf),
            "schedules": ScheduleConverter(self._idf),
            "zones": ZoneConverter(self._idf),
            "materials": MaterialConverter(self._idf),
            "constructions": ConstructionConverter(self._idf),
            "surfaces": SurfaceConverter(self._idf),
            "fenestrations": FenestrationConverter(self._idf),
            "hvac": HVACConverter(self._idf),
            "lights": LightConverter(self._idf),
            "people": PeopleConverter(self._idf),
        }

    @property
    def idf(self) -> IDF:
        return deepcopy(self._idf)

    def convert_all(self) -> None:
        for name, converter in self.converters.items():
            self.logger.info("Converting {}...", name)
            converter.convert(self.yaml_data)

    def save_idf(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info("Saving IDF to {}...", output_path)
        self._idf.save(output_path)

    def load_idf(self, idf_path: Path) -> None:
        self.logger.info("Loading IDF from {}...", idf_path)
        self._idf = IDF.load(idf_path)
        for converter in self.converters.values():
            converter.idf = self._idf

    def _create_blank_idf(self) -> IDF:
        self.logger.info("Creating a blank IDF instance.")
        return IDF()

    def _load_yaml(self, file_path: Path) -> dict:
        self.logger.info("Loading YAML file from {}.", file_path)
        with open(file_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
