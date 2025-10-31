from pathlib import Path
from eppy.modeleditor import IDF
from io import StringIO
from typing import Dict, cast, List
from copy import deepcopy
import yaml

from src.utils.logging import get_logger
from src.converters import BuildingConverter, ZoneConverter, SurfaceConverter
from src.validator.data_model import BaseSchema, IDDField
from src.converters.material_converter import MaterialConverter
from src.converters.construction_converter import ConstructionConverter
from src.converters.setting_converter import SettingsConverter


class ConverterManager:
    def __init__(self, idd_file: Path, file_to_convert: Path):
        self.logger = get_logger(__name__)
        IDF.setiddname(str(idd_file))
        self._idf = self._create_blank_idf()
        self.idf_field: IDDField = self._process_idf_field()
        self.yaml_data: Dict = self._load_yaml(file_to_convert)
        BaseSchema.set_idf_field(self.idf_field)
        self.converters = {
            "settings": SettingsConverter(self._idf),
            "building": BuildingConverter(self._idf),
            "zones": ZoneConverter(self._idf),
            "surfaces": SurfaceConverter(self._idf),
            "materials": MaterialConverter(self._idf),
            "constructions": ConstructionConverter(self._idf),
        }

    @property
    def idf(self) -> IDF:
        return deepcopy(self._idf)

    def convert_all(self) -> None:
        for name, converter in self.converters.items():
            self.logger.info(f"Converting {name}...")
            converter.convert(self.yaml_data)

    def save_idf(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Saving IDF to {output_path}...")
        self._idf.saveas(str(output_path))

    def load_idf(self, idf_path: Path) -> None:
        self.logger.info(f"Loading IDF from {idf_path}...")
        self._idf = IDF(str(idf_path))
        for converter in self.converters.values():
            converter.idf = self._idf

    def _create_blank_idf(self) -> IDF:
        self.logger.info("Creating a blank IDF instance.")
        idf_text = ""
        fhandle = StringIO(idf_text)
        return IDF(fhandle)

    def _load_yaml(self, file_path: Path) -> dict:
        self.logger.info(f"Loading YAML file from {file_path}.")
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _process_idf_field(self) -> IDDField:
        _idd_info = cast(List[Dict], self._idf.idd_info)
        idd_field = IDDField(_idd_info)
        return idd_field
