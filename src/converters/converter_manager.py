from pathlib import Path
from eppy.modeleditor import IDF
from io import StringIO
from typing import Dict
import yaml

from src.utils.logging import get_logger
from src.converters.zone_converter import ZoneConverter
from src.converters.building_converter import BuildingConverter
from src.converters.surface_converter import BuildingSurfaceConverter

class ConverterManager:

    def __init__(self, idd_file: Path, file_to_convert: Path):
        self.logger = get_logger(__name__)
        IDF.setiddname(str(idd_file))
        self.idf = self._create_blank_idf()
        self.yaml_data : Dict = self._load_yaml(file_to_convert)
        self.converters = {
            'building': BuildingConverter(self.idf),
            'zones': ZoneConverter(self.idf),
            'surfaces': BuildingSurfaceConverter(self.idf)
        }

    def convert_all(self) -> IDF:
        for name, converter in self.converters.items():
            self.logger.info(f"Converting {name}...")
            converter.convert(self.yaml_data)
        
        return self.idf
    
    def save_idf(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Saving IDF to {output_path}...")
        self.idf.saveas(str(output_path))

    def load_idf(self, idf_path: Path) -> None:
        self.logger.info(f"Loading IDF from {idf_path}...")
        self.idf = IDF(str(idf_path))
        for converter in self.converters.values():
            converter.idf = self.idf

    def _create_blank_idf(self) -> IDF:
        self.logger.info("Creating a blank IDF instance.")
        idf_text = ""
        fhandle = StringIO(idf_text)
        return IDF(fhandle)
    
    def _load_yaml(self, file_path: Path) -> dict:
        self.logger.info(f"Loading YAML file from {file_path}.")
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

