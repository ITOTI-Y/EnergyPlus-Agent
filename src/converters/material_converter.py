from eppy.modeleditor import IDF
from typing import Dict, Any, List
from src.converters.base_converter import BaseConverter
from src.utils.logging import get_logger
from src.validator.data_model import MaterialSchema

class MaterialConverter(BaseConverter):
    """
    Converts a list of material definitions from YAML data into either
    'MATERIAL' or 'MATERIAL:NOMASS' IDF objects based on the provided fields.
    """
    def __init__(self, idf: IDF):
        super().__init__(idf)
        self.logger = get_logger(__name__)

    def convert(self, data: Dict[str, Any]) -> None:
        """
        Processes a list of material definitions from the YAML data.
        """
        self.logger.info("Converting Material data...")
        material_list = data.get('Material', [])
        
        for material_data in material_list:
            try:
                validated_material = self.validate(material_data)
                self._add_to_idf(validated_material)
            except Exception as e:
                self.state['failed'] += 1
                self.logger.error(
                    f"Failed to convert Material '{material_data.get('Name', 'N/A')}': {e}", 
                    exc_info=True
                )
                continue

    def _add_to_idf(self, data: MaterialSchema) -> None:
        """
        Adds a single material to the IDF, intelligently choosing between
        'MATERIAL' and 'MATERIAL:NOMASS' based on the validated data.
        """
        idf_key = ""
        if data.Thermal_Resistance is not None:
            idf_key = "MATERIAL:NOMASS"
        else:
            idf_key = "MATERIAL"

        if self.idf.getobject(idf_key, data.Name):
            self.logger.warning(
                f"{idf_key} with name '{data.Name}' already exists. Skipping addition."
            )
            self.state['skipped'] += 1
            return
            
        try:
            self.logger.info(f"Adding {idf_key} '{data.Name}' to IDF.")
            
            material_dict = data.model_dump(exclude_unset=True)
            
            self.idf.newidfobject(idf_key, **material_dict)
            
            self.state['success'] += 1
        except Exception as e:
            self.state['failed'] += 1
            self.logger.error(f"Error adding {idf_key} '{data.Name}' to IDF: {e}", exc_info=True)

    def validate(self, data: Dict) -> MaterialSchema:
        """
        Validates a single material data dictionary using MaterialSchema.
        """
        return MaterialSchema.model_validate(data)
