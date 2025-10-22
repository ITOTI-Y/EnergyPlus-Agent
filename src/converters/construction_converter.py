from eppy.modeleditor import IDF
from typing import Dict, Any, List
from src.converters.base_converter import BaseConverter
from src.utils.logging import get_logger
from src.validator.data_model import ConstructionSchema

class ConstructionConverter(BaseConverter):
    def __init__(self, idf: IDF):
        super().__init__(idf)
        self.logger = get_logger(__name__)

    def convert(self, data: Dict[str, Any]) -> None:
        self.logger.info("Converting Construction data...")
        construction_list = data.get('Construction', [])
        
        for construction_data in construction_list:
            try:
                validated_construction = self.validate(construction_data)
                self._add_to_idf(validated_construction)
            except Exception as e:
                self.state['failed'] += 1
                self.logger.error(
                    f"Failed to convert Construction '{construction_data.get('Name', 'N/A')}': {e}", 
                    exc_info=True
                )
                continue

    def _add_to_idf(self, data: ConstructionSchema) -> None:
        if self.idf.getobject("CONSTRUCTION", data.Name):
            self.logger.warning(
                f"Construction with name '{data.Name}' already exists. Skipping addition."
            )
            self.state['skipped'] += 1
            return
            
        try:
            self.logger.info(f"Adding Construction '{data.Name}' to IDF.")
           
            for layer_name in data.Layers:
                if not (self.idf.getobject("MATERIAL", layer_name) or 
                        self.idf.getobject("MATERIAL:NOMASS", layer_name)):
                    raise ValueError(...)
            
            construction_obj = self.idf.newidfobject(
                "CONSTRUCTION",
                Name=data.Name
            )
            
            for i, layer_name in enumerate(data.Layers):
                if i == 0:
                    field_name = "Outside_Layer"
                else:
                    field_name = f"Layer_{i + 1}"
                
                setattr(construction_obj, field_name, layer_name)
                self.logger.debug(f"  - Set {field_name} to '{layer_name}' for '{data.Name}'.")
            self.state['success'] += 1
            self.logger.info(f"Construction '{data.Name}' with {len(data.Layers)} layers added successfully.")

        except ValueError as e:
            self.state['failed'] += 1
            self.logger.error(f"Failed to add Construction '{data.Name}': {e}")    
        
        except AttributeError as e:
            self.state['failed'] += 1
            self.logger.error(
                f"Error adding Construction '{data.Name}'. A specified field name was not found. Details: {e}",
                exc_info=True
            )
        except Exception as e:
            self.state['failed'] += 1
            self.logger.error(f"An unexpected error occurred while adding Construction '{data.Name}': {e}", exc_info=True)

    def validate(self, data: Dict) -> ConstructionSchema:
        return ConstructionSchema.model_validate(data)
