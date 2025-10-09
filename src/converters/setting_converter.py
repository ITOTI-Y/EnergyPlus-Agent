from eppy.modeleditor import IDF
from typing import Dict, Any, List, Tuple, Optional
from src.converters.base_converter import BaseConverter
from src.utils.logging import get_logger
from pydantic import Field, create_model, field_validator
from src.validator.data_model import BaseSchema

class SettingsConverter(BaseConverter):

    def __init__(self, idf: IDF):
        super().__init__(idf)
        self.logger = get_logger(__name__)
        self.setting_keys = [
            "SimulationControl", "Timestep", "RunPeriod", "GlobalGeometryRules",
            "Site:Location", "Output:VariableDictionary", "Output:Diagnostics",
            "Output:Table:SummaryReports", "OutputControl:Table:Style", "Output:Variable"
        ]

    def convert(self, data: Dict[str, Any]) -> None:
        self.logger.info("Settings Converter Starting...")
        
        version_tuple: Tuple[int, ...] = self.idf.idd_version
        global_settings_data = {key: data.get(key) for key in self.setting_keys if key in data}
        
        try:

            data_to_validate = {
                "version_data": {"version": version_tuple},
                "global_settings_data": global_settings_data
            }
            
            validated_data = self.validate(data_to_validate)
            
            self._add_to_idf(validated_data)
            
            self.state['success'] += 1
        except Exception as e:
            self.state['failed'] += 1
            self.logger.error(f"Error during settings conversion process: {e}", exc_info=True)

    def _create_dynamic_schema(self, schema_name: str, data_dict: Dict[str, Any]) -> type[BaseSchema]:
        fields = {}
        for key, value in data_dict.items():
            field_name = key.replace(' ', '_').replace(':', '_')
            fields[field_name] = (Any, Field(default=..., alias=key))
        
        DynamicSchema = create_model(
            schema_name.replace(':', '_'),
            __base__=BaseSchema,
            **fields
        )
        return DynamicSchema

    def validate(self, data: Dict) -> Dict:
        self.logger.info("Validating global settings...")
        class VersionSchema(BaseSchema):
            version: str
            @field_validator('version', mode='before')
            def format_version(cls, v): return ".".join(map(str, v))
        
        val_version_data = VersionSchema.model_validate(data.get("version_data", {}))

        validated_settings = {}
        raw_global_settings = data.get("global_settings_data", {})
     
        for idf_key, setting_data in raw_global_settings.items():
            if setting_data is None: continue
            
            try:
                if isinstance(setting_data, list):
                    validated_list = []
                    for item in setting_data:
                        DynamicSchema = self._create_dynamic_schema(idf_key, item)
                        validated_list.append(DynamicSchema.model_validate(item))
                    validated_settings[idf_key] = validated_list
                else:
                    DynamicSchema = self._create_dynamic_schema(idf_key, setting_data)
                    validated_settings[idf_key] = DynamicSchema.model_validate(setting_data)
            except Exception as e:
                self.logger.error(f"Validation failed for '{idf_key}': {e}", exc_info=True)
                raise

        return {
            "version_info": val_version_data,
            "validated_settings": validated_settings
        }

    def _add_to_idf(self, val_data: Dict) -> None:
        version_info = val_data.get("version_info")
        settings_to_add = val_data.get("validated_settings", {})

        if version_info and not self.idf.idfobjects.get("Version"):
            self.logger.info(f"Adding Version object '{version_info.version}' to IDF.")
            self.idf.newidfobject("Version", Version_Identifier=version_info.version)
        
        for idf_key, validated_model_or_list in settings_to_add.items():
            items_to_process = validated_model_or_list if isinstance(validated_model_or_list, list) else [validated_model_or_list]
            
            for validated_model in items_to_process:
                data_dict = validated_model.model_dump(exclude_none=True)
        
                if idf_key != "Output:Variable" and len(self.idf.idfobjects.get(idf_key, [])) > 0:
                     self.logger.warning(f"Object of type '{idf_key}' already exists. Skipping addition.")
                     break 
                
                self.logger.info(f"Adding object of type '{idf_key}' to IDF.")
                self.idf.newidfobject(idf_key, **data_dict)