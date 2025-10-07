from eppy.modeleditor import IDF
from typing import Dict, Any, List, Tuple
from src.converters.base_converter import BaseConverter
from src.validator.data_model import(
    BaseSchema,
)
from src.utils.logging import get_logger
try:
    from src.validator.data_model import SimulationControlSchema
except ImportError:
    class SimulationControlSchema(BaseSchema):pass

try:
    from src.validator.data_model import TimestepSchema
except ImportError:
    class TimestepSchema(BaseSchema):pass

try:
    from src.validator.data_model import RunPeriodSchema
except ImportError:
    class RunPeriodSchema(BaseSchema):pass

try:
    from src.validator.data_model import GlobalGeometryRulesSchema
except ImportError:
    class GlobalGeometryRulesSchema(BaseSchema):pass

try:
    from src.validator.data_model import SiteLocationSchema
except ImportError:
    class SiteLocationSchema(BaseSchema):pass   

try:
    from src.validator.data_model import OutputVariableDictionarySchema
except ImportError:             
    class OutputVariableDictionarySchema(BaseSchema):pass

try:
    from src.validator.data_model import OutputDiagnosticsSchema
except ImportError:             
    class OutputDiagnosticsSchema(BaseSchema):pass

try:
    from src.validator.data_model import OutputTableSummaryReportsSchema
except ImportError:            
    class OutputTableSummaryReportsSchema(BaseSchema):pass  

try:
    from src.validator.data_model import OutputControlTableStyleSchema                  
except ImportError:
    class OutputControlTableStyleSchema(BaseSchema):pass    

try:
    from src.validator.data_model import OutputVariableSchema
except ImportError:
    class OutputVariableSchema(BaseSchema):pass

class SettingsConverter(BaseConverter):
    def __init__(self,idf:IDF):
      super().__init__(idf)
      self.logger=get_logger(__name__)
    
      self.setting_map={
        "SimulationControl":SimulationControlSchema,
        "Timestep":TimestepSchema,
        "RunPeriod":RunPeriodSchema,
        "GlobalGeometryRules":GlobalGeometryRulesSchema,
        "SiteLocation":SiteLocationSchema,
        "OutputVariableDictionary":OutputVariableDictionarySchema,
        "OutputDiagnostics":OutputDiagnosticsSchema,
        "OutputTableSummaryReports":OutputTableSummaryReportsSchema,
        "OutputControl:Table:Style":OutputControlTableStyleSchema,
        "OutputVariable":OutputVariableSchema,
    }

    def convert(self, data: Dict[str, Any]) -> None:
      self.logger.info("Settings Converter Starting...")
      self._add_version_object()
      for idf_key, schema in self.setting_map.items():
          if idf_key in data:
             setting_data = data[idf_key]
             try:
                 if isinstance(setting_data, list):
                     for item in setting_data:
                         self._add_to_idf(idf_key, item)
                 else:
                     self._add_to_idf(idf_key, setting_data)
                 self.state['success'] += 1
             except Exception as e:
                 self.state['failed'] += 1
                 self.logger.error(f"Error processing setting '{idf_key}': {e}", exc_info=True)
          else:
                self.logger.debug(f"Setting '{idf_key}' not found in YAML. Skipping.")
                self.state['skipped'] += 1

    def _add_version_object(self) -> None:
        try:
            idd_version: Tuple[int, ...] = self.idf.idd_version
            version_str = ".".join(map(str, idd_version))
            
            self.logger.info(f"Ensuring Version object '{version_str}' exists in IDF.")
            if not self.idf.idfobjects.get("Version"):
                self.idf.newidfobject("Version", Version_Identifier=version_str)
        except Exception as e:
            self.logger.error(f"Error adding Version object: {e}", exc_info=True)

    def _add_to_idf(self, idf_key: str, data_to_add: Any) -> None:
        filtered_data = {key: value for key, value in data_to_add.items() if value is not None}
        
        cleaned_data = {key.replace(' ', '_'): value for key, value in filtered_data.items()}
       
        if len(self.idf.idfobjects.get(idf_key, [])) > 0 and idf_key != "Output:Variable":
             self.logger.warning(f"Object of type '{idf_key}' already exists. Skipping addition.")
             return

        self.logger.info(f"Adding object of type '{idf_key}' to IDF.")
        self.idf.newidfobject(idf_key, **cleaned_data)

    def validate(self, data: Dict) -> Any:
        pass

          

