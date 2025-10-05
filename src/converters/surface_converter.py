from eppy.modeleditor import IDF
from typing import Dict, Any

from src.converters.base_converter import BaseConverter
from src.validator.data_model import SurfaceSchema

class SurfaceConverter(BaseConverter):

    def __init__(self, idf: IDF):
        super().__init__(idf)

    def convert(self, data: Dict) -> None:
        self.logger.info("Converting BuildingSurface data...")
        for sd in data.get('BuildingSurface:Detailed', []):
            try:
                val_data = self.validate(sd)
                self._add_to_idf(val_data)
            except Exception as e:
                self.state['failed'] += 1
                self.logger.error(f"Error Validate BuildingSurface Data: {e}", exc_info=True)
                continue

    
    def _add_to_idf(self, data:Any) -> None:
        if self.idf.getobject("BuildingSurface:Detailed", name=data.name):
            self.logger.warning(f"BuildingSurface with name {data.name} already exists in IDF. Skipping addition.")
            self.state['skipped'] += 1
            return
        try:
            surface_obj = self.idf.newidfobject(
                "BuildingSurface:Detailed",
                Name=data.name,
                Surface_Type=data.surface_type,
                Construction_Name=data.construction_name,
                Zone_Name=data.zone_name,
                Space_Name=data.space_name or "",  
                Outside_Boundary_Condition=data.outside_boundary_condition,
                Outside_Boundary_Condition_Object=data.outside_boundary_condition_object or "",
                Sun_Exposure=data.sun_exposure,
                Wind_Exposure=data.wind_exposure,
                View_Factor_to_Ground=data.view_factor_to_ground
            )
            
            
            for i, vertex in enumerate(data.vertices, 1):
                setattr(surface_obj, f'Vertex_{i}_Xcoordinate', vertex['X'])
                setattr(surface_obj, f'Vertex_{i}_Ycoordinate', vertex['Y'])
                setattr(surface_obj, f'Vertex_{i}_Zcoordinate', vertex['Z'])
            
            self.state['success'] += 1
            self.logger.info(f"BuildingSurface with name {data.name} added to IDF.")
        except Exception as e:
            self.state['failed'] += 1
            self.logger.error(f"Error Adding BuildingSurface Data to IDF: {e}", exc_info=True)

    def validate(self, data: Dict) -> Any:
        val_data = SurfaceSchema.model_validate(data)
        return val_data