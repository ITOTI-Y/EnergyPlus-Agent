from eppy.modeleditor import IDF
from typing import Dict, List
from collections import defaultdict

from src.converters.base_converter import BaseConverter
from src.validator.data_model import FenestrationSurfaceSchema, GeometryWindowsSchema


class FenestrationConverter(BaseConverter):
    def __init__(self, idf: IDF):
        super().__init__(idf)

    def convert(self, data: Dict) -> None:
        self.logger.info("Converting FenestrationSurface data...")
        fenestration_data = data.get("FenestrationSurface:Detailed", [])
        
        # Group fenestration surfaces by building surface for geometry validation
        building_surface_to_fenestrations = defaultdict(list)
        for fenestration in fenestration_data:
            building_surface_to_fenestrations[fenestration["Building Surface Name"]].append(fenestration)
        
        # Pass the complete data to validate method so it can access building surface info
        val_data = self.validate(building_surface_to_fenestrations, data)
        for fenestration in val_data:
            try:
                self._add_to_idf(fenestration)
                self.logger.success(
                    f"Successfully converted FenestrationSurface: {fenestration.name}"
                )
                self.state["success"] += 1
            except Exception as e:
                self.state["failed"] += 1
                self.logger.error(
                    f"Error Converting FenestrationSurface Data: {e}", exc_info=True
                )

    def _add_to_idf(self, data: FenestrationSurfaceSchema) -> None:
        if self.idf.getobject("FenestrationSurface:Detailed", name=data.name):
            self.logger.warning(
                f"FenestrationSurface with name {data.name} already exists in IDF. Skipping addition."
            )
            self.state["skipped"] += 1
            return
            
        fenestration_obj = self.idf.newidfobject(
            "FenestrationSurface:Detailed",
            Name=data.name,
            Surface_Type=data.surface_type,
            Construction_Name=data.construction_name,
            Building_Surface_Name=data.building_surface_name,
            Outside_Boundary_Condition_Object=data.outside_boundary_condition_object or "",
            View_Factor_to_Ground=data.view_factor_to_ground or "",
            Frame_and_Divider_Name=data.frame_and_divider_name or "",
            Multiplier=data.multiplier,
            Number_of_Vertices=data.Number_of_Vertices,
        )

        # Set vertices (already validated by GeometryWindowsSchema)
        for i, vertex in enumerate(data.vertices, 1):
            setattr(fenestration_obj, f"Vertex_{i}_Xcoordinate", vertex[0])
            setattr(fenestration_obj, f"Vertex_{i}_Ycoordinate", vertex[1])
            setattr(fenestration_obj, f"Vertex_{i}_Zcoordinate", vertex[2])
        
        # Set the number of vertices based on actual vertex count
        fenestration_obj.Number_of_Vertices = data.Number_of_Vertices

    def validate(self, fenestration_data: Dict, complete_data: Dict) -> List[FenestrationSurfaceSchema]:
        val_data = []
        
        # Collect all fenestration surfaces for validation
        all_fenestrations = []
        for fenestrations in fenestration_data.values():
            all_fenestrations.extend(fenestrations)
        
        if not all_fenestrations:
            return val_data
            
        # Use GeometryWindowsSchema to validate fenestration geometry
        try:
            windows_data = {"windows": all_fenestrations}
            geometry = GeometryWindowsSchema.model_validate(windows_data)
            
            # Convert validated windows back to FenestrationSurfaceSchema
            for window in geometry.windows:
                fenestration_schema = FenestrationSurfaceSchema.model_validate({
                    "Name": window.name,
                    "Surface Type": window.surface_type,
                    "Construction Name": window.construction_name,
                    "Building Surface Name": window.building_surface_name,
                    "Outside Boundary Condition Object": window.outside_boundary_condition_object,
                    "View Factor to Ground": window.view_factor_to_ground,
                    "Frame and Divider Name": window.frame_and_divider_name,
                    "Multiplier": window.multiplier,
                    "Number of Vertices": window.Number_of_Vertices,
                    "Vertices": window.vertices
                })
                val_data.append(fenestration_schema)
                
        except Exception as e:
            self.logger.error(f"Geometry validation failed for fenestration surfaces: {e}")
            self.state["failed"] += len(all_fenestrations)
            
        return val_data

    def _get_building_surface_data(self, building_surface_name: str, complete_data: Dict) -> Dict:
        """Get building surface data by name from the complete dataset"""
        building_surfaces = complete_data.get("BuildingSurface:Detailed", [])
        
        for surface in building_surfaces:
            if surface.get("Name") == building_surface_name:
                return {
                    "Name": surface["Name"],
                    "Surface Type": surface["Surface Type"],
                    "Construction Name": surface["Construction Name"],
                    "Zone Name": surface["Zone Name"],
                    "Outside Boundary Condition": surface["Outside Boundary Condition"],
                    "Sun Exposure": surface["Sun Exposure"],
                    "Wind Exposure": surface["Wind Exposure"],
                    "View Factor to Ground": surface["View Factor to Ground"]
                }
        
        # Return None if building surface not found
        return None
