from eppy.modeleditor import IDF

from src.converters.base_converter import BaseConverter
from src.validator.data_model import (
    FenestrationSurfaceSchema,
    GeometrySchema,
)


class FenestrationConverter(BaseConverter):
    def __init__(self, idf: IDF):
        super().__init__(idf)

    def convert(self, data: dict) -> None:
        self.logger.info("Converting FenestrationSurface data...")
        fenestration_data = data.get("FenestrationSurface:Detailed", [])

        # Pass the complete data to validate method so it can access building surface info
        val_data = self.validate(fenestration_data)
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
            Outside_Boundary_Condition_Object=data.outside_boundary_condition_object
            or "",
            View_Factor_to_Ground=data.view_factor_to_ground or "",
            Frame_and_Divider_Name=data.frame_and_divider_name or "",
            Multiplier=data.multiplier,
            Number_of_Vertices=data.Number_of_Vertices,
        )

        for i, vertex in enumerate(data.vertices, 1):
            setattr(fenestration_obj, f"Vertex_{i}_Xcoordinate", vertex[0])
            setattr(fenestration_obj, f"Vertex_{i}_Ycoordinate", vertex[1])
            setattr(fenestration_obj, f"Vertex_{i}_Zcoordinate", vertex[2])

    def validate(
        self, fenestration_data: list[dict]
    ) -> list[FenestrationSurfaceSchema]:
        val_data = []
        try:
            fenestrationsurfaces = {"fenestrationsurfaces": fenestration_data}
            geometry = GeometrySchema.model_validate(fenestrationsurfaces)

            for f_surface in geometry.fenestrationsurfaces:
                val_data.append(f_surface)

        except Exception as e:
            self.logger.error(
                f"Geometry validation failed for fenestration surfaces: {e}"
            )
            self.state["failed"] += len(fenestration_data)
        return val_data
