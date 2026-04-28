from idfpy import IDF
from idfpy.models.thermal_zones import FenestrationSurfaceDetailed

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

        val_data = self.validate({"fenestrationsurfaces": fenestration_data})
        for fenestration in val_data.fenestrationsurfaces:
            try:
                self._add_to_idf(fenestration)
                self.logger.success(
                    "Successfully converted FenestrationSurface: {}",
                    fenestration.name,
                )
                self.state["success"] += 1
            except Exception:
                self.state["failed"] += 1
                self.logger.exception("Error Converting FenestrationSurface Data")

    def _add_to_idf(self, val_data: FenestrationSurfaceSchema) -> None:
        if self.idf.has("FenestrationSurface:Detailed", val_data.name):
            self.logger.warning(
                "FenestrationSurface with name {} already exists in IDF. "
                "Skipping addition.",
                val_data.name,
            )
            self.state["skipped"] += 1
            return

        if not self.idf.has("Construction", val_data.construction_name):
            raise ValueError(
                f"Construction {val_data.construction_name} does not exist in IDF"
            )

        verts = val_data.vertices
        kwargs: dict = dict(
            name=val_data.name,
            surface_type=val_data.surface_type,
            construction_name=val_data.construction_name,
            building_surface_name=val_data.building_surface_name,
            outside_boundary_condition_object=val_data.outside_boundary_condition_object or None,
            view_factor_to_ground=val_data.view_factor_to_ground if val_data.view_factor_to_ground != "autocalculate" else None,
            frame_and_divider_name=val_data.frame_and_divider_name or None,
            multiplier=val_data.multiplier,
            number_of_vertices=len(verts),
        )
        for i, vertex in enumerate(verts, 1):
            kwargs[f"vertex_{i}_x_coordinate"] = float(vertex[0])
            kwargs[f"vertex_{i}_y_coordinate"] = float(vertex[1])
            kwargs[f"vertex_{i}_z_coordinate"] = float(vertex[2])
        self.idf.add(FenestrationSurfaceDetailed(**kwargs))

    def validate(self, data: dict) -> GeometrySchema:
        try:
            geometry = GeometrySchema.model_validate(data)
        except Exception as e:
            self.logger.error(
                "Geometry validation failed for fenestration surfaces: {}", e
            )
            self.state["failed"] += len(data)
        return geometry
