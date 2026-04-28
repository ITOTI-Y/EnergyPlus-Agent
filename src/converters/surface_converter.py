from collections import defaultdict

from idfpy import IDF
from idfpy.models.thermal_zones import BuildingSurfaceDetailed, BuildingSurfaceDetailedVerticesItem

from src.converters.base_converter import BaseConverter
from src.validator.data_model import GeometrySchema, SurfaceSchema


class SurfaceConverter(BaseConverter):
    def __init__(self, idf: IDF):
        super().__init__(idf)

    def convert(self, data: dict) -> None:
        self.logger.info("Converting BuildingSurface data...")
        surface_data = data.get("BuildingSurface:Detailed", [])
        zone_to_surfaces = defaultdict(list)
        for surface in surface_data:
            zone_to_surfaces[surface["Zone Name"]].append(surface)
        val_data = self.validate(zone_to_surfaces)
        for surface in val_data:
            try:
                self._add_to_idf(surface)
                self.logger.success(
                    "Successfully converted BuildingSurface: {}", surface.name
                )
                self.state["success"] += 1
            except Exception:
                self.state["failed"] += 1
                self.logger.exception("Error Converting BuildingSurface Data")

    def _add_to_idf(self, val_data: SurfaceSchema) -> None:
        if self.idf.has("BuildingSurface:Detailed", val_data.name):
            self.logger.warning(
                "BuildingSurface with name {} already exists in IDF. "
                "Skipping addition.",
                val_data.name,
            )
            self.state["skipped"] += 1
            return
        vertices = [
            BuildingSurfaceDetailedVerticesItem(
                vertex_x_coordinate=float(v[0]),
                vertex_y_coordinate=float(v[1]),
                vertex_z_coordinate=float(v[2]),
            )
            for v in val_data.vertices
        ]
        self.idf.add(BuildingSurfaceDetailed(
            name=val_data.name,
            surface_type=val_data.surface_type,
            construction_name=val_data.construction_name,
            zone_name=val_data.zone_name,
            space_name=val_data.space_name or None,
            outside_boundary_condition=val_data.outside_boundary_condition,
            outside_boundary_condition_object=val_data.outside_boundary_condition_object or None,
            sun_exposure=val_data.sun_exposure,
            wind_exposure=val_data.wind_exposure,
            view_factor_to_ground=val_data.view_factor_to_ground,
            number_of_vertices=len(val_data.vertices),
            vertices=vertices,
        ))

    def validate(self, data: dict) -> list[SurfaceSchema]:
        val_data = []
        for _, surfaces in data.items():
            geometry = GeometrySchema.model_validate({"surfaces": surfaces})
            val_data.extend(geometry.surfaces)
        return val_data
