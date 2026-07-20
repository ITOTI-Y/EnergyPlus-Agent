from typing import Any

from idfpy.models import (
    BuildingSurfaceDetailed,
    BuildingSurfaceDetailedVerticesItem,
    FenestrationSurfaceDetailed,
)

from src.mcp.state import ConfigState
from src.mcp.tools.base import BaseTool, normalize_payload


def _vertex_items(
    vertices: list[Any] | None,
) -> list[BuildingSurfaceDetailedVerticesItem]:
    items: list[BuildingSurfaceDetailedVerticesItem] = []
    for vertex in vertices or []:
        if isinstance(vertex, dict):
            x = vertex.get("X", vertex.get("x", vertex.get("vertex_x_coordinate")))
            y = vertex.get("Y", vertex.get("y", vertex.get("vertex_y_coordinate")))
            z = vertex.get("Z", vertex.get("z", vertex.get("vertex_z_coordinate")))
        else:
            x, y, z = vertex[0], vertex[1], vertex[2]
        items.append(
            BuildingSurfaceDetailedVerticesItem(
                vertex_x_coordinate=float(x),
                vertex_y_coordinate=float(y),
                vertex_z_coordinate=float(z),
            )
        )
    return items


class SurfaceTool(BaseTool):
    def __init__(self, state: ConfigState):
        super().__init__(state, "Surface")

    @property
    def object_types(self) -> tuple[str, ...]:
        return ("BuildingSurface:Detailed",)

    def _create_model(self, data: dict[str, Any]) -> BuildingSurfaceDetailed:
        payload = normalize_payload(data)
        vertices = payload.pop("vertices", None)
        if vertices is not None:
            vertex_items = _vertex_items(vertices)
            payload["vertices"] = vertex_items
            payload["number_of_vertices"] = len(vertex_items)
        return BuildingSurfaceDetailed(**payload)

    def _get_name(self, instance: BuildingSurfaceDetailed) -> str:
        return instance.name

    def _check_references(self, name: str) -> list[str]:
        refs = []
        for fen in self.state.idf.all_of_type(FenestrationSurfaceDetailed).values():
            if fen.building_surface_name == name:
                refs.append(f"FenestrationSurface:{fen.name}")
        return refs
