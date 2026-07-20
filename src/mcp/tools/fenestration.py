from typing import Any

from idfpy.models import FenestrationSurfaceDetailed

from src.mcp.state import ConfigState
from src.mcp.tools.base import BaseTool, normalize_payload


class FenestrationTool(BaseTool):
    def __init__(self, state: ConfigState):
        super().__init__(state, "FenestrationSurface")

    @property
    def object_types(self) -> tuple[str, ...]:
        return ("FenestrationSurface:Detailed",)

    def _create_model(self, data: dict[str, Any]) -> FenestrationSurfaceDetailed:
        payload = normalize_payload(data)
        vertices = payload.pop("vertices", None)
        if str(payload.get("view_factor_to_ground", "")).lower() == "autocalculate":
            payload["view_factor_to_ground"] = None
        if vertices is not None:
            payload["number_of_vertices"] = len(vertices)
            for idx, vertex in enumerate(vertices, start=1):
                if isinstance(vertex, dict):
                    x = vertex.get("X", vertex.get("x"))
                    y = vertex.get("Y", vertex.get("y"))
                    z = vertex.get("Z", vertex.get("z"))
                else:
                    x, y, z = vertex[0], vertex[1], vertex[2]
                payload[f"vertex_{idx}_x_coordinate"] = float(x)
                payload[f"vertex_{idx}_y_coordinate"] = float(y)
                payload[f"vertex_{idx}_z_coordinate"] = float(z)
        return FenestrationSurfaceDetailed(**payload)

    def _get_name(self, instance: FenestrationSurfaceDetailed) -> str:
        return instance.name

    def _check_references(self, name: str) -> list[str]:
        return []
