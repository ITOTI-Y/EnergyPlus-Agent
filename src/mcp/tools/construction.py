from typing import Any

from idfpy.models import (
    BuildingSurfaceDetailed,
    Construction,
    FenestrationSurfaceDetailed,
)

from src.mcp.state import ConfigState
from src.mcp.tools.base import BaseTool, normalize_payload

_LAYER_FIELDS = [
    "outside_layer",
    "layer_2",
    "layer_3",
    "layer_4",
    "layer_5",
    "layer_6",
    "layer_7",
    "layer_8",
    "layer_9",
    "layer_10",
]


class ConstructionTool(BaseTool):
    def __init__(self, state: ConfigState):
        super().__init__(state, "Construction")

    @property
    def object_types(self) -> tuple[str, ...]:
        return ("Construction",)

    def _create_model(self, data: dict[str, Any]) -> Construction:
        payload = normalize_payload(data)
        layers = payload.pop("layers", None)
        if layers:
            for idx, layer in enumerate(layers[: len(_LAYER_FIELDS)]):
                payload[_LAYER_FIELDS[idx]] = layer
        return Construction(**payload)

    def _get_name(self, instance: Construction) -> str:
        return instance.name

    def _check_references(self, name: str) -> list[str]:
        refs = []
        for surface in self.state.idf.all_of_type(BuildingSurfaceDetailed).values():
            if surface.construction_name == name:
                refs.append(f"Surface:{surface.name}")
        for fen in self.state.idf.all_of_type(FenestrationSurfaceDetailed).values():
            if fen.construction_name == name:
                refs.append(f"Fenestration:{fen.name}")
        return refs
