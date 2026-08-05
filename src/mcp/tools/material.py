from typing import Any

from idfpy.models.constructions import (
    Construction,
    Material,
    MaterialAirGap,
    MaterialNoMass,
    WindowMaterialSimpleGlazingSystem,
)

from src.mcp.state import ConfigState
from src.mcp.tools.base import BaseTool, normalize_payload

_OBJECT_TYPE_TO_MATERIAL_TYPE = {
    "Material": "Standard",
    "Material:NoMass": "NoMass",
    "MaterialNoMass": "NoMass",
    "Material:AirGap": "AirGap",
    "MaterialAirGap": "AirGap",
    "WindowMaterial:SimpleGlazingSystem": "Glazing",
}


class MaterialTool(BaseTool):
    def __init__(self, state: ConfigState):
        super().__init__(state, "Material")

    @property
    def object_types(self) -> tuple[str, ...]:
        return (
            "Material",
            "Material:NoMass",
            "MaterialNoMass",
            "Material:AirGap",
            "MaterialAirGap",
            "WindowMaterial:SimpleGlazingSystem",
        )

    def _create_model(
        self,
        data: dict[str, Any],
        *,
        existing_object_type: str | None = None,
    ) -> Material | MaterialNoMass | MaterialAirGap | WindowMaterialSimpleGlazingSystem:
        payload = normalize_payload(data)
        material_type = payload.pop("type", None)
        if material_type is None:
            material_type = payload.pop("material_type", None)
        else:
            payload.pop("material_type", None)

        if material_type is None and existing_object_type is not None:
            material_type = _OBJECT_TYPE_TO_MATERIAL_TYPE.get(existing_object_type)

        if material_type is None:
            if "u_factor" in payload:
                material_type = "Glazing"
            elif "thermal_resistance" in payload and "roughness" in payload:
                material_type = "NoMass"
            elif "thermal_resistance" in payload:
                material_type = "AirGap"
            else:
                material_type = "Standard"
        if material_type == "Standard":
            return Material(**payload)
        if material_type == "NoMass":
            return MaterialNoMass(**payload)
        if material_type == "AirGap":
            return MaterialAirGap(**payload)
        if material_type == "Glazing":
            return WindowMaterialSimpleGlazingSystem(**payload)
        raise ValueError(f"Unknown material type: {material_type}")

    def _get_name(self, instance: Any) -> str:
        return instance.name

    def _check_references(self, name: str) -> list[str]:
        refs = []
        layer_fields = [
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
        for construction in self.state.idf.all_of_type(Construction).values():
            if any(
                getattr(construction, field, None) == name for field in layer_fields
            ):
                refs.append(f"Construction:{construction.name}")
        return refs
