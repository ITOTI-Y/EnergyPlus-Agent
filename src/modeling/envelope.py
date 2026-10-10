"""Materials, constructions and the polygons of surfaces and fenestration."""

from typing import Final, Literal

from idfpy import IDF
from idfpy.models.constructions import (
    Construction,
    Material,
    MaterialAirGap,
    MaterialNoMass,
    WindowMaterialSimpleGlazingSystem,
)
from idfpy.models.thermal_zones import (
    BuildingSurfaceDetailedVerticesItem,
    FenestrationSurfaceDetailed,
)
from pydantic import BaseModel, ConfigDict, Field

from src.modeling.errors import ObjectNotFoundError

type MaterialObject = (
    Material | MaterialNoMass | MaterialAirGap | WindowMaterialSimpleGlazingSystem
)

MATERIAL_TYPES: Final[tuple[type[MaterialObject], ...]] = (
    Material,
    MaterialNoMass,
    MaterialAirGap,
    WindowMaterialSimpleGlazingSystem,
)

MAX_LAYERS: Final = 10

type Roughness = Literal[
    "VeryRough", "Rough", "MediumRough", "MediumSmooth", "Smooth", "VerySmooth"
]


# Declared properties matter: a free-form dict reaches the LLM as an object
# schema without properties, and Gemini then sends empty objects.
class VertexSchema(BaseModel):
    """One polygon vertex in metres, world coordinates."""

    model_config = ConfigDict(populate_by_name=True)

    x: float = Field(alias="X")
    y: float = Field(alias="Y")
    z: float = Field(alias="Z")


def find_material(idf: IDF, name: str) -> MaterialObject:
    """Raises: ObjectNotFoundError: If no material type has the name."""
    for material_type in MATERIAL_TYPES:
        if (material := idf.get(material_type, name)) is not None:
            return material
    raise ObjectNotFoundError("Material", name)


def all_materials(idf: IDF) -> list[MaterialObject]:
    return [m for t in MATERIAL_TYPES for m in idf.all_of_type(t).values()]


def layer_fields(layers: list[str]) -> dict[str, str | None]:
    """All ten Construction layer fields, unused ones set to None.

    Raises:
        ValueError: If there are no layers or more than ten.
    """
    if not 1 <= len(layers) <= MAX_LAYERS:
        raise ValueError(f"A construction takes 1 to {MAX_LAYERS} layers.")
    padded = [*layers, *[None] * (MAX_LAYERS - len(layers))]
    names = ["outside_layer", *(f"layer_{i}" for i in range(2, MAX_LAYERS + 1))]
    return dict(zip(names, padded, strict=True))


def construction_from_layers(name: str, layers: list[str]) -> Construction:
    """Construction with ``layers`` ordered from outside to inside."""
    return Construction.model_validate({"name": name, **layer_fields(layers)})


def surface_geometry(vertices: list[VertexSchema]) -> dict[str, object]:
    """Vertex fields of BuildingSurface:Detailed.

    Raises:
        ValueError: If there are fewer than 3 vertices.
    """
    if len(vertices) < 3:
        raise ValueError(f"A surface needs >= 3 vertices, got {len(vertices)}.")
    return {
        "number_of_vertices": len(vertices),
        "vertices": [
            BuildingSurfaceDetailedVerticesItem(
                vertex_x_coordinate=v.x,
                vertex_y_coordinate=v.y,
                vertex_z_coordinate=v.z,
            )
            for v in vertices
        ],
    }


def fenestration_vertices(vertices: list[VertexSchema]) -> dict[str, float | None]:
    """Vertex fields of FenestrationSurface:Detailed, which holds 3 or 4 points.

    The fourth vertex is set to None for a triangle, so an update from four
    vertices to three clears it.

    Raises:
        ValueError: If there are fewer than 3 or more than 4 vertices.
    """
    if not 3 <= len(vertices) <= 4:
        raise ValueError(f"Fenestration takes 3 or 4 vertices, got {len(vertices)}.")
    fields: dict[str, float | None] = {"number_of_vertices": float(len(vertices))}
    for i in range(1, 5):
        v = vertices[i - 1] if i <= len(vertices) else None
        fields |= {
            f"vertex_{i}_x_coordinate": v.x if v else None,
            f"vertex_{i}_y_coordinate": v.y if v else None,
            f"vertex_{i}_z_coordinate": v.z if v else None,
        }
    return fields


def fenestration_from_vertices(
    vertices: list[VertexSchema], **fields: object
) -> FenestrationSurfaceDetailed:
    return FenestrationSurfaceDetailed.model_validate(
        {**fields, **fenestration_vertices(vertices)}
    )
