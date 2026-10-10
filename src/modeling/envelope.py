"""Materials, constructions and the polygons of surfaces and fenestration."""

from itertools import pairwise
from typing import Final, Literal

from idfpy import IDF
from idfpy.ext.construction.functions import is_window_material
from idfpy.models.constructions import (
    Construction,
    Material,
    MaterialAirGap,
    MaterialNoMass,
    WindowMaterialGas,
    WindowMaterialGlazing,
    WindowMaterialSimpleGlazingSystem,
)
from idfpy.models.thermal_zones import (
    BuildingSurfaceDetailedVerticesItem,
    FenestrationSurfaceDetailed,
)
from pydantic import BaseModel, ConfigDict, Field

from src.modeling.errors import ObjectNotFoundError

type MaterialObject = (
    Material
    | MaterialNoMass
    | MaterialAirGap
    | WindowMaterialSimpleGlazingSystem
    | WindowMaterialGlazing
    | WindowMaterialGas
)

MATERIAL_TYPES: Final[tuple[type[MaterialObject], ...]] = (
    Material,
    MaterialNoMass,
    MaterialAirGap,
    WindowMaterialSimpleGlazingSystem,
    WindowMaterialGlazing,
    WindowMaterialGas,
)

WINDOW_LAYERING: Final = (
    "A window construction is either a single WindowMaterial:SimpleGlazingSystem "
    "or WindowMaterial:Glazing layers with exactly one WindowMaterial:Gas between "
    "each pair, starting and ending with glazing."
)

# Fenestration types whose construction must transmit light; every other
# surface and the Door type need an opaque construction.
_GLAZED_TYPES: Final = frozenset({"Window", "GlassDoor"})

type ConstructionKind = Literal["window", "opaque", "mixed"]

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


def layering_problem(materials: list[MaterialObject]) -> str | None:
    """Why EnergyPlus would reject these layers as one construction, if it would.

    EnergyPlus stops on a window construction that breaks its layering rules
    and on a surface that mixes window and opaque materials; a
    Material:AirGap between panes is accepted but its resistance is ignored.
    """
    glazed = [is_window_material(m) for m in materials]
    if not any(glazed):
        return None
    if not all(glazed):
        return (
            "mixes window and opaque materials; separate glazing layers with "
            f"WindowMaterial:Gas, not Material:AirGap. {WINDOW_LAYERING}"
        )
    if len(materials) > 1 and any(
        isinstance(m, WindowMaterialSimpleGlazingSystem) for m in materials
    ):
        return f"combines a SimpleGlazingSystem with other layers. {WINDOW_LAYERING}"
    is_gas = [isinstance(m, WindowMaterialGas) for m in materials]
    if is_gas[0] or is_gas[-1] or any(a == b for a, b in pairwise(is_gas)):
        return f"has glazing and gas layers in an invalid order. {WINDOW_LAYERING}"
    return None


def checked_construction(idf: IDF, name: str, layers: list[str]) -> Construction:
    """Construction whose layers exist and form a valid assembly.

    Raises:
        ObjectNotFoundError: If a layer names no material.
        ValueError: If the layer count or layering is invalid.
    """
    construction = construction_from_layers(name, layers)
    if problem := layering_problem([find_material(idf, layer) for layer in layers]):
        raise ValueError(f"Construction '{name}' {problem}")
    return construction


def layer_changes(
    idf: IDF, construction: Construction, layers: list[str]
) -> dict[str, str | None]:
    """Layer fields that replace a construction's layers.

    The new layers must form a valid assembly of the same kind, since the
    surfaces and openings already using the construction depend on it.

    Raises:
        ObjectNotFoundError: If a layer names no material.
        ValueError: If the layering is invalid or the kind would change.
    """
    replacement = checked_construction(idf, construction.name, layers)
    glazed = is_window_material(find_material(idf, layers[0]))
    current = construction_kind(construction)
    if current != ("window" if glazed else "opaque") and construction.referencing():
        raise ValueError(
            f"Construction '{construction.name}' is {current} and in use; new "
            "layers must keep it that way. Create a separate construction instead."
        )
    return {k: v for k, v in replacement.model_dump().items() if k != "name"}


def construction_kind(construction: Construction) -> ConstructionKind:
    """Whether a construction in the model is glazed, opaque or invalidly mixed."""
    if construction.is_window_construction:
        return "window"
    if construction.is_opaque_construction:
        return "opaque"
    return "mixed"


def check_construction_fits(
    idf: IDF, construction_name: str, surface_type: str
) -> None:
    """Reject a construction whose kind does not suit the surface type.

    Window and GlassDoor need a window construction; Door and every base
    surface need an opaque one. EnergyPlus stops on either mismatch.

    Raises:
        ValueError: On a mismatch, listing the constructions that would fit.
    """
    construction = idf.get(Construction, construction_name)
    if construction is None:
        return  # reported as a missing reference
    needed: ConstructionKind = "window" if surface_type in _GLAZED_TYPES else "opaque"
    if construction_kind(construction) == needed:
        return
    fitting = sorted(
        c.name
        for c in idf.all_of_type(Construction).values()
        if construction_kind(c) == needed
    )
    raise ValueError(
        f"A {surface_type} needs a {needed} construction; '{construction_name}' "
        f"is not one. Constructions that fit: {fitting or 'none yet'}."
    )


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
