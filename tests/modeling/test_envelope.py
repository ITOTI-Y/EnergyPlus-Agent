import pytest
from idfpy import IDF
from idfpy.models.constructions import MaterialNoMass

from src.modeling.envelope import (
    VertexSchema,
    construction_from_layers,
    fenestration_vertices,
    find_material,
    layer_fields,
)
from src.modeling.errors import ObjectNotFoundError


def _square(n: int) -> list[VertexSchema]:
    corners = [(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 1.5)]
    return [VertexSchema(X=x, Y=0.0, Z=z) for x, z in corners[:n]]


def test_construction_layers_are_ordered_and_padded():
    construction = construction_from_layers("Wall", ["Brick", "Insulation", "Board"])

    assert construction.outside_layer == "Brick"
    assert construction.layer_3 == "Board"
    assert layer_fields(["Brick"])["layer_10"] is None


@pytest.mark.parametrize("layers", [[], [f"L{i}" for i in range(11)]])
def test_construction_rejects_layer_count(layers):
    with pytest.raises(ValueError, match="1 to 10 layers"):
        construction_from_layers("Wall", layers)


def test_triangular_fenestration_clears_fourth_vertex():
    fields = fenestration_vertices(_square(3))

    assert fields["number_of_vertices"] == 3.0
    assert fields["vertex_4_x_coordinate"] is None


@pytest.mark.parametrize("count", [2, 5])
def test_fenestration_rejects_vertex_count(count):
    with pytest.raises(ValueError, match="3 or 4 vertices"):
        fenestration_vertices(_square(count))


def test_find_material_searches_every_material_type():
    idf = IDF()
    idf.add(MaterialNoMass(name="Board", roughness="Smooth", thermal_resistance=0.5))

    assert isinstance(find_material(idf, "Board"), MaterialNoMass)
    with pytest.raises(ObjectNotFoundError):
        find_material(idf, "Missing")
