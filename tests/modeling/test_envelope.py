import pytest
from idfpy import IDF
from idfpy.models.constructions import (
    Construction,
    MaterialAirGap,
    MaterialNoMass,
    WindowMaterialGas,
    WindowMaterialGlazing,
    WindowMaterialSimpleGlazingSystem,
)
from idfpy.models.thermal_zones import BuildingSurfaceDetailed

from src.modeling.envelope import (
    VertexSchema,
    check_construction_fits,
    checked_construction,
    construction_from_layers,
    fenestration_vertices,
    find_material,
    layer_changes,
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


def _window_materials(idf: IDF) -> None:
    idf.add(
        WindowMaterialGlazing(
            name="Glass", optical_data_type="SpectralAverage", thickness=0.006
        )
    )
    idf.add(WindowMaterialGas(name="Air", gas_type="Air", thickness=0.013))
    idf.add(MaterialAirGap(name="Cavity", thermal_resistance=0.18))
    idf.add(
        WindowMaterialSimpleGlazingSystem(
            name="Simple", u_factor=2.0, solar_heat_gain_coefficient=0.4
        )
    )
    idf.add(MaterialNoMass(name="Board", roughness="Smooth", thermal_resistance=0.5))


@pytest.mark.parametrize(
    "layers",
    [["Glass", "Air", "Glass"], ["Simple"], ["Board", "Cavity", "Board"]],
    ids=["double_glazing", "simple_glazing", "opaque_with_cavity"],
)
def test_valid_layerings_are_accepted(layers):
    idf = IDF()
    _window_materials(idf)

    assert checked_construction(idf, "C", layers).outside_layer == layers[0]


@pytest.mark.parametrize(
    ("layers", "reason"),
    [
        (["Glass", "Cavity", "Glass"], "mixes window and opaque"),
        (["Simple", "Air", "Glass"], "SimpleGlazingSystem with other layers"),
        (["Glass", "Glass"], "invalid order"),
        (["Air", "Glass"], "invalid order"),
        (["Glass", "Air"], "invalid order"),
    ],
    ids=[
        "airgap_between_panes",
        "simple_plus_layers",
        "adjacent_glass",
        "gas_first",
        "gas_last",
    ],
)
def test_invalid_layerings_are_rejected(layers, reason):
    idf = IDF()
    _window_materials(idf)

    with pytest.raises(ValueError, match=reason):
        checked_construction(idf, "C", layers)


def test_construction_must_suit_surface_type():
    idf = IDF()
    _window_materials(idf)
    idf.add(Construction(name="Window", outside_layer="Simple"))
    idf.add(Construction(name="Wall", outside_layer="Board"))

    check_construction_fits(idf, "Window", "Window")
    check_construction_fits(idf, "Wall", "Door")
    with pytest.raises(ValueError, match=r"Constructions that fit: \['Wall'\]"):
        check_construction_fits(idf, "Window", "Wall")
    with pytest.raises(ValueError, match="needs a window construction"):
        check_construction_fits(idf, "Wall", "GlassDoor")


def test_layers_of_a_used_construction_keep_its_kind():
    idf = IDF()
    _window_materials(idf)
    wall = construction_from_layers("Wall", ["Board"])
    idf.add(wall)
    idf.add(
        BuildingSurfaceDetailed.model_validate(
            {
                "name": "S",
                "surface_type": "Wall",
                "construction_name": "Wall",
                "zone_name": "Z",
                "outside_boundary_condition": "Outdoors",
                "vertices": [],
            }
        )
    )

    assert layer_changes(idf, wall, ["Board", "Board"])["layer_2"] == "Board"
    with pytest.raises(ValueError, match="in use"):
        layer_changes(idf, wall, ["Glass", "Air", "Glass"])
