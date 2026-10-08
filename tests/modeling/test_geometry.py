import shutil
from pathlib import Path

import pytest
from idfpy import IDF
from idfpy.models.constructions import Construction, Material
from idfpy.models.simulation import Building
from idfpy.models.thermal_zones import (
    BuildingSurfaceDetailed,
    FenestrationSurfaceDetailed,
    Zone,
)

from src.modeling.errors import ReferencedObjectError
from src.modeling.geometry import (
    PlanPointSchema,
    ZoneConstructions,
    create_zone_geometry,
)
from src.modeling.validation import model_issues
from src.runner.runner import run_energyplus
from src.state.defaults import add_design_days, new_model

DATA = Path(__file__).parents[2] / "data"
CONSTRUCTIONS = ZoneConstructions("Ext", "Roof", "Slab", "Int", "Deck")


def _model() -> IDF:
    idf = new_model()
    idf.add(Building(name="B", solar_distribution="FullInteriorAndExterior"))
    idf.add(
        Material(
            name="Concrete",
            roughness="Rough",
            thickness=0.2,
            conductivity=1.4,
            density=2100.0,
            specific_heat=900.0,
        )
    )
    for name in ("Ext", "Roof", "Slab", "Int", "Deck"):
        idf.add(Construction(name=name, outside_layer="Concrete"))
    return idf


def _plan(*corners: tuple[float, float]) -> list[PlanPointSchema]:
    return [PlanPointSchema(X=x, Y=y) for x, y in corners]


def _rect(x0: float, y0: float, x1: float, y1: float) -> list[PlanPointSchema]:
    return _plan((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def _zone(idf: IDF, name: str, plan: list[PlanPointSchema], z: float, height: float):
    idf.add(Zone(name=name))
    return create_zone_geometry(idf, name, plan, z, height, CONSTRUCTIONS)


def _surfaces(idf: IDF, zone: str) -> list[BuildingSurfaceDetailed]:
    return [
        s
        for s in idf.all_of_type(BuildingSurfaceDetailed).values()
        if s.zone_name == zone
    ]


def _area(idf: IDF, zone: str, *types: str, boundary: str | None = None) -> float:
    return sum(
        s.area
        for s in _surfaces(idf, zone)
        if s.surface_type in types
        and (boundary is None or s.outside_boundary_condition == boundary)
    )


def _pairs(idf: IDF) -> set[tuple[str, str, float]]:
    return {
        (s.zone_name, s.surface_type, round(s.area, 3))
        for s in idf.all_of_type(BuildingSurfaceDetailed).values()
        if s.outside_boundary_condition == "Surface"
    }


def test_box_has_outdoor_walls_ground_floor_and_roof():
    idf = _model()

    _zone(idf, "A", _rect(0, 0, 10, 8), 0, 3)

    assert {
        (s.surface_type, s.outside_boundary_condition) for s in _surfaces(idf, "A")
    } == {
        ("Wall", "Outdoors"),
        ("Floor", "Ground"),
        ("Roof", "Outdoors"),
    }
    assert _area(idf, "A", "Wall") == pytest.approx(108.0)
    assert model_issues(idf) == []


@pytest.mark.parametrize("order", [("A", "B"), ("B", "A")])
def test_side_by_side_zones_share_a_wall_in_either_order(order):
    idf = _model()
    plans = {"A": _rect(0, 0, 5, 8), "B": _rect(5, 0, 10, 8)}

    for name in order:
        _zone(idf, name, plans[name], 0, 3)

    assert _pairs(idf) == {("A", "Wall", 24.0), ("B", "Wall", 24.0)}
    assert model_issues(idf) == []


def test_taller_neighbour_wall_is_split_at_the_lower_roof():
    idf = _model()
    _zone(idf, "Low", _rect(0, 0, 5, 8), 0, 3)

    _zone(idf, "High", _rect(5, 2, 10, 6), 0, 5)

    assert _pairs(idf) == {("Low", "Wall", 12.0), ("High", "Wall", 12.0)}
    # The high zone's wall facing Low keeps an outdoor strip above Low's roof.
    assert _area(idf, "High", "Wall") == pytest.approx(5 * 2 * 5 + 4 * 5 * 2)
    assert model_issues(idf) == []


def test_misaligned_storeys_pair_only_their_overlap():
    idf = _model()
    _zone(idf, "Ground", _rect(0, 0, 10, 8), 0, 3)

    _zone(idf, "Upper", _rect(2, 2, 7, 6), 3, 3)

    assert _pairs(idf) == {("Ground", "Ceiling", 20.0), ("Upper", "Floor", 20.0)}
    roof = [s for s in _surfaces(idf, "Ground") if s.surface_type == "Roof"]
    # 80 m2 roof minus the 20 m2 under Upper, cut around the hole into convex parts.
    assert sum(s.area for s in roof) == pytest.approx(60.0)
    assert all(s.is_convex for s in roof)
    assert model_issues(idf) == []


def test_l_shaped_zone_gets_convex_faces_and_exterior_solar_distribution():
    idf = _model()

    result = _zone(
        idf, "L", _plan((0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)), 0, 3
    )

    assert all(s.is_convex for s in _surfaces(idf, "L"))
    assert _area(idf, "L", "Roof") == pytest.approx(64.0)
    building = idf.get(Building, "B")
    assert building is not None
    assert building.solar_distribution == "FullExterior"
    assert result.notes


def test_overlapping_zone_is_rejected():
    idf = _model()
    _zone(idf, "A", _rect(0, 0, 5, 8), 0, 3)

    with pytest.raises(ValueError, match="overlap zone 'A'"):
        _zone(idf, "X", _rect(4, 0, 9, 8), 0, 3)


def test_face_with_an_opening_is_not_split():
    idf = _model()
    _zone(idf, "A", _rect(0, 0, 5, 8), 0, 3)
    [east] = [
        s for s in _surfaces(idf, "A") if s.surface_type == "Wall" and s.normal[0] > 0.5
    ]
    idf.add(
        FenestrationSurfaceDetailed(
            name="Win",
            surface_type="Window",
            construction_name="Ext",
            building_surface_name=east.name,
            number_of_vertices=4,
            vertex_1_x_coordinate=5,
            vertex_1_y_coordinate=3,
            vertex_1_z_coordinate=2,
            vertex_2_x_coordinate=5,
            vertex_2_y_coordinate=3,
            vertex_2_z_coordinate=1,
            vertex_3_x_coordinate=5,
            vertex_3_y_coordinate=5,
            vertex_3_z_coordinate=1,
            vertex_4_x_coordinate=5,
            vertex_4_y_coordinate=5,
            vertex_4_z_coordinate=2,
        )
    )
    before = set(idf.all_of_type(BuildingSurfaceDetailed))

    with pytest.raises(ReferencedObjectError):
        _zone(idf, "B", _rect(5, 0, 10, 8), 0, 3)

    assert set(idf.all_of_type(BuildingSurfaceDetailed)) == before


@pytest.mark.skipif(shutil.which("energyplus") is None, reason="EnergyPlus not on PATH")
def test_extruded_building_simulates(tmp_path):
    idf = _model()
    _zone(idf, "A", _rect(0, 0, 5, 8), 0, 3)
    _zone(idf, "Upper", _rect(0, 0, 4, 8), 3, 3)
    _zone(idf, "B", _rect(5, 2, 10, 6), 0, 5)
    _zone(idf, "L", _plan((20, 0), (30, 0), (30, 4), (24, 4), (24, 10), (20, 10)), 0, 3)
    add_design_days(idf, DATA / "weather" / "Shenzhen.ddy")
    idf.save(tmp_path / "in.idf")

    result = run_energyplus(
        tmp_path / "in.idf", DATA / "weather" / "Shenzhen.epw", tmp_path
    )

    assert result.succeeded, result.errors
    assert not [m for m in result.messages if "non-convex" in m.text]
