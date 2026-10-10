import math
import random
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


type Corners = list[tuple[float, float]]
type ZoneSpec = tuple[str, Corners, float, float]


def _box(x0: float, y0: float, x1: float, y1: float) -> Corners:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _rotated(corners: Corners, degrees: float) -> Corners:
    a = math.radians(degrees)
    return [
        (x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a))
        for x, y in corners
    ]


def _plan_area(corners: Corners) -> float:
    pairs = zip(corners, corners[1:] + corners[:1], strict=True)
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in pairs)) / 2


# Paired-face counts follow from the layouts: e.g. the 3 x 3 x 2 grid has
# 2 x 12 shared walls and 9 shared floors, two faces each.
COMPLEX_LAYOUTS: dict[str, tuple[list[ZoneSpec], int]] = {
    "grid_3x3x2": (
        [
            (f"Z{i}{j}_{k}", _box(i * 6, j * 5, i * 6 + 6, j * 5 + 5), k * 3.5, 3.5)
            for i in range(3)
            for j in range(3)
            for k in range(2)
        ],
        66,
    ),
    "staggered_storeys": (
        [
            ("S0", _box(0, 0, 12, 8), 0, 3),
            ("S1", _box(3, 2, 15, 10), 3, 3),
            ("S2", _box(-2, 4, 10, 12), 6, 3),
        ],
        4,
    ),
    "courtyard_ring": (
        [
            ("North", _box(0, 12, 16, 16), 0, 4),
            ("South", _box(0, 0, 16, 4), 0, 4),
            ("East", _box(12, 4, 16, 12), 0, 4),
            ("West", _box(0, 4, 4, 12), 0, 4),
        ],
        8,
    ),
    "rotated_l_with_neighbour": (
        [
            (
                "L",
                _rotated([(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)], 30),
                0,
                3,
            ),
            ("R", _rotated(_box(10, 0, 14, 4), 30), 0, 3),
        ],
        2,
    ),
    "slanted_shared_edge": (
        [
            ("Tri", [(0, 0), (8, 0), (0, 6)], 0, 3),
            ("Trap", [(8, 0), (12, 0), (12, 6), (0, 6)], 0, 3),
        ],
        2,
    ),
    "wall_against_two_heights": (
        [
            ("Long", _box(0, 0, 4, 12), 0, 6),
            ("Low", _box(4, 0, 9, 6), 0, 3),
            ("Mid", _box(4, 6, 9, 12), 0, 4.5),
        ],
        6,
    ),
    "corner_contact_only": (
        [("P", _box(0, 0, 5, 5), 0, 3), ("Q", _box(5, 5, 10, 10), 0, 3)],
        0,
    ),
    "cantilevered_upper_floor": (
        [("Base", _box(0, 0, 6, 6), 0, 3), ("Over", _box(-2, 0, 8, 6), 3, 3)],
        2,
    ),
    "atrium_beside_three_storeys": (
        [
            ("Atrium", _box(0, 0, 6, 10), 0, 10.5),
            *[(f"Floor_{k}", _box(6, 0, 16, 10), k * 3.5, 3.5) for k in range(3)],
        ],
        10,
    ),
    "concave_non_rectilinear": (
        [
            (
                "Star",
                [(0, 0), (6, 2), (12, 0), (10, 6), (12, 12), (6, 9), (0, 12), (2, 6)],
                0,
                3,
            ),
            ("Next", [(12, 0), (18, 0), (18, 12), (12, 12), (10, 6)], 0, 3),
        ],
        4,
    ),
    "thirds_coordinates": (
        [
            (f"T{i}", _box(i * 10 / 3, 0, (i + 1) * 10 / 3, 7 / 3), 0, 2.9)
            for i in range(3)
        ]
        + [("Top", _box(0, 0, 10, 7 / 3), 2.9, 3.1)],
        10,
    ),
    "half_offset_upper_grid": (
        [
            (f"G{i}{j}_0", _box(i * 6, j * 6, i * 6 + 6, j * 6 + 6), 0, 3)
            for i in range(2)
            for j in range(2)
        ]
        + [
            (f"G{i}{j}_1", _box(i * 6 + 3, j * 6 + 3, i * 6 + 9, j * 6 + 9), 3, 3)
            for i in range(2)
            for j in range(2)
        ],
        34,
    ),
}


def _build(layout: list[ZoneSpec]) -> IDF:
    idf = _model()
    # Creation order must not matter; a fixed shuffle exercises that.
    for name, corners, z, height in random.Random(7).sample(layout, len(layout)):
        _zone(idf, name, _plan(*corners), z, height)
    return idf


@pytest.mark.parametrize("layout", COMPLEX_LAYOUTS, ids=str)
def test_complex_layouts_give_closed_convex_paired_zones(layout):
    zones, paired_faces = COMPLEX_LAYOUTS[layout]

    idf = _build(zones)

    assert model_issues(idf) == []
    assert len(_pairs_all(idf)) == paired_faces
    for name, corners, _, height in zones:
        surfaces = _surfaces(idf, name)
        # A closed zone with outward normals has zero total area vector, and
        # the divergence theorem gives its volume.
        closure = [sum(s.area * s.normal[k] for s in surfaces) for k in range(3)]
        volume = (
            sum(
                s.area * sum(c * n for c, n in zip(s.centroid, s.normal, strict=True))
                for s in surfaces
            )
            / 3
        )
        assert max(map(abs, closure)) < 1e-3, name
        assert volume == pytest.approx(_plan_area(corners) * height, rel=1e-6), name
        assert all(s.is_convex for s in surfaces), name


def _pairs_all(idf: IDF) -> list[str]:
    return [
        s.name
        for s in idf.all_of_type(BuildingSurfaceDetailed).values()
        if s.outside_boundary_condition == "Surface"
    ]


@pytest.mark.skipif(shutil.which("energyplus") is None, reason="EnergyPlus not on PATH")
@pytest.mark.parametrize(
    "layout",
    [
        "atrium_beside_three_storeys",
        "concave_non_rectilinear",
        "half_offset_upper_grid",
    ],
)
def test_complex_layouts_simulate(layout, tmp_path):
    idf = _build(COMPLEX_LAYOUTS[layout][0])
    add_design_days(idf, DATA / "weather" / "Shenzhen.ddy")
    idf.save(tmp_path / "in.idf")

    result = run_energyplus(
        tmp_path / "in.idf", DATA / "weather" / "Shenzhen.epw", tmp_path
    )

    assert result.succeeded, result.errors
