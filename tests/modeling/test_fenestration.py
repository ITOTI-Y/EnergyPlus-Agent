import json
import shutil
from pathlib import Path

import pytest
from idfpy import IDF
from idfpy.models.constructions import (
    Construction,
    Material,
    WindowMaterialGas,
    WindowMaterialGlazing,
    WindowMaterialSimpleGlazingSystem,
)
from idfpy.models.simulation import Building
from idfpy.models.thermal_zones import (
    BuildingSurfaceDetailed,
    FenestrationSurfaceDetailed,
    Zone,
)

from src.modeling.envelope import VertexSchema, fenestration_from_vertices
from src.modeling.errors import DuplicateNameError
from src.modeling.fenestration import (
    PARTNER_SUFFIX,
    add_fenestration,
    add_windows_by_ratio,
    facing,
    placement_problem,
    remove_fenestration,
    update_fenestration,
)
from src.modeling.geometry import (
    PlanPointSchema,
    ZoneConstructions,
    create_zone_geometry,
)
from src.runner.runner import run_energyplus
from src.state.defaults import add_design_days

DATA = Path(__file__).parents[2] / "data"
# building_schema has outdoor wall Zone_East_Wall_South (y=0, x 5..10, z 0..3,
# facing -y) and the shared wall Zone_East_Wall_Internal / Zone_West_Wall_Internal
# at x=5, y 0..5, z 0..3.
SOUTH_WINDOW = [(6.0, 0.0, 2.0), (6.0, 0.0, 1.0), (8.0, 0.0, 1.0), (8.0, 0.0, 2.0)]
INTERNAL_DOOR = [(5.0, 3.0, 2.0), (5.0, 3.0, 0.0), (5.0, 2.0, 0.0), (5.0, 2.0, 2.0)]


def _model() -> IDF:
    return IDF.from_dict(
        json.loads((DATA / "schemas" / "building_schema.epJSON").read_text())
    )


def _opening(name: str, kind: str, wall: str, points, construction: str):
    return fenestration_from_vertices(
        [VertexSchema(X=x, Y=y, Z=z) for x, y, z in points],
        name=name,
        surface_type=kind,
        construction_name=construction,
        building_surface_name=wall,
    )


def _window(points=SOUTH_WINDOW, name="Win"):
    return _opening(name, "Window", "Zone_East_Wall_South", points, "Window_Const")


def _door(name="Door"):
    return _opening(
        name, "Door", "Zone_East_Wall_Internal", INTERNAL_DOOR, "Interior_Wall_Const"
    )


def test_reversed_vertex_order_is_corrected():
    idf = _model()

    [window], flipped = add_fenestration(idf, _window(SOUTH_WINDOW[::-1]))

    assert flipped
    assert window.vertices_as_tuples == SOUTH_WINDOW


@pytest.mark.parametrize(
    ("points", "reason"),
    [
        ([(x, 0.3, z) for x, _, z in SOUTH_WINDOW], "off the plane"),
        ([(x + 2.5, y, z) for x, y, z in SOUTH_WINDOW], "outside surface"),
    ],
    ids=["off_plane", "outside"],
)
def test_misplaced_window_is_rejected(points, reason):
    idf = _model()

    with pytest.raises(ValueError, match=reason):
        add_fenestration(idf, _window(points))
    assert "Win" not in idf.all_of_type(FenestrationSurfaceDetailed)


def test_window_with_opaque_construction_is_rejected():
    idf = _model()
    window = _window()
    window.construction_name = "Exterior_Wall_Const"

    with pytest.raises(ValueError, match="needs a window construction"):
        add_fenestration(idf, window)


def test_interzone_opening_gets_a_mirrored_partner():
    idf = _model()

    door, partner = add_fenestration(idf, _door())[0]

    assert partner.name == "Door" + PARTNER_SUFFIX
    assert partner.building_surface_name == "Zone_West_Wall_Internal"
    assert partner.vertices_as_tuples == door.vertices_as_tuples[::-1]
    assert door.outside_boundary_condition_object == partner.name
    assert partner.outside_boundary_condition_object == door.name


def test_interzone_opening_is_all_or_nothing():
    idf = _model()
    idf.add(
        _opening(
            "Door" + PARTNER_SUFFIX,
            "Window",
            "Zone_East_Wall_South",
            SOUTH_WINDOW,
            "Window_Const",
        )
    )

    with pytest.raises(DuplicateNameError):
        add_fenestration(idf, _door())
    assert "Door" not in idf.all_of_type(FenestrationSurfaceDetailed)


def test_pair_is_updated_and_removed_together():
    idf = _model()
    add_fenestration(idf, _door())
    idf.add(Construction(name="Heavy_Door", outside_layer="Concrete_20cm"))

    update_fenestration(
        idf, "Door", new_name="Hall_Door", construction_name="Heavy_Door"
    )
    partner = idf.get(FenestrationSurfaceDetailed, "Door" + PARTNER_SUFFIX)
    assert partner is not None
    assert partner.construction_name == "Heavy_Door"
    assert partner.outside_boundary_condition_object == "Hall_Door"

    assert remove_fenestration(idf, "Hall_Door") == [
        "Hall_Door",
        "Door" + PARTNER_SUFFIX,
    ]
    remaining = idf.all_of_type(FenestrationSurfaceDetailed)
    assert {"Hall_Door", "Door" + PARTNER_SUFFIX}.isdisjoint(remaining)


@pytest.mark.skipif(shutil.which("energyplus") is None, reason="EnergyPlus not on PATH")
def test_triple_glazing_and_paired_door_simulate(tmp_path):
    idf = _model()
    idf.add(
        WindowMaterialGlazing(
            name="Clear6",
            optical_data_type="SpectralAverage",
            thickness=0.006,
            solar_transmittance_at_normal_incidence=0.775,
            front_side_solar_reflectance_at_normal_incidence=0.071,
            back_side_solar_reflectance_at_normal_incidence=0.071,
            visible_transmittance_at_normal_incidence=0.881,
            front_side_visible_reflectance_at_normal_incidence=0.08,
            back_side_visible_reflectance_at_normal_incidence=0.08,
        )
    )
    idf.add(WindowMaterialGas(name="Argon12", gas_type="Argon", thickness=0.012))
    idf.add(
        Construction(
            name="Triple",
            outside_layer="Clear6",
            layer_2="Argon12",
            layer_3="Clear6",
            layer_4="Argon12",
            layer_5="Clear6",
        )
    )
    window = _window()
    window.construction_name = "Triple"
    add_fenestration(idf, window)
    add_fenestration(idf, _door())
    add_design_days(idf, DATA / "weather" / "Shenzhen.ddy")
    idf.save(tmp_path / "in.idf")

    result = run_energyplus(
        tmp_path / "in.idf", DATA / "weather" / "Shenzhen.epw", tmp_path
    )

    assert result.succeeded, result.errors


def _steel_door_model() -> IDF:
    """The two-zone model with a door construction whose layers differ."""
    idf = _model()
    idf.add(
        Material(
            name="Steel",
            roughness="Smooth",
            thickness=0.002,
            conductivity=45.0,
            density=7800.0,
            specific_heat=500.0,
        )
    )
    idf.add(
        Material(
            name="Foam",
            roughness="Rough",
            thickness=0.04,
            conductivity=0.03,
            density=30.0,
            specific_heat=1400.0,
        )
    )
    idf.add(Construction(name="SteelDoor", outside_layer="Steel", layer_2="Foam"))
    return idf  # fmt: skip


def test_interzone_partner_takes_the_construction_reversed():
    idf = _steel_door_model()
    door = _door()
    door.construction_name = "SteelDoor"

    _, partner = add_fenestration(idf, door)[0]

    assert partner.construction_name == "SteelDoor_Reversed"
    reverse = idf.all_of_type(Construction)["SteelDoor_Reversed"]
    assert (reverse.outside_layer, reverse.layer_2) == ("Foam", "Steel")


def test_construction_change_reaches_the_partner_reversed():
    idf = _steel_door_model()
    add_fenestration(idf, _door())

    door, partner = update_fenestration(idf, "Door", construction_name="SteelDoor")

    assert door.construction_name == "SteelDoor"
    assert partner.construction_name == "SteelDoor_Reversed"


@pytest.mark.skipif(shutil.which("energyplus") is None, reason="EnergyPlus not on PATH")
def test_asymmetric_paired_door_simulates_without_a_layer_order_warning(tmp_path):
    idf = _steel_door_model()
    door = _door()
    door.construction_name = "SteelDoor"
    add_fenestration(idf, door)
    add_design_days(idf, DATA / "weather" / "Shenzhen.ddy")
    idf.save(tmp_path / "in.idf")

    result = run_energyplus(
        tmp_path / "in.idf", DATA / "weather" / "Shenzhen.epw", tmp_path
    )

    assert result.succeeded, result.errors
    # Without the reversed partner: "does not have the same materials in the
    # reverse order as the construction ... of adjacent surface".
    assert not [m for m in result.messages if "reverse order" in m.text]


def _office(width: float = 10.0) -> IDF:
    idf = IDF()
    idf.add(Building(name="B"))
    idf.add(Zone(name="Office"))
    idf.add(
        WindowMaterialSimpleGlazingSystem(
            name="Glass", u_factor=2.0, solar_heat_gain_coefficient=0.4
        )
    )
    idf.add(Construction(name="Window", outside_layer="Glass"))
    idf.add(
        Material(
            name="Brick",
            roughness="Rough",
            thickness=0.2,
            conductivity=0.9,
            density=1900.0,
            specific_heat=800.0,
        )
    )
    idf.add(Construction(name="Ext", outside_layer="Brick"))
    create_zone_geometry(
        idf,
        "Office",
        [
            PlanPointSchema(X=x, Y=y)
            for x, y in [(0, 0), (width, 0), (width, 8), (0, 8)]
        ],
        3.0,
        3.5,
        ZoneConstructions("Ext", "Ext", "Ext", "Ext", "Ext"),
    )
    return idf


def test_ratio_windows_cover_the_ratio_on_the_chosen_facings():
    idf = _office()

    created, problems = add_windows_by_ratio(
        idf, None, ["South", "North"], 0.4, "Window", 0.8
    )

    assert problems == []
    assert len(created) == 2
    windows = idf.all_of_type(FenestrationSurfaceDetailed)
    for name in created:
        window = windows[name]
        wall = idf.get(BuildingSurfaceDetailed, window.building_surface_name)
        assert wall is not None
        assert window.area / wall.area == pytest.approx(0.4)
        assert placement_problem(window, wall) is None
        assert facing(wall) in ("South", "North")
        # Sill measured from the storey's floor at z = 3.0, window below the top.
        heights = [z for *_, z in window.vertices_as_tuples]
        assert min(heights) == pytest.approx(3.8)
        assert max(heights) <= 6.5


def test_ratio_too_large_for_a_wall_is_reported_not_raised():
    idf = _office(width=1.0)

    # The east wall is 8 m x 3.5 m: 99% leaves less than the edge clearance.
    created, problems = add_windows_by_ratio(
        idf, ["Office"], ["East"], 0.99, "Window", 0.8
    )

    assert created == []
    assert len(problems) == 1 and "does not fit" in problems[0]


def test_facing_follows_the_building_north_axis():
    idf = _office()
    south = next(
        s
        for s in idf.all_of_type(BuildingSurfaceDetailed).values()
        if s.surface_type == "Wall" and s.azimuth == 180.0
    )

    assert facing(south) == "South"
    assert facing(south, north_axis=90.0) == "West"
