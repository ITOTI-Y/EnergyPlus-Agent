import pytest
from shapely.geometry import Polygon

from src.modeling.zoning import MIN_CORE_WIDTH_M, PERIMETER_DEPTH_M, perimeter_core


def test_a_rectangle_splits_into_four_perimeter_zones_and_a_core():
    rectangle = [(0, 0), (30, 0), (30, 20), (0, 20)]

    zones = dict(perimeter_core(rectangle))

    assert list(zones) == ["S", "E", "N", "W", "Core"]
    polygons = {name: Polygon(corners) for name, corners in zones.items()}
    assert sum(p.area for p in polygons.values()) == pytest.approx(600)
    # Zones tile the plan without overlapping.
    for a in polygons:
        for b in polygons:
            if a < b:
                assert polygons[a].intersection(polygons[b]).area == pytest.approx(0)
    d = PERIMETER_DEPTH_M
    assert polygons["Core"].bounds == pytest.approx((d, d, 30 - d, 20 - d))
    assert polygons["S"].bounds == pytest.approx((0, 0, 30, d))
    assert all(Polygon(c).exterior.is_ccw for c in zones.values())


def test_clockwise_corners_give_the_same_zones():
    clockwise = [(0, 0), (0, 20), (30, 20), (30, 0)]
    counterclockwise = [(0, 0), (30, 0), (30, 20), (0, 20)]

    def areas(corners):
        return {n: Polygon(c).area for n, c in perimeter_core(corners)}

    assert areas(clockwise) == pytest.approx(areas(counterclockwise))


def test_a_rotated_rectangle_names_sides_by_where_they_face():
    # 30 x 20 m turned 30 degrees counterclockwise.
    rotated = [(0, 0), (25.98, 15), (15.98, 32.32), (-10, 17.32)]

    names = [name for name, _ in perimeter_core(rotated)]

    assert sorted(names) == ["Core", "E", "N", "S", "W"]


@pytest.mark.parametrize(
    ("corners", "message"),
    [
        ([(0, 0), (30, 0), (30, 20)], "rectangle"),
        ([(0, 0), (30, 0), (25, 20), (0, 20)], "right angles"),
        ([(0, 0), (30, 0), (30, 12), (0, 12)], "too short"),
    ],
)
def test_plans_that_cannot_be_split_are_refused(corners, message):
    with pytest.raises(ValueError, match=message):
        perimeter_core(corners)


def test_the_narrowest_plan_that_splits_leaves_the_minimum_core():
    side = 2 * PERIMETER_DEPTH_M + MIN_CORE_WIDTH_M
    zones = dict(perimeter_core([(0, 0), (side, 0), (side, side), (0, side)]))

    assert Polygon(zones["Core"]).area == pytest.approx(MIN_CORE_WIDTH_M**2)
