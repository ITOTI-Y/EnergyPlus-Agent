import numpy as np
import pytest
from PIL import Image

from src.agent.segmentation import cut_out, silhouette


def _mask(rows: slice, columns: slice, shape=(100, 200)) -> np.ndarray:
    mask = np.zeros(shape, bool)
    mask[rows, columns] = True
    return mask


def test_a_tree_down_to_the_street_stays_part_of_the_building():
    building = _mask(slice(10, 90), slice(40, 120))
    # A tree hides columns 60-80 from row 50 down to the street.
    building[50:, 60:80] = False

    target = silhouette([building], [(30, 0, 130, 100)])

    assert target[10:90, 40:120].all()
    assert not target[90:].any() and not target[:, :40].any()


def test_neighbours_outside_the_boxes_are_left_out():
    tower = _mask(slice(10, 90), slice(40, 120))
    neighbour = _mask(slice(30, 100), slice(115, 200))

    target = silhouette([tower, neighbour], [(30, 0, 125, 100)])

    assert target[10:90, 40:115].all()
    assert not target[:, 120:].any()


def test_a_wide_box_keeps_only_the_building_most_inside_it():
    tower = _mask(slice(10, 90), slice(60, 140))
    # Mostly inside the box too, as a box around a wide canopy holds it.
    neighbour = _mask(slice(40, 100), slice(0, 50))

    target = silhouette([neighbour, tower], [(0, 0, 200, 100)])

    assert target[10:90, 60:140].all() and not target[:, :50].any()


def test_no_building_inside_the_boxes_is_an_error():
    with pytest.raises(ValueError, match="no building"):
        silhouette([_mask(slice(0, 50), slice(0, 50))], [(100, 0, 200, 100)])


def test_the_cut_out_is_cropped_and_transparent_outside():
    image = Image.new("RGB", (200, 100), "red")
    mask = _mask(slice(10, 90), slice(40, 120))
    mask[10:30, 100:120] = False

    cut = cut_out(image, mask)

    assert cut.mode == "RGBA" and cut.size == (80, 80)
    pixels = np.asarray(cut)
    assert pixels[5, 70, 3] == 0 and tuple(pixels[50, 10]) == (255, 0, 0, 255)
