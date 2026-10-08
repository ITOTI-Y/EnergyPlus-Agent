import pytest
from idfpy import IDF
from idfpy.models.constructions import Construction, Material
from idfpy.models.thermal_zones import Zone
from pydantic import ValidationError

from src.modeling import objects
from src.modeling.errors import (
    DuplicateNameError,
    MissingReferenceError,
    ReferencedObjectError,
)


def _brick(name: str = "Brick") -> Material:
    return Material(
        name=name,
        roughness="Rough",
        thickness=0.1,
        conductivity=0.9,
        density=1900.0,
        specific_heat=800.0,
    )


def _wall_model() -> IDF:
    idf = IDF()
    idf.add(_brick())
    idf.add(Construction(name="Wall", outside_layer="Brick"))
    return idf


def test_create_rejects_missing_reference_and_leaves_model_unchanged():
    idf = IDF()

    with pytest.raises(MissingReferenceError) as caught:
        objects.create(idf, Construction(name="Wall", outside_layer="Brick"))

    assert caught.value.data is not None
    [missing] = caught.value.data["missing_references"]
    assert "Brick" in missing
    assert not idf.all_of_type(Construction)


def test_create_rejects_duplicate_name():
    idf = IDF()
    objects.create(idf, Zone(name="Z1"))

    with pytest.raises(DuplicateNameError):
        objects.create(idf, Zone(name="Z1"))


def test_update_rename_cascades_to_references():
    idf = _wall_model()

    objects.update(idf, objects.get(idf, Material, "Brick"), {"name": "Clay_Brick"})

    assert objects.get(idf, Construction, "Wall").outside_layer == "Clay_Brick"


def test_update_rejects_bad_value_without_partial_changes():
    idf = _wall_model()
    brick = objects.get(idf, Material, "Brick")

    with pytest.raises(ValidationError):
        objects.update(idf, brick, {"thickness": 0.2, "roughness": "Bumpy"})

    assert brick.thickness == 0.1


def test_update_rejects_missing_reference():
    idf = _wall_model()
    wall = objects.get(idf, Construction, "Wall")

    with pytest.raises(MissingReferenceError):
        objects.update(idf, wall, {"outside_layer": "Concrete"})

    assert wall.outside_layer == "Brick"


def test_update_rejects_taken_name():
    idf = _wall_model()
    idf.add(_brick("Stone"))

    with pytest.raises(DuplicateNameError):
        objects.update(idf, objects.get(idf, Material, "Brick"), {"name": "Stone"})


def test_delete_refuses_referenced_object():
    idf = _wall_model()

    with pytest.raises(ReferencedObjectError) as caught:
        objects.delete(idf, objects.get(idf, Material, "Brick"), "Brick")

    assert caught.value.data == {"references": ["Construction:Wall"]}
    assert idf.get(Material, "Brick") is not None


def test_delete_removes_unreferenced_object():
    idf = _wall_model()

    objects.delete(idf, objects.get(idf, Construction, "Wall"), "Wall")

    assert idf.get(Construction, "Wall") is None


def test_delete_allowed_once_the_reference_moved_away():
    # Needs idfpy >= 26.1.4: earlier versions kept the old referrer indexed.
    idf = _wall_model()
    idf.add(_brick("Stone"))

    objects.update(
        idf, objects.get(idf, Construction, "Wall"), {"outside_layer": "Stone"}
    )
    objects.delete(idf, objects.get(idf, Material, "Brick"), "Brick")

    assert idf.get(Material, "Brick") is None
